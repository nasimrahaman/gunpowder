import logging
import multiprocessing
import os
import time
import torch

from gunpowder.pytorch.net_io_wrapper import NetIoWrapper
from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.producer_pool import ProducerPool, WorkersDied
from gunpowder.volume import Volume
from gunpowder.volume_spec import VolumeSpec

logger = logging.getLogger(__name__)


class PredictProcessDied(Exception):
    pass


class Predict(BatchFilter):
    """Augments a batch with network predictions.

    Args:

        inputs (dict): Dictionary from :class:``VolumeType`` to the names of
            input layers in the network.

        outputs (dict): Dictionary from :class:``VolumeType`` to the names of
            output layers in the network. New volumes will be generated by this
            node for each entry (if requested downstream). Set the resolution of
            the new volume via parameter ``output_resolutions``.

        volume_specs (dict, optional): An optional dictionary of
            :class:`VolumeType` to :class:`VolumeSpec` to set the volume specs
            generated volumes (``outputs``). This is useful to set the
            ``voxel_size``, for example, if they differ from the voxel size of
            the input volumes. Only fields that are not ``None`` in the given
            :class:`VolumeSpec` will be used.

        use_gpu (int): Which GPU to use. Set to ``None`` for CPU mode.
    """

    def __init__(
            self,
            model_file,
            inputs,
            outputs,
            volume_specs=None,
            use_gpu=None):

        assert os.path.isfile(model_file), "{} does not exist".format(model_file)

        # start prediction as a producer pool, so that we can gracefully exit if
        # anything goes wrong
        self.worker = ProducerPool([lambda gpu=use_gpu: self.__predict(gpu)], queue_size=1)
        self.batch_in = multiprocessing.Queue(maxsize=1)

        self.model_file = model_file
        self.net_initialized = False
        self.inputs = inputs
        self.outputs = outputs
        self.volume_specs = {} if volume_specs is None else volume_specs

    def setup(self):

        # get common voxel size of inputs, or None if they differ
        common_voxel_size = None
        for identifier in self.inputs:

            voxel_size = self.spec[identifier].voxel_size

            if common_voxel_size is None:
                common_voxel_size = voxel_size
            elif common_voxel_size != voxel_size:
                common_voxel_size = None
                break

        # announce provided outputs
        for identifier in self.outputs.keys():

            if identifier in self.volume_specs:
                spec = self.volume_specs[identifier].copy()
            else:
                spec = VolumeSpec()

            if spec.voxel_size is None:
                assert common_voxel_size is not None, (
                    "There is no common voxel size of the inputs, and no "
                    "VolumeSpec has been given for %s that defines "
                    "voxel_size." % identifier)

                spec.voxel_size = common_voxel_size

            if spec.interpolatable is None:
                # default for predictions
                spec.interpolatable = False

            self.provides(identifier, spec)

        self.worker.start()

    def teardown(self):
        self.worker.stop()

    def prepare(self, request):

        # remove request parts that we provide
        for volume_type in self.outputs.keys():
            if volume_type in request:
                del request[volume_type]

    def process(self, batch, request):

        self.batch_in.put((batch, request))

        try:
            out = self.worker.get()
        except WorkersDied:
            raise PredictProcessDied()

        for volume_type in self.outputs.keys():
            if volume_type in request:
                batch.volumes[volume_type] = out.volumes[volume_type]

    def __predict(self, use_gpu):

        if not self.net_initialized:

            logger.info("Initializing model...")
            self.net = torch.load(self.model_file).eval()

            if use_gpu is not None:
                logger.debug("Predict process: using GPU %d" % use_gpu)
                self.net.cuda()

            self.net_io = NetIoWrapper(self.net, self.outputs.values(), use_gpu)
            self.net_initialized = True

        start = time.time()

        batch, request = self.batch_in.get()

        fetch_time = time.time() - start

        self.net_io.set_inputs({input_name: batch.volumes[volume_type].data
                                for volume_type, input_name in self.inputs.items()
                                })

        self.net_io.forward()
        output = self.net_io.get_outputs()

        predict_time = time.time() - start

        logger.info("Predict process: time=%f (including %f waiting for batch)" % (
        predict_time, fetch_time))

        for volume_type, output_name in self.outputs.items():
            spec = self.spec[volume_type].copy()
            spec.roi = request[volume_type].roi
            batch.volumes[volume_type] = Volume(
                output[output_name][0],  # strip #batch dimension
                spec)

        return batch
