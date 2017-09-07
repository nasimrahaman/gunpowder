import logging
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict

logger = logging.getLogger(__name__)


class NetIoWrapper(object):
    def __init__(self, net, names_net_outputs, use_gpu=None):
        # Privates
        self._input_cache = OrderedDict()
        self._output_cache = OrderedDict()
        self._net_output_names = list(names_net_outputs)
        # Publics
        self.net = net
        self.use_gpu = use_gpu

    def to_device(self, tensor):
        if self.use_gpu:
            assert self.use_gpu == 0, "can only use GPU0, try setting " \
                                      "CUDA_VISIBLE_DEVICES instead."
            return tensor.cuda()
        else:
            return tensor.cpu()

    def wrap(self, array):
        return Variable(self.to_device(torch.from_numpy(array.copy()).contiguous()),
                        requires_grad=False, volatile=True)

    def unwrap(self, variable):
        return variable.data.cpu().numpy()

    def set_inputs(self, data):
        # Convert to torch tensors
        data = OrderedDict([(name, self.wrap(array)) for name, array in data.items()])
        # Clear and update
        self._input_cache.clear()
        self._input_cache.update(data)

    def forward(self):
        outputs = self.net(*self._input_cache.values())
        # Update output cache
        self._output_cache.clear()
        self._output_cache\
            .update(OrderedDict([(name, self.unwrap(output))
                                 for name, output in zip(self._net_output_names, outputs)]))

    def get_outputs(self):
        return self._output_cache

