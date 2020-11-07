import logging
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from donas_utils.countmacs import MAC_Counter
from genotypes import PRIMITIVES

class Flatten(nn.Module):
    def forward(self, x):
        return x.mean(3).mean(2)

class LookUpTable:
    """
    LookupTable which saved the macs of each block.
    Calculate the approximate macs of arch param(probability arch param or one hot arch param).
    """
    def __init__(self, op_names, max_nodes, dataset):
        self.max_nodes = max_nodes
        self.op_names = op_names
        self.dataset = dataset

        self.edges = []
        for i in range(1, self.max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                self.edges.append(node_str)
        self.edges_key = sorted(self.edges)
        self.edge2index = {key:i for i, key in enumerate(self.edges_key)}

        self.basic_macs = self._calculate_basic_operations()
        self.search_cell_macs = self._calculate_cell_operations()


    def get_model_macs(self, arch_param):
        """
        Arg:
            arch_param(layer_nums, (split_block**2+1)*expansion_nums)
        """
        model_macs = []
        for i in range(1, self.max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = arch_param[self.edge2index[node_str]]
                for o_i, op_name in enumerate(self.op_names):
                    model_macs.append(self.cell_masc[node_str+op_name] * weights[o_i])

        return torch.sum(torch.stack(model_macs)) + self.basic_macs

    def _calculate_cell_operations(self, write_to_file=None):
        N = 5
        channel_list = [16, 32, 64]
        if self.dataset == "ImageNet-16-120":
            input_list = [16, 8, 4]
        else:
            input_list = [32, 16, 8]
        self.cell_masc = {}
        for c, input_size in zip(channel_list, input_list):
            for i in range(1, self.max_nodes):
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    for op_name in self.op_names:
                        if op_name not in ["none", "skip_connect", "avg_pool_3x3"]:
                            op = OPS[op_name](c, c, 1, False, True)
                            if node_str+op_name in self.cell_masc:
                                self.cell_masc[node_str+op_name] += self._calculate_macs(op, c, input_size)*N
                            else:
                                self.cell_masc[node_str+op_name] = self._calculate_macs(op, c, input_size)*N
                        else:
                            self.cell_masc[node_str+op_name] = 0

    def _calculate_basic_operations(self, write_to_file=None):
        model_macs = 0

        first = nn.Sequential(
                        nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(16))
                                        

        model_macs += self._calculate_macs(first, 3, 32) 

        reduction = ResNetBasicblock(16, 32, 2)
        model_macs += self._calculate_macs(reduction, 16, 32)

        reduction = ResNetBasicblock(32, 64, 2)
        model_macs += self._calculate_macs(reduction, 32, 16)

        return model_macs

    def _calculate_macs(self, model, input_channel, input_size):
        counter = MAC_Counter(model, [1, input_channel, input_size, input_size])
        macs = counter.print_summary()["total_gmacs"]*1000
        return macs

    def get_validation_arch_param(self, arch_param):
        one_hot_arch_param = torch.zeros(arch_param.shape)

        for a, oa in zip(arch_param, one_hot_arch_param):
            oa[torch.argmax(a).item()] = 1
        return one_hot_arch_param

        


