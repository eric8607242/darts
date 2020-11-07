import os

import logging
import json

import numpy as np
import torch


class BackbonePool:
    def __init__(self, MAX_EDGES, SEACH_SPACE, lookup_table, arch_param_nums, xargs):
        self.arch_param_nums = arch_param_nums 
        self.xargs = xargs
        self.MAX_EDGES = MAX_EDGES
        self.SEACH_SPACE = SEACH_SPACE

        if os.path.exists(self.xargs.path_to_backbone_pool):
            logging.info("Load backbone pool from {}".format(self.xargs.path_to_backbone_pool))
            self.backbone_pool = self._load_backbone_pool()
        else:
            logging.info("Generate backbone pool")
            self.backbone_pool = self._generate_backbone_pool(lookup_table, 10)

    def get_backbone(self, macs):
        backbone_keys = np.array([int(k) for k in self.backbone_pool.keys()])
        backbone_diff = np.absolute(backbone_keys - macs)

        backbone_index = backbone_diff.argmin()
        backbone = self.backbone_pool[str(backbone_keys[backbone_index])]

        return torch.Tensor(backbone)

    def get_backbone_keys(self):
        return self.backbone_pool.keys()

    def _load_backbone_pool(self):
        backbone_pool = None

        with open(self.xargs.path_to_backbone_pool) as f:
            backbone_pool = json.load(f)
        return backbone_pool

    def save_backbone_pool(self, path_to_backbone_pool, backbone_pool=None):
        if backbone_pool is None:
            backbone_pool = self.backbone_pool

        with open(path_to_backbone_pool, "w") as f:
            json.dump(backbone_pool, f)

    def _generate_backbone_pool(self,lookup_table, bias=10):
        backbone_pool = {}

        low_macs = self.xargs.low_macs
        high_macs = self.xargs.high_macs
        pool_interval = (high_macs - low_macs)//(1+1)

        for mac in range(low_macs+pool_interval, high_macs-1, pool_interval):
            gen_mac, arch_param = self.generate_arch_param(lookup_table)
            
            while gen_mac > mac + bias or gen_mac < mac - bias:
                gen_mac, arch_param = self.generate_arch_param(lookup_table)

            backbone_pool[str(mac)] = arch_param.tolist()
            logging.info("Target mac {} : Backbone generate {}".format(mac, gen_mac))

        self.save_backbone_pool(self.xargs.path_to_backbone_pool, backbone_pool=backbone_pool)

        return backbone_pool

    def generate_arch_param(self, lookup_table, p=False, skip=False):
        arch_param = torch.zeros(self.MAX_EDGES, len(self.SEACH_SPACE))
        for a in arch_param:
            index = np.random.randint(low=0, high=len(self.SEACH_SPACE))
            a[index] = 1

        mac = lookup_table.get_model_macs(arch_param.cuda())
        return mac, arch_param
