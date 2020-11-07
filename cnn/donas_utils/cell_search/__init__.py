##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# The macro structure is defined in NAS-Bench-201
from .search_model_darts    import TinyNetworkDarts
from .search_model_setn     import TinyNetworkSETN
from .generic_model         import GenericNAS201Model
from .genotypes             import Structure as CellStructure, architectures as CellArchitectures


nas201_super_nets = {'DARTS-V1': TinyNetworkDarts,
                     "SETN": TinyNetworkSETN}
