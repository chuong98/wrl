from mmcv.utils import Registry, build_from_cfg


NETWORKS = Registry('network')
POLICIES = Registry('policy')
BUFFERS = Registry('buffer')
TRAINERS = Registry('trainer')

def build_network(cfg, default_args=None):
    return build_from_cfg(cfg, NETWORKS, default_args)

def build_policy(cfg, default_args=None):
    return build_from_cfg(cfg, POLICIES, default_args)

def build_buffer(cfg, default_args=None):
    return build_from_cfg(cfg, BUFFERS, default_args)