from mmcv.utils import Registry, build_from_cfg


NETWORKS = Registry('network')
POLICIES = Registry('policy')
BUFFERS = Registry('buffer')
TRAINERS = Registry('trainer')
RLHOOKS = Registry('rlhook')

def build_network(cfg, default_args=None):
    return build_from_cfg(cfg, NETWORKS, default_args)

def build_policy(cfg, default_args=None):
    return build_from_cfg(cfg, POLICIES, default_args)

def build_buffer(cfg, default_args=None):
    return build_from_cfg(cfg, BUFFERS, default_args)

def build_trainer(cfg, default_args=None):
    return build_from_cfg(cfg, TRAINERS, default_args)

def build_rlhook(cfg, default_args=None):
    return build_from_cfg(cfg, RLHOOKS, default_args)