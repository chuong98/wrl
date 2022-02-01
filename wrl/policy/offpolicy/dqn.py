from copy import deepcopy
from typing import Any
from tianshou.policy.modelfree.dqn import DQNPolicy as DQNPolicy_TS

from ...builder import POLICIES, build_network
from mmcv.runner.optimizer import build_optimizer

@POLICIES.register_module()
class DQNPolicy(DQNPolicy_TS):
    def __init__(self,
                state_shape,
                action_shape,
                network = dict(type='MLPNet'),
                optim = dict(type='AdamW', lr=1e-3),
                discount_factor: float = 0.99,
                estimation_step: int = 1,
                target_update_freq: int = 0,
                reward_normalization: bool = False,
                is_double: bool = True,
                **kwargs: Any,
                ):
        super(DQNPolicy_TS, self).__init__(**kwargs)
        network_cfg = network.copy()
        network_cfg['in_channels']=state_shape[0]
        network_cfg['out_channels']=action_shape
        self.model = build_network(network_cfg)
        self.optim = build_optimizer(self.model, optim)
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double