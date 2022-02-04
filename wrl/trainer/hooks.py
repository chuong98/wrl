from ..builder import RLHOOKS 
from tianshou.policy import BasePolicy

        
@RLHOOKS.register_module()
class EpsConstHook():
    def __init__(self, eps, policy: BasePolicy):
        self.policy = policy
        self.eps = eps

    def run(self, epoch=None, env_step=None):
        self.policy.set_eps(self.eps)

@RLHOOKS.register_module()
class EpsLinearDecayHook():
    def __init__(self,  eps_start, eps_final, decay_steps, policy: BasePolicy):
        self.policy = policy
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.decay_steps = decay_steps

    def run(self, epoch=None, env_step=None):
        if env_step <= self.decay_steps:
            eps = self.eps_start - env_step / self.decay_steps * \
                (self.eps_start - self.eps_final)
        else:
            eps = self.eps_final
        self.policy.set_eps(eps)

@RLHOOKS.register_module()
class EpsExpDecayHook():
    def __init__(self,  eps_start, eps_final, decay_rate, policy: BasePolicy):
        self.policy = policy
        self.eps_start = eps_start
        self.eps_final = eps_final
        assert decay_rate > 0 and decay_rate < 1, "decay_rate must be (0, 1)"
        self.decay_rate = decay_rate

    def run(self, epoch=None, env_step=None):
        eps = max(self.eps_start * self.decay_rate**env_step, self.eps_final)
        self.policy.set_eps(eps)
