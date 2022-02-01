import gym 

def build_venv(cfg):
    env_name = cfg.type
    env = gym.make(env_name)
    if cfg.multi_process is not None:
        from tianshou.env import venvs as ts_venvs
        venv = getattr(ts_venvs,cfg.multi_process.type)
        train_env = venv([lambda: gym.make(env_name) for 
                                _ in range(cfg.multi_process.train_num)])
        test_env = venv([lambda: gym.make(env_name) for 
                                _ in range(cfg.multi_process.test_num)])
    else:
        train_env = gym.make(env_name)
        test_env = gym.make(env_name)
    return train_env, test_env, env