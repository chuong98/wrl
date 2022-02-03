import argparse
import torch
import numpy as np
import random
import gym
from wrl.builder import build_policy
from mmcv.utils import Config, DictAction, import_modules_from_strings
import tianshou as ts


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('ckpt', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--render', type=float, 
        help='the sleep time between rendering consecutive frames.')
    parser.add_argument('--gpus', type=int, default=1, help='numer of gpus')
    args = parser.parse_args()

    return args

def parse_cfg(args):
    """
        Some lengthy setup (log info, resume ckpt) from mmcv config: 
    """
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    
    # CUDA devices
    cfg.agent.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.agent.gpus = args.gpus

    return cfg


def main():
    args = parse_args()
    cfg = parse_cfg(args)
    
    # Build Gym Env 
    env = gym.make(cfg.env.type)
    env.seed(args.seed)

    # Build The Agent (Policy)
    cfg.agent.state_shape = env.observation_space.shape or env.observation_space.n
    cfg.agent.action_shape= env.action_space.shape or env.action_space.n
    agent = build_policy(cfg.agent)
    agent.load_state_dict(torch.load(args.ckpt))
    agent.eval()
    agent.set_eps(cfg.exploration.eps_test)

    # Build Collector
    test_collector = ts.data.Collector(agent, env,
                        exploration_noise=cfg.exploration.eps_test>0)  

    # Let's watch its performance!
    result = test_collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

if __name__ == '__main__':
    main()    
    