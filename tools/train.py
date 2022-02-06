import argparse
import os.path as osp
import torch
import numpy as np
import random
import pprint
from wrl.builder import build_buffer, build_policy, build_trainer
from wrl.envs.venvs import build_venv
from mmcv.utils import Config, DictAction, mkdir_or_exist, import_modules_from_strings
import tianshou as ts

from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-epoch', help='the epoch to resume from')
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
        help='the sleep time between rendering consecutive frames. Eg. Set 1/24 for fps=24')
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

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dir',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_epoch is not None:
        cfg.resume_epoch = args.resume_epoch

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

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
    train_envs,test_envs, env = build_venv(cfg.env, args.seed)
    print('Reward Threshold:', env.spec.reward_threshold)

    # Build The Agent (Policy)
    cfg.agent.state_shape = env.observation_space.shape or env.observation_space.n
    cfg.agent.action_shape= env.action_space.shape or env.action_space.n
    agent = build_policy(cfg.agent)

    # Build Collector
    train_collector = ts.data.Collector(agent, train_envs, 
                        buffer=build_buffer(cfg.collector.train_buffer), 
                        exploration_noise=cfg.collector.exploration_noise.train)
    test_collector = ts.data.Collector(agent, test_envs,
                        exploration_noise=cfg.collector.exploration_noise.test)  

    # We may need to collect some random data before training
    precollected_steps = cfg.collector.get('precollected_steps', 0)
    if precollected_steps>0:
        train_collector.collect(n_steps=precollected_steps)

    # Trainer
    logger = ts.utils.TensorboardLogger(SummaryWriter(f'{cfg.work_dir}/log'))
    trainer = build_trainer(
        cfg.trainer,
        default_args=dict(
            policy=agent, 
            train_collector=train_collector, 
            test_collector=test_collector, 
            logger=logger,
            reward_threshold = env.spec.reward_threshold,
            work_dir = cfg.work_dir,
            resume_epoch = cfg.resume_epoch,)
    )

    result = trainer.train()
    print(f'Finished training! Use {result["duration"]}')

    # Let's watch its performance!
    pprint.pprint(result)
    agent.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=len(test_envs), render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

if __name__ == '__main__':
    main()    
    