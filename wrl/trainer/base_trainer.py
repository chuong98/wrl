from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, Optional, Union
import torch 
import numpy as np
import os.path as osp
import time, tqdm, os
from mmcv import fileio
from collections import defaultdict

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import gather_info, test_episode
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config
from ..builder import build_rlhook

class BaseTrainner(metaclass=ABCMeta):
    def __init__(self, 
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Optional[Collector],
        max_epoch: int,
        step_per_epoch: int,
        step_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        update_per_step: Union[int, float] = 1,
        logger: BaseLogger = LazyLogger(),
        verbose: bool = True,
        test_in_train: bool = True,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        train_hook: Optional[Dict] = None,
        test_hook: Optional[Dict] = None,
        early_stop=True,
        save_policy=True,
        ckpt_cfg=dict(interval=1, max_to_keep=2),
        reward_threshold: Optional[float]=None,
        work_dir: Optional[str]=None,
        resume_epoch: Optional[int]=None,
        ):

        self.policy = policy
        self.train_collector = train_collector
        self.test_collector = test_collector 
        self.logger = logger
        self.verbose = verbose 
        self.resume_epoch= resume_epoch
        self.reward_metric = reward_metric 

        self.test_in_train = test_in_train and (
            train_collector.policy == policy and test_collector is not None)

        self._max_epoch = max_epoch 
        self._step_per_epoch = step_per_epoch
        self._step_per_collect = step_per_collect 
        self._episode_per_test = episode_per_test
        self._batch_size = batch_size
        self._update_per_step = update_per_step 
 
        self.train_hook = train_hook 
        self.test_hook = test_hook 

        self.with_early_stop = early_stop
        self.with_save_policy = save_policy
        self.reward_threshold = reward_threshold
        self.ckpt_cfg = ckpt_cfg
        self.work_dir = work_dir

        if train_hook:
            self.train_hook = build_rlhook(train_hook, default_args=dict(policy=self.policy))
        if test_hook:
            self.test_hook = build_rlhook(test_hook, default_args=dict(policy=self.policy))

    def train(self):
        self.before_run()
        for epoch in range(1 + self.start_epoch, 1 + self._max_epoch):
            self.before_epoch(epoch)
            with tqdm.tqdm(
                total=self._step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config
            ) as t:
                while t.n < t.total:
                    self.run_iter(epoch, t)
                    t.update()
            self.end_epoch(epoch)
            if self.with_early_stop and self.early_stop(self.best_reward):
                break
        return self.end_run()

    def before_run(self):
        if self.resume_epoch:
            self.resume_from_checkpoint(self.resume_epoch)
        else:
            self.start_epoch, self.env_step, self.gradient_step = 0, 0, 0
            
        self.last_rew, self.last_len = 0.0, 0
        self.stat: Dict[str, MovAvg] = defaultdict(MovAvg)
        self.start_time = time.time()
        self.train_collector.reset_stat()

        if self.test_collector is not None:
            self.test_collector.reset_stat()
            test_result = test_episode(
                self.policy, self.test_collector, self.test_hook.run, self.start_epoch, 
                self._episode_per_test, self.logger, self.env_step,
                self.reward_metric
            )
            self.best_epoch = self.start_epoch
            self.best_reward, self.best_reward_std = test_result["rew"], test_result["rew_std"]

    def end_run(self):
        if self.test_collector is None and self.with_save_policy:
            self.save_policy()
        if self.test_collector is None:    
            return gather_info(self.start_time, self.train_collector, None, 0.0, 0.0)
        else:
            return gather_info(
                self.start_time, self.train_collector, self.test_collector, 
                self.best_reward, self.best_reward_std)

    def before_epoch(self, epoch):
        self.policy.train()

    @abstractmethod
    def run_iter(self, t):
        pass

    def end_epoch(self, epoch):
        self.logger.save_data(epoch, self.env_step, self.gradient_step, self.save_checkpoint) 
        # test
        if self.test_collector is not None:
            test_result = test_episode(
                self.policy, self.test_collector, self.test_hook.run, 
                epoch, self._episode_per_test, self.logger, self.env_step,
                self.reward_metric
            )
            rew, rew_std = test_result["rew"], test_result["rew_std"]
            if self.best_epoch < 0 or self.best_reward < rew:
                self.best_epoch, self.best_reward, self.best_reward_std = epoch, rew, rew_std
                if self.with_save_policy:
                    self.save_policy()
            if self.verbose:
                print(
                    f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
                    f"ard: {self.best_reward:.6f} ± {self.best_reward_std:.6f} in #{self.best_epoch}"
                )

    # ----------------------Save/Stop/Resume training ----------------------
    def save_policy(self):
        torch.save(self.policy.state_dict(), 
                    os.path.join(self.work_dir, 'policy.pth'))

    def early_stop(self, reward):
        return reward > self.reward_threshold

    def save_checkpoint(self, epoch, env_step, gradient_step):
        torch.save(
            {
                'model': self.policy.state_dict(),
                'optim': self.policy.optim.state_dict(),
            }, osp.join(self.work_dir, 
                            f'ckpt_{epoch}.pth')
        )
        fileio.dump(
            {   
                'buffer':self.train_collector.buffer,
                'epoch': epoch,
                'env_step': env_step,
                'gradient_step': gradient_step,
            }, osp.join(self.work_dir, f'buffer_{epoch}.pkl')
        )
        # remove other checkpoints
        max_keep_ckpts = self.ckpt_cfg.get("max_keep_ckpts", 5)
        interval = self.ckpt_cfg.get("interval", 1)
        if max_keep_ckpts > 0:
            redundant_ckpts = range(
                epoch - max_keep_ckpts * interval, 0,
                -interval)
            for _step in redundant_ckpts:
                ckpt_path = osp.join(self.work_dir, f'ckpt_{_step}.pth')
                buffer_path = osp.join(self.work_dir, f'buffer_{_step}.pkl')    
                if osp.exists(ckpt_path):
                    os.remove(ckpt_path)
                if osp.exists(buffer_path):
                    os.remove(buffer_path)

    def resume_from_checkpoint(self, epoch):
        print(f"Loading agent under {self.work_dir}")
        ckpt_path = osp.join(self.work_dir, f'ckpt_{epoch}.pth')
        buffer_path = osp.join(self.work_dir, f'buffer_{epoch}.pkl')                   
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.policy.model.device)
            self.policy.load_state_dict(checkpoint['model'])
            self.policy.optim.load_state_dict(checkpoint['optim'])
            print(f"Successfully restore policy and optim from {ckpt_path}.")
        else:
            print(f"Fail to restore policy and optimizer from {ckpt_path}.")
        if os.path.exists(buffer_path):
            result = fileio.load(buffer_path)
            self.train_collector.buffer = result['buffer']
            self.start_epoch = result['epoch'] -1
            self.env_step = result['env_step']
            self.gradient_step = result['gradient_step']
            print(f"Successfully restore buffer from {buffer_path}.")
            print(f"Resume from epoch: {self.start_epoch+1}, env_step: {self.env_step}, gradient_step: {self.gradient_step}")
        else:
            print(f"Fail to restore buffer from {buffer_path}.")
        
    
