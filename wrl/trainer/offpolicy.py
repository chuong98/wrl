from .base_trainer import BaseTrainner
from tianshou.trainer import gather_info, test_episode
from ..builder import TRAINERS

@TRAINERS.register_module()
class OffPolicyTrainer(BaseTrainner):

    def run_iter(self, epoch, t):
        if self.train_hook is not None:
            self.train_hook.run(epoch, self.env_step)

        # Collect data
        result = self.train_collector.collect(n_step=self._step_per_collect)
        if result["n/ep"] > 0 and self.reward_metric:
            rew = self.reward_metric(result["rews"])
            result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())

        self.env_step += int(result["n/st"])
        t.update(result["n/st"])
        self.logger.log_train_data(result, self.env_step)
        if result["n/ep"] > 0:
            self.last_rew = result['rew']  
            self.last_len = result['len'] 
        data = {
            "env_step": str(self.env_step),
            "rew": f"{self.last_rew:.2f}",
            "len": str(int(self.last_len)),
            "n/ep": str(int(result["n/ep"])),
            "n/st": str(int(result["n/st"])),
        }

        # Test in train
        if result["n/ep"] > 0:
            if self.test_in_train and self.with_early_stop and self.early_stop(result["rew"]):
                test_result = test_episode(
                    self.policy, self.test_collector, self.test_hook.run, epoch, 
                    self._episode_per_test, self.logger, self.env_step
                )
                if self.early_stop(test_result["rew"]):
                    if self.with_save_policy:
                        self.save_policy()
                    self.logger.save_data(
                        epoch, self.env_step, self.gradient_step, self.save_checkpoint
                    )
                    t.set_postfix(**data)
                    return gather_info(
                        self.start_time, self.train_collector, self.test_collector,
                        test_result["rew"], test_result["rew_std"]
                    )
                else:
                    self.policy.train()

        # Update Policy 
        for _ in range(round(self._update_per_step * result["n/st"])):
            self.gradient_step += 1
            losses = self.policy.update(self._batch_size, self.train_collector.buffer)
            for k in losses.keys():
                self.stat[k].add(losses[k])
                losses[k] = self.stat[k].get()
                data[k] = f"{losses[k]:.3f}"
            self.logger.log_update_data(losses, self.gradient_step)
            t.set_postfix(**data)