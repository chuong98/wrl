# from abc import ABCMeta, abstractmethod
# from tianshou.data import Collector
# from tianshou.policy import BasePolicy
# from typing import Callable, Dict, Optional, Union

# class BaseRunner(metaclass=ABCMeta):
#     def __init__(self, 
#         policy: BasePolicy,
#         train_collector: Collector,
#         test_collector: Optional[Collector],
#         max_epoch: int,
#         step_per_epoch: int,
#         step_per_collect: int,
#         episode_per_test: int,
#         batch_size: int,
#         update_per_step: Union[int, float] = 1):
