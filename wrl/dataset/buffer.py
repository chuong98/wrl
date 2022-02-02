from ..builder import BUFFERS 

from tianshou.data import (
    ReplayBuffer, VectorReplayBuffer,  
    PrioritizedReplayBuffer, PrioritizedVectorReplayBuffer
)

BUFFERS.register_module(ReplayBuffer)
BUFFERS.register_module(VectorReplayBuffer)
BUFFERS.register_module(PrioritizedReplayBuffer)
BUFFERS.register_module(PrioritizedVectorReplayBuffer)
