num_parallel_envs=10
env = dict(
    type='CartPole-v0',
    multi_process=dict(
        type='DummyVectorEnv',
        # type='SubprocVectorEnv',
        train_num=num_parallel_envs,
        test_num=100
    )
)

agent = dict(
    type='DQNPolicy',
    network=dict(
        type='MLPNet', 
        hidden_layers=[64,64,64],
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        drop_rate=0.05),
    optim = dict(type='Adam', lr=1e-3),
    discount_factor=0.99,
    estimation_step = 3,
    target_update_freq = 320,
    reward_normalization = False,
    is_double=True,
)

collector=dict(
    train_buffer=dict(
        type='VectorReplayBuffer',
        total_size=20000, 
        buffer_num=num_parallel_envs),
    exploration_noise=dict(train=True,test=True)
)

trainer=dict(
    type='OffPolicyTrainer',
    max_epoch=10, 
    step_per_epoch= 10000,
    step_per_collect = 10,
    episode_per_test=5,
    batch_size=64,
    update_per_step=1,
    train_hook=dict(type='EpsConstHook', eps=0.1),
    test_hook=dict(type='EpsConstHook', eps=0.05),   
    ckpt_cfg=dict(interval=1, max_to_keep=2),
)

resume_epoch = None
