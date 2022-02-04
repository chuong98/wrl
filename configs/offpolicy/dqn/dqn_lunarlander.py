num_parallel_envs=16
env = dict(
    type='LunarLander-v2',
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
        hidden_layers=[64,128,128],
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        drop_rate=0.05),
    optim = dict(type='Adam', lr=13e-3),
    discount_factor=0.99,
    estimation_step = 4,
    target_update_freq = 500,
    reward_normalization = False,
    is_double=True,
)

collector=dict(
    train_buffer=dict(
        type='VectorReplayBuffer',
        total_size=100000, 
        buffer_num=num_parallel_envs),
    exploration_noise=dict(train=True,test=True),
    precollected_steps=128*num_parallel_envs,
)

trainer=dict(
    type='OffPolicyTrainer',
    max_epoch=10, 
    step_per_epoch= 80000,
    step_per_collect = 16,
    episode_per_test=5,
    batch_size=128,
    update_per_step=0.0625,
    train_hook=dict(type='EpsExpDecayHook', eps_start=0.73, eps_end=0.01, eps_decay=1-5e-6),
    test_hook=dict(type='EpsConstHook', eps=0.01),   
    ckpt_cfg=dict(interval=1, max_to_keep=2),
)

resume_epoch = None
