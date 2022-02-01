env = dict(
    type='CartPole-v0',
    multi_process=dict(type='DummyVectorEnv', train_num=10, test_num=100))
agent = dict(
    type='DQNPolicy',
    network=dict(
        type='MLPNet',
        hidden_layers=[128, 128, 128],
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        drop_rate=0.05),
    optim=dict(type='Adam', lr=0.001),
    discount_factor=0.99,
    estimation_step=1,
    target_update_freq=1,
    reward_normalization=True,
    is_double=True)
work_dir = './work_dirs/dqn_cartpole'
