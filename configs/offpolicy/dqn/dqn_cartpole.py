env = dict(
    type='CartPole-v0',
    multi_process=dict(
        type='DummyVectorEnv',
        train_num=10,
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
    reward_normalization = True,
    is_double=True,
)
trainer=dict(
    type='offpolicy',
    epoch=10, 
    batch_size=64,
    step_per_epoch= 10000,
    step_per_collect = 10,
    buffer_size = 20000,
)
eps_cfg=dict(
    train = 0.1,
    test = 0.05,
)
logger=dict(type='', folder='log/dqn')