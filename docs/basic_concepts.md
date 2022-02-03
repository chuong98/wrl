# Basic Concepts 

## I. Environment and Agent 
In RL, Agent interacts with Environment( Env) by performing an `action`, then observing the outcome `(state, reward,done,info)`. For example, consider that a robot moves in a 2D maze:
+ `action`: the robots moves left-right-up-down. 
+ `state` : the robot's position in the maze `(x,y)`.
+ `reward` is 1 if it can arrive the destination, and 0 otherwise. 
+ `done` if it finishes the tasks (arrive the target or time out).
+ `info` : extra info return from environment (it is often used in simulation for debug). 
  
Our objective is to find the good `policy` so that the agent can get the highest reward. 

1. **To create an environment**, we just need to specify the name in `config.py` file, for example:
    ```python
    env = dict(
        type='CartPole-v0',                 # gym environment  Name
        multi_process=dict(
            type='SubprocVectorEnv',        # Create several env running in parallel  
            train_num=10,                       # num env in training.
            test_num=100                        # num env in testing.    
        )
    )
    ```
In the `main.py`, using the function to build environments:
```python
    train_envs,test_envs, sample_env = build_venv(cfg.env)`
```
where `sample_env` is returned for convinience to get some info about the environment, such as `state_space`, `action_space`, `reward_threshold`.

2. **To creat an agent (policy)**, we specify in the `config.py`. For example,
    ```python
    agent = dict(
        type='DQNPolicy',                       # The type of policy (a.k.a agent or algorithm)
        network=dict(
            type='MLPNet',                      # The DNN network used in the policy    
            hidden_layers=[64,64,64],
            norm_cfg=dict(type='LN'),
            act_cfg=dict(type='GELU'),),
        optim = dict(type='Adam', lr=1e-3),     # The optimizer to perform Gradient backwad of Loss funtion 
        discount_factor=0.99,                   # Some other extra params of the policy        
        estimation_step = 3,
        target_update_freq = 320,
    )
    ```
In the `main.py`, using the function to build the agent:
    ```python
    # Build The Agent (Policy)
    cfg.agent.state_shape = env.observation_space.shape or env.observation_space.n
    cfg.agent.action_shape= env.action_space.shape or env.action_space.n
    agent = build_policy(cfg.agent)
    ```
where the `state_shape` and `action_shape` are added depend on the environment `sample_env`.

3. **To allow the `agent` interact with the `env` and collect the data** (`buffer`) to train the policy, Tianshou provides the class `collector`. 
In the `config.py` file, specify the buffer by
    ```python
    train_buffer=dict(
        type='VectorReplayBuffer',
        total_size=20000, 
        buffer_num=num_parallel_envs,
    )
    ```
and in the main file:
    ```python    
    train_collector = tianshou.data.Collector(agent, train_envs, 
                            buffer=build_buffer(cfg.train_buffer))
    test_collector = tianshou.data.Collector(agent, test_envs)
    ```
The main function of `collector` is the [`collect` function](https://github.com/thu-ml/tianshou/blob/c25926dd8f5b6179f7f76486ee228982f48b4469/tianshou/data/collector.py#L144), which can be summarized in the following lines:
    ```python
    result = self.policy(self.data, last_state)                         # The Agent predicts the action from the data
    act = to_numpy(result.act)                      
    policy = result.get("policy", Batch())
    self.data.update(policy=policy, act=act)                            # Update the data with new action/policy 
    result = self.env.step(action_remap(act), ready_env_ids)            # apply action to environment
    obs_next, rew, done, info = result
    self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)  # Update the data with new state/reward/done/info
    ```

4. Finally, we need the `trainer` function to perform the main loop, such as, control how the `collector` collects data, how `agent` updates its policy, when to stop/save ckpt, and logging data. In the `config.py` file,
    ```python
    trainer=dict(
        type='offpolicy',
        max_epoch=10, 
        step_per_epoch= 10000,
        step_per_collect = 10,
        episode_per_test=5,
        batch_size=64,
        update_per_step=1,
    )
    ```
and in the `train.py`
    ```python
    result = trainer_fn(agent, train_collector, test_collector,
                        stop_fn=stop_fn, save_fn =save_fn, 
                        logger=logger, **cfg.trainer))
    ```
The follow diagram illustrates how the components interact and what the functions are called in the trainer.
<img src="https://tianshou.readthedocs.io/en/master/_images/concepts_arch2.png"
     alt="Trainer Diagram"
     style="float: left; margin-right: 10px;" />

5. Finally, to train an Agent, you simply need:
    ```
    python tools/train.py $CFG --seed $RANDOM
    ```
## Test and Visualize the Policy:
+ To Visualize the Log files, use vscode `F1 -> Python: Launch TensorBoard` then select the directory containing `log` folder.
+ To test a policy, use:
    ```
    python tools/test.py $CFG $CKPT --seed $RANDOM --render 0.1 # If want to render video
    ```
