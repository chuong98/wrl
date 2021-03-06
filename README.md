# Weekend Reinforcement Learning 

About: This package provides a wraper of [Tianshou RL package](https://github.com/thu-ml/tianshou.git) by using the `mmcv.registry()` to convert a config file to Torch module/class/function. Using the register provides the flexibility and reduce the redundant code. 
+ For example, in the conventional way, you will need to write a parser to parse the arguments (to change some model parameters), and rewrite the whole `train.py` file for different methods/ algorithm. 
+ In constrast, with the registry, you only need to change the config files, and using a single `train.py` for all kind combination of env/algorithm policy. 


# Install
```
conda create -n chuong_RL python=3.8 -y
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install openmim 
mim install mmcv-full
pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade
```

# Tutorials:
1. [Config Usage and How to implement new Algorithm](docs/Config_Usage.md)
2. [Problem Formulation and Notations](docs/ProblemFormulation_Notation.ipynb)
3. Off-Policy algorithms:
   1. [Deep Q-learning (DQN -Nature 2015)](docs/Q-Learning.ipynb)
   2. [Double-DQN (AAAI 2016), Priority Experience Replay (PER) (ICLR 2016), and Dueling DQN (ICML 2016)](docs/DoubleDQN_DuelingDQN.ipynb)
   3. [Dealing with Sparse Reward]
      1. Hindsight Experience Replay (HER - NIPS 2017)
      2. First Return, Then Explore (Nature 2020)
4. On-Policy algorithms:
   1. [Intro to Policy Gradient](docs/Vanila_Policy_Optimization.ipynb)
   2. [Trust Region Policy Optimization(TRPO) (ICML 2015) and Proximal Policy Optimization (PPO)](Proximal_Policy_Optimization.ipynb)
5. Mixing Q-learning and Policy Algorithm:
   1. Deep Deterministic Policy Gradient (DDPG - ICLR2016)
   2. Twin Delay DDPG (TD3 - ICML2018)
   3. Soft Actor-Critc (SAC - ICML 2018)  