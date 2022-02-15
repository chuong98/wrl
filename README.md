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
   1. [Deep Q-learning (DQN)](docs/Q-Learning.ipynb)
   2. [Dual DQN and Dueling DQN](docs/DualDQN_DuelingDQN.ipynb)
4. On-Policy algorithms:
   1. [Intro to Policy Gradient](docs/Vanila_Policy_Optimization.ipynb)
   2. [Trust Region Policy Optimization(TRPO) and Proximal Policy Optimization(PPO)](Proximal_Policy_Optimization.ipynb)