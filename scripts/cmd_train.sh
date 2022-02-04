CFG="configs/offpolicy/dqn/dqn_cartpole.py"
python tools/train.py $CFG --seed $RANDOM --gpus 1 --resume-epoch 2