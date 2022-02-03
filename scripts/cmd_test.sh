CFG="configs/offpolicy/dqn/dqn_cartpole.py"
CKPT="work_dir/dqn_cartpole/policy.pth"
python tools/test.py $CFG $CKPT --seed $RANDOM 
# --render 1/24