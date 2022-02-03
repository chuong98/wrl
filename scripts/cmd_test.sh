CFG="configs/offpolicy/dqn/dqn_cartpole.py"
CKPT="work_dir/dqn_cartpole/policy.pth"
FPS= expr 1/24
python tools/test.py $CFG $CKPT --seed $RANDOM  --render 0.04