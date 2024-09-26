from gym_ras.api import make_env
from gym_ras.env.wrapper import Visualizer, ActionOracle
import argparse
from tqdm import tqdm
import time
import numpy as np
parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='What the program does',
    epilog='Text at the bottom of help')

parser.add_argument('-p', type=int)
parser.add_argument('--repeat', type=int, default=10)
parser.add_argument('--action', type=str, default="oracle")
# parser.add_argument('--yaml-dir', type=str, default="./gym_ras/config.yaml")
# parser.add_argument('--yaml-tag', type=str, nargs='+', default=[])
parser.add_argument('--env-tag', type=str, nargs='+', default=['gasv2_surrol'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
parser.add_argument('--oracle', type=str, default='keyboard')
parser.add_argument('--no-vis', action="store_true")
parser.add_argument('--eval', action="store_true")

args = parser.parse_args()

env, env_config = make_env(tags=args.env_tag, seed=args.seed)


ts = []
for i in range(args.repeat):
    start = time.time()
    env.reset()
    t = time.time() - start
    ts.append(t)
print("reset time:", np.mean(np.array(ts)))
ts = []
for i in range(args.repeat):
    start = time.time()
    _ = env.step(env.action_space.sample())
    t = time.time() - start
    ts.append(t)
print("step time:", np.mean(np.array(ts)))

