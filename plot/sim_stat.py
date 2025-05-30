from gym_ras.tool.config import load_yaml
import argparse
import glob
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str) 
args = parser.parse_args()

path = Path(args.dir) / '*.yml'
path = str(path)

print(path)
sr = []
scores= []
for file in glob.glob(path):
    yml_args = load_yaml(file)
    sr.append(yml_args['success_rate'])
    scores.extend(yml_args['score'])
    print(yml_args['success_rate'])
print("scores len:", len(scores))

sr = np.array(sr)
sr = sr * 100
print("success rate mean (std):")
print(f"{np.round(np.mean(sr), decimals=1)} ({np.round(np.std(sr), decimals=1)})")


scores = np.array(scores)
scores = scores * 100
print("score mean (std):")
print(f"{np.round(np.mean(scores), decimals=1)} ({np.round(np.std(scores), decimals=1)})")