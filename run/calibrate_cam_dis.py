import numpy as np
import yaml
import pathlib
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default='./data/dvrk_cal')
args = parser.parse_args()
savedir = pathlib.Path(args.savedir)
savedir.mkdir(parents=True, exist_ok=True)

data = {}


from gym_ras.tool.stereo_dvrk import VisPlayer
engine = VisPlayer()
engine.init_run()
ds = []
for i in range(5):
    d = engine.get_center_depth()  
    print(d)
    ds.append(d)


dis = np.mean(np.array(d))
print(dis)
data["cam_dis"] = dis.item()
print(f"camera distance: {dis}")
with open(str(savedir / 'cam_dis.yaml'), 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

engine.close()