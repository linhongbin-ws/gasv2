from gym_ras.tool.img_tool import CV2_Visualizer
import os
import argparse
from pathlib import Path
import time
os.environ["XDG_SESSION_TYPE"] = "xcb"

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default="./data/stream_cam")
parser.add_argument('--depth-c', type=float, default="0.3")
parser.add_argument('--depth-r', type=float, default="0.1")
parser.add_argument('--seg', action="store_true")
parser.add_argument('--vis-nodepth', type=str, default="")
parser.add_argument('--vis-tag', type=str, nargs='+', default=[])
parser.add_argument("--cam", type=str, default="stereo")

args = parser.parse_args()

if args.seg:
    from gym_ras.tool.track_any_tool import Tracker
    from gym_ras.tool.seg_tool import TrackAnySegmentor
    tracker = Tracker(sam=False, xmem=True)
    tracker.load_template()
    tracker.update_template()
    segment_engine = TrackAnySegmentor()
    segment_engine.model = tracker

if args.cam == "stereo":
    from gym_ras.tool.stereo_csr import VisPlayer
    engine = VisPlayer()
    engine.init_run()
    print("finish init")
visualizer = CV2_Visualizer(
    update_hz=10, render_dir=args.savedir, vis_tag=args.vis_tag, keyboard=True
)

# img = {"rgb":rgb, "depth": depth}
is_quit = False
while not is_quit:
    img = engine.get_image()
    if args.seg:
        masks = segment_engine.predict(img['rgb'])
        print(masks)
        img["mask"] = {str(k): v[0] for k,v in masks.items()}
    is_quit = visualizer.cv_show(img)
    print("d")
time.sleep(1.0)
print("exit")
del visualizer
engine.cap_0.release()
engine.cap_2.release()
del engine
time.sleep(1.0)



    # print(results)
