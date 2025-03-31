from gym_ras.tool.img_tool import CV2_Visualizer
import os
import argparse
from pathlib import Path
import time
os.environ["XDG_SESSION_TYPE"] = "xcb"
from gym_ras.tool.keyboard import Keyboard
parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default="./data/stream_cam")
parser.add_argument('--depth-c', type=float, default="0.3")
parser.add_argument('--depth-r', type=float, default="0.1")
parser.add_argument('--seg', action="store_true")
parser.add_argument('--vis-nodepth', type=str, default="")
parser.add_argument('--vis-tag', type=str, nargs='+', default=["rgb","depth"])
parser.add_argument("--cam", type=str, default="stereo")

args = parser.parse_args()
kb = Keyboard(blocking=False)


if args.cam == "stereo":
    from gym_ras.tool.stereo_dvrk import VisPlayer
    from gym_ras.tool.config import load_yaml
    cam_dis_file = "./data/dvrk_cal/cam_dis.yaml"
    if cam_dis_file != "":
        _args = load_yaml(cam_dis_file)
        _cam_offset_z = _args['cam_dis']
    else:
        _cam_offset_z = 0.2
    engine = VisPlayer(depth_center=_cam_offset_z,
                 depth_range=0.1,)
    engine.init_run()
    print("finish init")
if args.seg:
    from gym_ras.tool.track_any_tool import Tracker
    from gym_ras.tool.seg_tool import TrackAnySegmentor
    tracker = Tracker(sam=False, xmem=True)
    tracker.load_template()
    tracker.update_template()
    segment_engine = TrackAnySegmentor()
    segment_engine.model = tracker
    engine._segmentor = segment_engine
visualizer = CV2_Visualizer(
    update_hz=50, render_dir=args.savedir, vis_tag=args.vis_tag, keyboard=False
)

is_quit = False
ch = ''
while (ch != 'q'):
    img = engine.get_image()
    if "mask" in img:
        img["mask"] = {str(k): v[0] for k,v in img["mask"].items()}

    is_quit = visualizer.cv_show(img)
    ch = kb.get_char()
engine.close()
kb.close()

time.sleep(1.0)


