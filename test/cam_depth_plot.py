from gym_ras.tool.img_tool import CV2_Visualizer
import os
import argparse
from pathlib import Path
import time
from matplotlib.pyplot import imshow, subplot, axis, cm, show
import matplotlib.pyplot as plt
from gym_ras.tool.stereo_dvrk import VisPlayer
engine = VisPlayer()
engine.init_run()
print("finish init")
for i in range(2):
    img = engine.get_image()

imshow(img['depReal'])
plt.colorbar()
show()
engine.close()



