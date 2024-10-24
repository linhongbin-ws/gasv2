from test.ds_util import DS_Controller
import time


class MyDS(DS_Controller):
    def on_circle_press(self):
        print("circle press")

ds = MyDS()

start = time.time()

while (time.time() - start) <3:
    # print(time.time() - start)
    pass
ds.close()
