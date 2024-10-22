from gym_ras.tool.stereo_csr import VisPlayer
engine = VisPlayer()
engine.init_run()


for i in range(5):
    d = engine.get_center_depth()  
    print(d)
engine.close()