from gym_ras.tool.ds_util import DS_Controller


class MyDS(DS_Controller):
    def __init__(self, interface="/dev/input/js0", connecting_using_ds4drv=False, event_definition=None, event_format=None):
        super().__init__(interface, connecting_using_ds4drv, event_definition, event_format)

ds = MyDS()

ds.listen()