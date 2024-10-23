from pyPS4Controller.controller import Controller



class MyController(Controller):
    def on_x_press(self):
       print("Hello world")

    def on_x_release(self):
       print("Goodbye world")

    def on_triangle_press(self):
        pass #print("on_triangle_press")

    def on_triangle_release(self):
        pass #print("on_triangle_release")

    def on_circle_press(self):
        pass #print("on_circle_press")

    def on_circle_release(self):
        pass #print("on_circle_release")

    def on_square_press(self):
        pass #print("on_square_press")

    def on_square_release(self):
        pass #print("on_square_release")

    def on_L1_press(self):
        pass #print("on_L1_press")

    def on_L1_release(self):
        pass #print("on_L1_release")

    def on_L2_press(self, value):
        pass #print("on_L2_press: {}".format(value))

    def on_L2_release(self):
        pass #print("on_L2_release")

    def on_R1_press(self):
        pass #print("on_R1_press")

    def on_R1_release(self):
        pass #print("on_R1_release")

    def on_R2_press(self, value):
        pass #print("on_R2_press: {}".format(value))

    def on_R2_release(self):
        pass #print("on_R2_release")

    def on_up_arrow_press(self):
        pass #print("on_up_arrow_press")

    def on_up_down_arrow_release(self):
        pass #print("on_up_down_arrow_release")

    def on_down_arrow_press(self):
        pass #print("on_down_arrow_press")

    def on_left_arrow_press(self):
        pass #print("on_left_arrow_press")

    def on_left_right_arrow_release(self):
        pass #print("on_left_right_arrow_release")

    def on_right_arrow_press(self):
        pass #print("on_right_arrow_press")

    def on_L3_up(self, value):
        pass #print("on_L3_up: {}".format(value))

    def on_L3_down(self, value):
        pass #print("on_L3_down: {}".format(value))

    def on_L3_left(self, value):
        pass #print("on_L3_left: {}".format(value))

    def on_L3_right(self, value):
        pass #print("on_L3_right: {}".format(value))

    def on_L3_y_at_rest(self):
        """L3 joystick is at rest after the joystick was moved and let go off"""
        pass #print("on_L3_y_at_rest")

    def on_L3_x_at_rest(self):
        """L3 joystick is at rest after the joystick was moved and let go off"""
        pass #print("on_L3_x_at_rest")

    def on_L3_press(self):
        """L3 joystick is clicked. This event is only detected when connecting without ds4drv"""
        pass #print("on_L3_press")

    def on_L3_release(self):
        """L3 joystick is released after the click. This event is only detected when connecting without ds4drv"""
        pass #print("on_L3_release")

    def on_R3_up(self, value):
        pass #print("on_R3_up: {}".format(value))

    def on_R3_down(self, value):
        pass #print("on_R3_down: {}".format(value))

    def on_R3_left(self, value):
        pass #print("on_R3_left: {}".format(value))

    def on_R3_right(self, value):
        pass #print("on_R3_right: {}".format(value))

    def on_R3_y_at_rest(self):
        """R3 joystick is at rest after the joystick was moved and let go off"""
        pass #print("on_R3_y_at_rest")

    def on_R3_x_at_rest(self):
        """R3 joystick is at rest after the joystick was moved and let go off"""
        pass #print("on_R3_x_at_rest")

    def on_R3_press(self):
        """R3 joystick is clicked. This event is only detected when connecting without ds4drv"""
        pass #print("on_R3_press")

    def on_R3_release(self):
        """R3 joystick is released after the click. This event is only detected when connecting without ds4drv"""
        pass #print("on_R3_release")

    def on_options_press(self):
        pass #print("on_options_press")

    def on_options_release(self):
        pass #print("on_options_release")

    def on_share_press(self):
        """this event is only detected when connecting without ds4drv"""
        pass #print("on_share_press")

    def on_share_release(self):
        """this event is only detected when connecting without ds4drv"""
        pass #print("on_share_release")

    def on_playstation_button_press(self):
        """this event is only detected when connecting without ds4drv"""
        pass #print("on_playstation_button_press")

    def on_playstation_button_release(self):
        """this event is only detected when connecting without ds4drv"""
        pass #print("on_playstation_button_release")

    def get_char():
        

controller = MyController(interface="/dev/input/js0", connecting_using_ds4drv=False,
                        )
controller.listen()