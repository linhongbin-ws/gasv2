from pyPS4Controller.controller import Controller
from threading import Thread, Event
import struct
import time
import os



class DS_Controller(Controller):
    def __init__(self, parallel=True, interface="/dev/input/js0", 
                 connecting_using_ds4drv=False, event_definition=None, event_format=None):
        super().__init__(interface,connecting_using_ds4drv,event_definition,event_format)
        self._parallel = parallel
        self._thread = Thread(target=self.listen)
        self._stop_event = Event()
        self._thread.start()

    
    def listen(self, timeout=30, on_connect=None, on_disconnect=None, on_sequence=None):
        """
        Start listening for events on a given self.interface
        :param timeout: INT, seconds. How long you want to wait for the self.interface.
                        This allows you to start listening and connect your controller after the fact.
                        If self.interface does not become available in N seconds, the script will exit with exit code 1.
        :param on_connect: function object, allows to register a call back when connection is established
        :param on_disconnect: function object, allows to register a call back when connection is lost
        :param on_sequence: list, allows to register a call back on specific input sequence.
                            e.g [{"inputs": ['up', 'up', 'down', 'down', 'left', 'right,
                                             'left', 'right, 'start', 'options'],
                                  "callback": () -> None)}]
        :return: None
        """
        def on_disconnect_callback():
            self.is_connected = False
            if on_disconnect is not None:
                on_disconnect()

        def on_connect_callback():
            self.is_connected = True
            if on_connect is not None:
                on_connect()

        def wait_for_interface():
            print("Waiting for interface: {} to become available . . .".format(self.interface))
            for i in range(timeout):
                if os.path.exists(self.interface):
                    print("Successfully bound to: {}.".format(self.interface))
                    on_connect_callback()
                    return
                time.sleep(1)
            print("Timeout({} sec). Interface not available.".format(timeout))
            # exit(1)

        def read_events():
            try:
                return _file.read(self.event_size)
            except IOError:
                print("Interface lost. Device disconnected?")
                on_disconnect_callback()
                # exit(1)

        def check_for(sub, full, start_index):
            return [start for start in range(start_index, len(full) - len(sub) + 1) if
                    sub == full[start:start + len(sub)]]

        def unpack():
            __event = struct.unpack(self.event_format, event)
            return (__event[3:], __event[2], __event[1], __event[0])

        wait_for_interface()
        try:
            _file = open(self.interface, "rb")
            event = read_events()
            if on_sequence is None:
                on_sequence = []
            special_inputs_indexes = [0] * len(on_sequence)
            while not self._stop_event.is_set() and event:
                (overflow, value, button_type, button_id) = unpack()
                if button_id not in self.black_listed_buttons:
                    self.__handle_event(button_id=button_id, button_type=button_type, value=value, overflow=overflow,
                                        debug=self.debug)
                for i, special_input in enumerate(on_sequence):
                    check = check_for(special_input["inputs"], self.event_history, special_inputs_indexes[i])
                    if len(check) != 0:
                        special_inputs_indexes[i] = check[0] + 1
                        special_input["callback"]()
                event = read_events()
        except KeyboardInterrupt:
            print("\nExiting (Ctrl + C)")
            on_disconnect_callback()
            # exit(1)
    def __handle_event(self, button_id, button_type, value, overflow, debug):

        event = self.event_definition(button_id=button_id,
                                      button_type=button_type,
                                      value=value,
                                      connecting_using_ds4drv=self.connecting_using_ds4drv,
                                      overflow=overflow,
                                      debug=debug)

        if event.R3_event():
            self.event_history.append("right_joystick")
            if event.R3_y_at_rest():
                self.on_R3_y_at_rest()
            elif event.R3_x_at_rest():
                self.on_R3_x_at_rest()
            elif event.R3_right():
                self.on_R3_right(event.value)
            elif event.R3_left():
                self.on_R3_left(event.value)
            elif event.R3_up():
                self.on_R3_up(event.value)
            elif event.R3_down():
                self.on_R3_down(event.value)
        elif event.L3_event():
            self.event_history.append("left_joystick")
            if event.L3_y_at_rest():
                self.on_L3_y_at_rest()
            elif event.L3_x_at_rest():
                self.on_L3_x_at_rest()
            elif event.L3_up():
                self.on_L3_up(event.value)
            elif event.L3_down():
                self.on_L3_down(event.value)
            elif event.L3_left():
                self.on_L3_left(event.value)
            elif event.L3_right():
                self.on_L3_right(event.value)
        elif event.circle_pressed():
            self.event_history.append("circle")
            self.on_circle_press()
        elif event.circle_released():
            self.on_circle_release()
        elif event.x_pressed():
            self.event_history.append("x")
            self.on_x_press()
        elif event.x_released():
            self.on_x_release()
        elif event.triangle_pressed():
            self.event_history.append("triangle")
            self.on_triangle_press()
        elif event.triangle_released():
            self.on_triangle_release()
        elif event.square_pressed():
            self.event_history.append("square")
            self.on_square_press()
        elif event.square_released():
            self.on_square_release()
        elif event.L1_pressed():
            self.event_history.append("L1")
            self.on_L1_press()
        elif event.L1_released():
            self.on_L1_release()
        elif event.L2_pressed():
            self.event_history.append("L2")
            self.on_L2_press(event.value)
        elif event.L2_released():
            self.on_L2_release()
        elif event.R1_pressed():
            self.event_history.append("R1")
            self.on_R1_press()
        elif event.R1_released():
            self.on_R1_release()
        elif event.R2_pressed():
            self.event_history.append("R2")
            self.on_R2_press(event.value)
        elif event.R2_released():
            self.on_R2_release()
        elif event.options_pressed():
            self.event_history.append("options")
            self.on_options_press()
        elif event.options_released():
            self.on_options_release()
        elif event.left_right_arrow_released():
            self.on_left_right_arrow_release()
        elif event.up_down_arrow_released():
            self.on_up_down_arrow_release()
        elif event.left_arrow_pressed():
            self.event_history.append("left")
            self.on_left_arrow_press()
        elif event.right_arrow_pressed():
            self.event_history.append("right")
            self.on_right_arrow_press()
        elif event.up_arrow_pressed():
            self.event_history.append("up")
            self.on_up_arrow_press()
        elif event.down_arrow_pressed():
            self.event_history.append("down")
            self.on_down_arrow_press()
        elif event.playstation_button_pressed():
            self.event_history.append("ps")
            self.on_playstation_button_press()
        elif event.playstation_button_released():
            self.on_playstation_button_release()
        elif event.share_pressed():
            self.event_history.append("share")
            self.on_share_press()
        elif event.share_released():
            self.on_share_release()
        elif event.R3_pressed():
            self.event_history.append("R3")
            self.on_R3_press()
        elif event.R3_released():
            self.on_R3_release()
        elif event.L3_pressed():
            self.event_history.append("L3")
            self.on_L3_press()
        elif event.L3_released():
            self.on_L3_release()

    def on_x_press(self):
        pass #print("on_triangle_press")

    def on_x_release(self):
        pass #print("on_triangle_press")

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

    def close(self):
        self._stop_event.set()
        print("exit ds thread..")
        self._thread.join()
        print("exit ds thread, done")



class dVRK_DS(DS_Controller):
    def on_circle_press(self):
        self._press = "circle"
        self._wait = False
        print("circle press")

    def get_char(self):
        self._wait = True
        while self._wait:
            pass
        return self._press