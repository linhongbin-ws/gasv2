from pynput import keyboard
import time
import threading
from copy import deepcopy

class Keyboard():
    def __init__(self, blocking=True):
        self._thread = None
        if not blocking:
            self.is_active = True
            self._thread = threading.Thread(target=self._worker)
            self._thread.start()
            self._char = None
        

    def _worker(self):
        while self.is_active:
            self._char = self._get_char()

    def get_char(self):
        if self._thread is None:
            return self._get_char()
        else:
            return deepcopy(self._char)

    def _get_char(self):
        with keyboard.Events() as events:
            for event in events:
                if isinstance(event, keyboard.Events.Press) and not isinstance(event, keyboard.Events.Release):
                    break
        return event.key.char if hasattr(event.key, "char") else event.key

    def close(self):
        if self._thread is not None:
            self.is_active = False
            del self._thread


if __name__ == "__main__":
    device = Keyboard(blocking=False)
    while True:
        ch = device.get_char()
        print(ch)
        if ch == "q":
            break
    device.close()