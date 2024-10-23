import threading
from queue import Empty, Queue
from dataclasses import dataclass
import time



class PSM_Controller():
    def __init__(self) -> None:
        self.request_queue: Queue = Queue()
        self._thread = threading.Thread(target=self._send_worker)
        self._term = threading.Event()
        self._thread.start()


    def _send_worker(self):
        while not self._term.is_set():
            try:
                request = self.request_queue.get(timeout=0.1)
                self._send_func(request)

            except Empty:
                continue

            except Exception as e:
                print(e)
    

    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        if name[0] == "_":
            raise Exception("cannot find {}".format(name))
        else:
            return getattr(self._psm, name)

    def close(self):
        self.term.set()
        self._thread.join()
    