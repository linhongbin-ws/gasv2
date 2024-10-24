import threading
from queue import Empty, Queue
from csrk.arm_proxy import ArmProxy
from csrk.node import Node
import PyCSR
from dataclasses import dataclass
import time

@dataclass
class MoveCP_MSG():
    frame: PyCSR.Frame
    acc: float
    duration: float
    jaw: float


class PSM_Controller():
    def __init__(self) -> None:
        node_ = Node("NDDS_QOS_PROFILES.CSROS.xml") # NOTE: path Where you put the ndds xml file
        # ecm = ArmProxy(node_, "psa2")
        # psa1 = ArmProxy(node_, "psa1")
        self._psm = ArmProxy(node_, "psa3")

        while(not self._psm.is_connected):
            self._psm.measured_cp()
            # To check if the arm is connected
            self._psm.read_rtrk_arm_state()
            print("connection: ",self._psm.is_connected)

        self.request_queue: Queue = Queue()
        self._thread = threading.Thread(target=self._send_worker)
        self.is_active = True
        self._thread.start()

    def _send_func(self, request):
        self._psm.move_cp(request.frame, acc=request.acc, duration=request.duration, jaw=request.jaw)
        time.sleep(request.duration)

    def _send_worker(self):
        while self.is_active:
            try:
                request = self.request_queue.get(timeout=0.1)
                self._send_func(request)

            except Empty:
                continue

            except Exception as e:
                print(e)
            print("kk")
        print("jump outsdfsdfasdfdsafasfasfasf")
    def move_cp(self, frame: PyCSR.Frame, acc: float, duration: float, jaw: float):
        request = MoveCP_MSG(frame=frame, acc=acc, duration=duration, jaw=jaw)
        self.request_queue.put(request)

    def __getattr__(self, name):
        """__getattr__ is only invoked if the attribute wasn't found the usual ways."""
        if name[0] == "_":
            raise Exception("cannot find {}".format(name))
        else:
            return getattr(self._psm, name)

    def close(self):
        print("call delelte")
        self.is_active = False
        self._thread.join()
        del self._thread
        del self._psm
    
    def open_gripper(self, value):
        jp = self._psm.measured_jp()
        jp[6] = value
        self._psm.move_jp(jp,max_vel=3, acc=0.2)
        time.sleep(2)