import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import rtde_control
import rtde_receive
from transform3d import Transform as T


@dataclass
class ProbeResult:
    base_t_tcps_contact: list[T]
    base_forces_contact: np.ndarray

    @cached_property
    def base_t_tcp_contact(self):
        """
        Simply represent the contact pose as the last pose during contact,
        since the force controller should be closed to settled at that point.
        """
        return self.base_t_tcps_contact[-1]


class Robot:
    def __init__(self, ip: str, zero_tcp_offset=True):
        self.recv = rtde_receive.RTDEReceiveInterface(ip)
        self.ctrl = rtde_control.RTDEControlInterface(ip)
        if zero_tcp_offset:
            self.ctrl.setTcp(T())
            assert np.linalg.norm(self.ctrl.getTCPOffset()) == 0

    def moveL(self, *args, **kwargs):
        # throw if move does not succeed
        assert self.ctrl.moveL(*args, **kwargs)

    def teachMode(self):
        assert self.ctrl.teachMode()

    def endTeachMode(self):
        assert self.ctrl.endTeachMode()

    def base_t_tcp(self):
        return T.from_xyz_rotvec(self.recv.getActualTCPPose())

    def zero_ft(self):
        time.sleep(0.2)
        self.ctrl.zeroFtSensor()
        time.sleep(0.2)

    def probe(
        self,
        tcp_t_tip: T,
        target_force=3.0,
        move_speed=5e-2,
        max_approach_speed=5e-3,
        contact_time=0.5,
        timeout=5.0,
        zero_ft=True,
    ) -> ProbeResult:
        base_t_tcp = self.base_t_tcp()

        self.ctrl.forceModeSetDamping(1e-2)
        self.ctrl.forceModeSetGainScaling(1)

        if zero_ft:
            self.zero_ft()

        self.ctrl.forceMode(
            base_t_tcp @ tcp_t_tip,  # task_frame
            [0, 0, 1, 0, 0, 0],  # selection_vector
            [0, 0, target_force, 0, 0, 0],  # wrench
            2,  # type: no additional transform
            [1, 1, max_approach_speed, 1, 1, 1],  # speed limit for compliant axis
        )

        start = time.time()
        time_contact_detected = None
        base_forces_contact = []
        base_t_tcps_contact = []

        try:
            while True:
                # approx 100 hz polling
                time.sleep(1e-2)
                base_force = self.recv.getActualTCPForce()[:3]
                if (
                    np.linalg.norm(base_force) > target_force
                    and time_contact_detected is None
                ):
                    time_contact_detected = time.time()
                if time_contact_detected is not None:
                    base_forces_contact.append(base_force)
                    base_t_tcps_contact.append(self.base_t_tcp())

                    contact_duration = time.time() - time_contact_detected
                    if contact_duration >= contact_time:
                        break
                if time.time() - start > timeout:
                    raise TimeoutError()
        finally:
            self.ctrl.forceModeStop()
            self.moveL(base_t_tcp, speed=move_speed)

        return ProbeResult(
            base_t_tcps_contact=base_t_tcps_contact,
            base_forces_contact=base_forces_contact,
        )
