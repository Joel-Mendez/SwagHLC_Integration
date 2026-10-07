"""Live SWAG sensor subscriptions (2026-10-07).

Caches the latest live data published by the swag_boards_ros nodes so the HLC can use it once its
models move from the RRD recordings to the SWAG boards. NOT used by inference yet -- the HLC still
classifies the RRD replay. Prints a 1 Hz status line (message rates per topic group).

Topics (absolute names, published by swag_boards_ros):
  /muscle_activation/{left,right}  Float32MultiArray, 15 values at 100 Hz: EMG envelopes averaged per
                          muscle number 1..15 (thigh 1-8, shank 9-15) via data_handler's muscle_map.yaml.
                          (Raw /sensor/emg_* is ~1-2 kHz per board -- too heavy for Python callbacks.)
  /kinematics/{hip,knee,ankle}_angle_{left,right}, hip_velocity_*, gait_phase_*, walking_speed_*
                          Float64 at 100 Hz (deg, deg/s, rad, km/h)
  /sensor/imu_<id>        Float64MultiArray [x, y, z, w, read_time] orientation quaternion -- OPTIONAL
                          (include_imu=True): ~1 kHz per board, same CPU concern as raw EMG.
Segment ids follow the data_handler launch parameters (e.g. 1 = left thigh, 2 = right thigh in the
two-thigh test launch).
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

IMU_IDS = (1, 2, 3, 4, 5, 6)
KINEMATICS = [f"{j}_angle_{s}" for j in ("hip", "knee", "ankle") for s in ("left", "right")] + \
             [f"{k}_{s}" for k in ("hip_velocity", "gait_phase", "walking_speed") for s in ("left", "right")]


class LiveSensorCache:
    def __init__(self, node, print_status: bool = True, include_imu: bool = False) -> None:
        from std_msgs.msg import Float32MultiArray, Float64, Float64MultiArray

        self._lock = threading.Lock()
        self._imu: dict[int, tuple[float, ...]] = {}
        self._emg: dict[str, tuple[float, ...]] = {}
        self._kin: dict[str, float] = {}
        self._counts: Counter = Counter()
        for side in ("left", "right"):
            node.create_subscription(Float32MultiArray, f"/muscle_activation/{side}",
                                     lambda m, s=side: self._on_emg(s, m), 10)
        for k in KINEMATICS:
            node.create_subscription(Float64, f"/kinematics/{k}", lambda m, k=k: self._on_kin(k, m), 10)
        if include_imu:
            for i in IMU_IDS:
                node.create_subscription(Float64MultiArray, f"/sensor/imu_{i}", lambda m, i=i: self._on_imu(i, m), 10)
        if print_status:
            node.create_timer(1.0, self._print_status)
        print(f"[LiveSensors] subscribed: /muscle_activation/{{left,right}}, /kinematics/* ({len(KINEMATICS)} topics)"
              + (", /sensor/imu_1..6" if include_imu else "") + " (cache only, not used by inference)")

    def _on_imu(self, i: int, m) -> None:
        with self._lock:
            self._imu[i] = tuple(m.data); self._counts[f"imu_{i}"] += 1

    def _on_emg(self, side: str, m) -> None:
        with self._lock:
            self._emg[side] = tuple(m.data); self._counts[f"muscle_act_{side}"] += 1

    def _on_kin(self, k: str, m) -> None:
        with self._lock:
            self._kin[k] = float(m.data); self._counts["kinematics"] += 1

    def latest(self) -> dict[str, Any]:
        """Snapshot of the most recent value of every topic seen so far."""
        with self._lock:
            return {"muscle_activation": dict(self._emg), "kinematics": dict(self._kin), "imu": dict(self._imu)}

    def _print_status(self) -> None:
        with self._lock:
            c = dict(self._counts); self._counts.clear()
            hip_l, hip_r = self._kin.get("hip_angle_left"), self._kin.get("hip_angle_right")
        if not c:
            print("[LiveSensors] no live data (swag_boards_ros not running?)")
            return
        parts = [f"{k} {v}/s" for k, v in sorted(c.items())]
        hips = f" | hip L {hip_l:.1f} R {hip_r:.1f} deg" if hip_l is not None and hip_r is not None else ""
        print(f"[LiveSensors] {', '.join(parts)}{hips}")
