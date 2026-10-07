"""ROS 2 transport backend.

Hybrid design:
  * sensor/* topics  -> InProcTransport (fast, no serialization overhead)
  * intent/* topics  -> publish to BOTH inproc (HLC terminal monitor keeps
                        working) AND a ROS2 std_msgs/String topic (MLC receives)

IntentPrediction is sent as a plain string: stable_label or label or str(argmax).
This matches what the MLC stores in classifierOutput_.

Requires rclpy and a sourced ROS2 install. Use mode: async in the run config —
rclpy is not safe across fork(), so mode: process will not work.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from swag_hlc.transport.base import Publisher, Subscriber, Transport
from swag_hlc.transport.inproc import InProcTransport

_INSTALL_HINT = (
    "ROS 2 (rclpy) is not available. Source your ROS2 install "
    "(e.g. source /opt/ros/humble/setup.bash) and try again."
)


def _require_rclpy():
    try:
        import rclpy  # noqa: F401
    except Exception as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    return rclpy


class _Ros2IntentPublisher(Publisher):
    """Publishes IntentPrediction to inproc (HLC monitor) + ROS2 (MLC)."""

    def __init__(self, ros_pub, inproc_pub: Publisher) -> None:
        self._ros_pub = ros_pub
        self._inproc_pub = inproc_pub

    def publish(self, message: Any) -> None:
        self._inproc_pub.publish(message)

        from std_msgs.msg import String
        from swag_hlc.messages.schema import IntentPrediction

        msg = String()
        if isinstance(message, IntentPrediction):
            msg.data = message.stable_label or message.label or str(message.argmax)
        else:
            msg.data = str(message)
        self._ros_pub.publish(msg)


class Ros2Transport(Transport):
    """ROS2 bridge: sensor topics stay inproc, intent topics go to ROS2 + inproc."""

    def __init__(self, namespace: str = "swag", **_: Any) -> None:
        self._ns = namespace
        self._inproc = InProcTransport()

        rclpy = _require_rclpy()
        rclpy.init()
        self._node = rclpy.create_node("swag_hlc")
        # Live SWAG board data (cache + 1 Hz status only; inference still uses the RRD replay)
        from swag_hlc.transport.live_sensors import LiveSensorCache
        self.live = LiveSensorCache(self._node)
        self._spin_thread = threading.Thread(
            target=lambda: rclpy.spin(self._node), daemon=True
        )
        self._spin_thread.start()
        print(f"[Ros2Transport] node 'swag_hlc' ready, intent → /{namespace}/intent/*")

    def _ros_topic(self, topic: str) -> str:
        return f"/{self._ns}/{topic}" if self._ns else f"/{topic}"

    def declare_topic(self, topic: str) -> None:
        self._inproc.declare_topic(topic)

    def publisher(self, topic: str) -> Publisher:
        inproc_pub = self._inproc.publisher(topic)
        if topic.startswith("intent/"):
            from std_msgs.msg import String
            ros_pub = self._node.create_publisher(String, self._ros_topic(topic), 10)
            print(f"[Ros2Transport] intent publisher → {self._ros_topic(topic)}")
            return _Ros2IntentPublisher(ros_pub, inproc_pub)
        return inproc_pub

    def subscriber(self, topic: str) -> Subscriber:
        return self._inproc.subscriber(topic)

    def close(self) -> None:
        import rclpy
        rclpy.shutdown()
        self._spin_thread.join(timeout=2.0)
