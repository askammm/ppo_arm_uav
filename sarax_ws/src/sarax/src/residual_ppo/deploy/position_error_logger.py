#!/usr/bin/env python3

import csv
import os
from datetime import datetime
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import Vector3Stamped


class PositionErrorLogger:
    def __init__(self) -> None:
        self.topic: str = rospy.get_param("~topic", "/debug/pos_error")
        self.log_dir: str = rospy.get_param(
            "~log_dir",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "position_error"),
        )
        self.run_label: str = rospy.get_param("~run_label", "").strip()

        if not self.run_label:
            self.run_label = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, f"{self.run_label}.csv")

        self._start_time: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._prev_norm: Optional[float] = None
        self._error_integral: float = 0.0
        self._rows_written: int = 0

        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(
            ["time_sec", "e_x", "e_y", "e_z", "error_norm", "error_integral"]
        )
        self._csv_file.flush()

        rospy.Subscriber(self.topic, Vector3Stamped, self._callback, queue_size=100)
        rospy.on_shutdown(self._close)

        rospy.loginfo(
            "[position_error_logger] Logging %s to %s", self.topic, self.csv_path
        )

    def _callback(self, msg: Vector3Stamped) -> None:
        stamp = msg.header.stamp.to_sec()
        now = stamp if stamp > 0.0 else rospy.Time.now().to_sec()

        if self._start_time is None:
            self._start_time = now

        rel_time = now - self._start_time
        error_vec = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        error_norm = float(np.linalg.norm(error_vec))

        if self._prev_time is not None and self._prev_norm is not None:
            dt = max(0.0, now - self._prev_time)
            self._error_integral += 0.5 * (self._prev_norm + error_norm) * dt

        self._writer.writerow(
            [
                f"{rel_time:.6f}",
                f"{error_vec[0]:.9f}",
                f"{error_vec[1]:.9f}",
                f"{error_vec[2]:.9f}",
                f"{error_norm:.9f}",
                f"{self._error_integral:.9f}",
            ]
        )
        self._rows_written += 1

        if self._rows_written % 50 == 0:
            self._csv_file.flush()

        self._prev_time = now
        self._prev_norm = error_norm

    def _close(self) -> None:
        if not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
            rospy.loginfo(
                "[position_error_logger] Saved %d samples to %s",
                self._rows_written,
                self.csv_path,
            )


if __name__ == "__main__":
    rospy.init_node("position_error_logger", anonymous=False)
    PositionErrorLogger()
    rospy.spin()
