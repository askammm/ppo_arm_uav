#!/usr/bin/env python3

import math
from typing import Any, Dict, List

import rospy
from geometry_msgs.msg import Vector3Stamped
from mavros_msgs.srv import CommandBool, SetMode
from m4e_mav_trajectory_planner.srv import PlanTakeoff
from std_srvs.srv import Trigger


class FixedScheduleRunner:
    def __init__(self) -> None:
        self.waypoint_topic: str = rospy.get_param("~waypoint_topic", "waypoint")
        self.arming_service: str = rospy.get_param(
            "~arming_service", "mavros/cmd/arming"
        )
        self.offboard_service: str = rospy.get_param(
            "~offboard_service", "mavros/set_mode"
        )
        self.takeoff_service: str = rospy.get_param(
            "~takeoff_service", "plan_takeoff"
        )
        self.fold_service: str = rospy.get_param(
            "~fold_service", "/m4e_mani_user_control/foldArm"
        )
        self.extend_service: str = rospy.get_param(
            "~extend_service", "/m4e_mani_user_control/extendArm"
        )
        self.loop_hz: float = float(rospy.get_param("~loop_hz", 20.0))
        self.start_delay: float = float(rospy.get_param("~start_delay", 0.0))
        self.sequence: List[Dict[str, Any]] = list(rospy.get_param("~sequence", []))

        if not self.sequence:
            raise rospy.ROSInitException("~sequence is empty. Load a schedule YAML first.")

        self.sequence.sort(key=lambda step: float(step["time"]))
        self._next_index = 0
        self._started = False
        self._start_time = None

        self._waypoint_pub = rospy.Publisher(
            self.waypoint_topic, Vector3Stamped, queue_size=1
        )

        rospy.loginfo(
            "[fixed_schedule_runner] Loaded %d steps. start_delay=%.2fs",
            len(self.sequence),
            self.start_delay,
        )

    def run(self) -> None:
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if not self._started:
                self._start_time = now + self.start_delay
                self._started = True
                rospy.loginfo(
                    "[fixed_schedule_runner] Starting timeline at t=%.3f",
                    self._start_time,
                )

            elapsed = now - float(self._start_time)
            while self._next_index < len(self.sequence):
                step = self.sequence[self._next_index]
                if elapsed + 1e-9 < float(step["time"]):
                    break
                self._execute_step(step)
                self._next_index += 1

            if self._next_index >= len(self.sequence):
                rospy.loginfo_once("[fixed_schedule_runner] All scheduled steps executed.")

            rate.sleep()

    def _execute_step(self, step: Dict[str, Any]) -> None:
        action = str(step.get("action", "")).strip()
        step_time = float(step["time"])
        rospy.loginfo(
            "[fixed_schedule_runner] t=%.2f executing action=%s",
            step_time,
            action,
        )

        if action == "arm":
            self._call_service(self.arming_service, CommandBool, True)
        elif action == "disarm":
            self._call_service(self.arming_service, CommandBool, False)
        elif action == "offboard":
            self._call_service(self.offboard_service, SetMode, 0, "OFFBOARD")
        elif action == "takeoff":
            altitude = float(step["altitude"])
            self._call_service(self.takeoff_service, PlanTakeoff, altitude)
        elif action == "waypoint":
            self._publish_waypoint(
                float(step["x"]), float(step["y"]), float(step["z"])
            )
        elif action == "fold_arm":
            self._call_service(self.fold_service, Trigger)
        elif action == "extend_arm":
            self._call_service(self.extend_service, Trigger)
        elif action == "wait":
            pass
        else:
            rospy.logwarn(
                "[fixed_schedule_runner] Unknown action '%s' at t=%.2f", action, step_time
            )

    def _call_service(self, name: str, srv_type: Any, *args: Any) -> None:
        try:
            rospy.wait_for_service(name, timeout=5.0)
            proxy = rospy.ServiceProxy(name, srv_type)
            proxy(*args)
            rospy.loginfo("[fixed_schedule_runner] Called service %s", name)
        except Exception as exc:
            rospy.logerr("[fixed_schedule_runner] Service %s failed: %s", name, exc)

    def _publish_waypoint(self, x: float, y: float, z: float) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.vector.x = x
        msg.vector.y = y
        msg.vector.z = z
        if any(math.isnan(v) for v in (x, y, z)):
            rospy.logerr("[fixed_schedule_runner] Refusing to publish NaN waypoint.")
            return
        self._waypoint_pub.publish(msg)
        rospy.loginfo(
            "[fixed_schedule_runner] Published waypoint x=%.3f y=%.3f z=%.3f",
            x,
            y,
            z,
        )


if __name__ == "__main__":
    rospy.init_node("fixed_schedule_runner", anonymous=False)
    FixedScheduleRunner().run()
