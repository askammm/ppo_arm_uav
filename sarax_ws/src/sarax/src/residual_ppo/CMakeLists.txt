cmake_minimum_required(VERSION 3.0.2)
project(residual_ppo)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  mavros_msgs
  std_srvs
  m4e_mav_trajectory_planner
)

catkin_package()

catkin_install_python(PROGRAMS
  deploy/residual_ppo_node.py
  deploy/position_error_logger.py
  deploy/fixed_schedule_runner.py
  tools/plot_position_error_csv.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
