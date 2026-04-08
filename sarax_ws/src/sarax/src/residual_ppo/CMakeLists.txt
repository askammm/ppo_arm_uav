cmake_minimum_required(VERSION 3.0.2)
project(residual_ppo)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  deploy/residual_ppo_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
