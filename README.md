# ppo_arm_uavd
this projrct is to realize collaborative control between robotic arms and drones and use gazebo to simulation<br>
system:ubuntu 20.04<br>
this project is based on Sarax(join a robot arm in gazebo )<br>
what new between sarax and this project: <br>
add a small ppo network (inpput:the Pose error of arm and drone /desired arms' trajectory command for the next moment)(output:force/Torque)(in sarax_ws/src/sarax/src/residual_ppo)

