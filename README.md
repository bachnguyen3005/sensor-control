# sensor-control
This is a project for Sensor and Control subject where we need to perform Visual servoing that make the effector of UR5 following the checkerboard pattern.

I have used the position controller written in Python to control the UR5. 
The camera is the laptop's webcam which has been calibrated to achieve instric and extrinsic parameters. 


Please watch the youtube video for the demo: [![Video demo link](https://youtu.be/2KWZNxD7UUA/0.jpg)](https://youtu.be/2KWZNxD7UUA)

# realsense_ur5 
A package that contain a node to calculate publish the pose as the solution of IK from the RoboticsToolBox. 
This package is written for ROS-NOETIC

# Instruction
`test_visual_seroving_Facedown.py` is for eye-to-hand visual seroing simulation.
```
//TODO if you want, you can do another script for eye-on-hand. 

//TODO create another node for the circle pattern.
```
