#!/usr/bin/env python

import rospy
import pyrealsense2 as rs
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState  # ROS message for joint angles
import swift

# Define the dimensions of the checkerboard (number of inner corners)
CHECKERBOARD = (9, 9)  # Adjust based on your checkerboard pattern

class RealSenseUR5Node:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('realsense_ur5_node', anonymous=True)
        
        # Publisher to publish joint angles (IK solutions)
        self.joint_pub = rospy.Publisher('/ur5_joint_states', JointState, queue_size=10)

        # Initialize the RealSense pipeline
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipe.start(cfg)

        # Launch the Swift simulator
        self.env = swift.Swift()
        self.env.launch(realtime=True)

        # Initialize the UR5 robot model
        self.robot = rtb.models.UR5()
        self.robot.q = np.deg2rad([0, -115, 115, -90, -90, 0])

        # Add the robot to the environment
        self.env.add(self.robot)

        # Define the time step
        self.dt = 0.01

        # Create Matplotlib figure and axis
        self.fig, self.ax = plt.subplots()
        self.image_plot = self.ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  # Placeholder image

    def interpolate_coordinates(self, x_pixel, y_pixel, img_width=640, img_height=480):
        x_mapped = -0.5 + (x_pixel / img_width) * 0.9
        y_mapped = (y_pixel / img_height) * 0.4
        return x_mapped, y_mapped

    def update_robot_pose(self, Tep):
        # Solve the inverse kinematics
        ik_solution = self.robot.ikine_LM(Tep, q0=self.robot.q)

        if ik_solution.success:
            rospy.loginfo("IK Solution found!")
            self.robot.q = ik_solution.q  # Apply the IK solution
            
            # Publish joint angles to ROS topic
            self.publish_joint_states(ik_solution.q)
        else:
            rospy.logwarn("IK Solution failed.")

        # Step the simulator
        self.env.step(self.dt)

    def publish_joint_states(self, joint_angles):
        # Create a JointState message
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [f'joint_{i+1}' for i in range(len(joint_angles))]
        joint_state_msg.position = joint_angles

        # Publish the message
        self.joint_pub.publish(joint_state_msg)

    def update_frame(self):
        frames = self.pipe.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return  # Skip if no frame captured

        # Convert frame to NumPy array and grayscale
        color_image = np.asanyarray(color_frame.get_data())
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Update the Matplotlib image
        self.image_plot.set_data(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.pause(0.01)  # Allow the plot to refresh

        # Detect the checkerboard corners
        ret2, corners = cv2.findChessboardCornersSB(
            gray_frame, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH
        )

        if ret2:
            corners_refined = cv2.cornerSubPix(
                gray_frame, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            # Extract corner locations (top-left, top-right, bottom-right, bottom-left)
            top_left = corners_refined[0][0]
            top_right = corners_refined[CHECKERBOARD[1] - 1][0]
            bottom_right = corners_refined[-1][0]
            bottom_left = corners_refined[-CHECKERBOARD[1]][0]

            # Calculate the center of the checkerboard
            center_x = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4
            center_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4

            mapped_x, mapped_y = self.interpolate_coordinates(center_x, center_y)
            Tep = sm.SE3.Trans(mapped_y, mapped_x, 0.4) * sm.SE3.Ry(np.pi / 2)

        else:
            Tep = sm.SE3(0.3, 0.3, 0.4) * sm.SE3.Ry(np.pi / 2)
            rospy.loginfo("No checkerboard detected, setting Tep to default.")

        # Update robot pose using IK solution
        self.update_robot_pose(Tep)

    def run(self):
        # Main loop
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.update_frame()
            rate.sleep()

        # Stop the RealSense pipeline when done
        self.pipe.stop()

        # Keep the simulator open
        self.env.hold()


if __name__ == '__main__':
    try:
        node = RealSenseUR5Node()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down RealSense UR5 node.")