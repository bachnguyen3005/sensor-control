import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the dimensions of the checkerboard (number of inner corners)
CHECKERBOARD = (9, 9)  # Adjust based on your checkerboard pattern

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Launch the Swift simulator
env = swift.Swift()
env.launch(realtime=True)

# Initialize the UR5 robot model
panda = rtb.models.UR5()
q0 = np.deg2rad([0, -115, 115, -90, -90, 0])  # Initial joint configuration
print(q0)
panda.q = q0

# Add the robot to the environment
env.add(panda)

# Define the time step
dt = 0.01

# Function to interpolate the checkerboard center coordinate
def interpolate_coordinates(x_pixel, y_pixel, img_width=600, img_height=400):
    x_pixel = min(x_pixel, img_width)
    y_pixel = min(y_pixel, img_height)
    x_mapped = -0.5 + (x_pixel / img_width) * 0.9
    y_mapped = (y_pixel / img_height) * 0.4
    return x_mapped, y_mapped

# Function to update the robot's pose using IK
def update_robot_pose(Tep):
    # Solve inverse kinematics to find joint angles for the target pose
    ik_solution = panda.ikine_LM(Tep, q0=panda.q)

    if ik_solution.success:
        print("IK Solution found!")
        print(f"Moving to: {ik_solution.q}")
        panda.q = ik_solution.q  # Apply the IK solution
    else:
        print("IK Solution failed.")

    # Step the simulation to render the new pose
    env.step(dt)

# Create a Matplotlib figure and axis for webcam display
fig, ax = plt.subplots()
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image_plot = ax.imshow(frame_rgb)

# Function to update the frame in the animation
def update_frame(i):
    ret, frame = cap.read()
    if not ret:
        return image_plot  # No update if frame is not captured

    # Convert frame from BGR to grayscale for checkerboard detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the checkerboard corners
    ret2, corners = cv2.findChessboardCornersSB(
        gray_frame, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH
    )

    if ret2:
        # Refine corner positions (optional for more accuracy)
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

        # Interpolate the center coordinates to the robot coordinate range
        mapped_center_x, mapped_center_y = interpolate_coordinates(
            center_x, center_y, img_width=600, img_height=400
        )

        print(f"Center (pixels): ({center_x:.2f}, {center_y:.2f})")
        print(f"Center (mapped): ({mapped_center_x:.2f}, {mapped_center_y:.2f})")

        # Define the target pose using the mapped coordinates
        Tep =sm.SE3.Trans(mapped_center_y, mapped_center_x, 0.4) * sm.SE3.Ry(np.pi / 2)

    else:
        # If no checkerboard is detected, move to a default pose
        Tep = sm.SE3(0.3, 0.3, 0.4) * sm.SE3.Ry(np.pi / 2)
        print("No checkerboard detected, setting Tep to default.")

    # Update the robot's pose using IK
    update_robot_pose(Tep)

    # Convert frame from BGR to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Update the image data
    image_plot.set_data(frame_rgb)

    return image_plot

# Create the animation, which updates the plot with the new frame
ani = FuncAnimation(fig, update_frame, interval=50)

# Show the plot
plt.show()

# Release the webcam after closing the plot
cap.release()

# Keep the simulator open to visualize the pose
env.hold()
