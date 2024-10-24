import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the dimensions of the checkerboard (number of inner corners)
CHECKERBOARD = (9, 9)  # Adjust based on your checkerboard pattern

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the frame resolution to 1024x1024


# Create a Matplotlib figure and axis
fig, ax = plt.subplots()

# Capture an initial frame to set up the image display
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Initialize the plot with the first frame
image_plot = ax.imshow(frame_rgb)

# Function to interpolate pixel coordinates to the desired range
def interpolate_coordinates(x_pixel, y_pixel, img_width=600, img_height=400):
    # Cap x_pixel to img_width and y_pixel to img_height
    x_pixel = min(x_pixel, img_width)
    y_pixel = min(y_pixel, img_height)
    
    # Interpolate x-coordinate (horizontal axis)
    x_mapped = -0.5 + (x_pixel / img_width) * 0.9
    
    # Interpolate y-coordinate (vertical axis)
    y_mapped = (y_pixel / img_height) * 0.4
    
    return x_mapped, y_mapped

# Function to update the frame in the animation
def update_frame(i):
    ret, frame = cap.read()
    if not ret:
        return image_plot  # No update if frame is not captured
    
    # Convert frame from BGR to grayscale for checkerboard detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the checkerboard corners
    ret, corners = cv2.findChessboardCornersSB(gray_frame, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH)

    # If checkerboard is detected
    if ret:
        # Refine corner positions (optional for more accuracy)
        corners_refined = cv2.cornerSubPix(
            gray_frame, corners, (11, 11), (-1, -1), 
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        
        # Draw the corners on the frame
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners_refined, ret)
        
        # Extract corner locations (top-left, top-right, bottom-right, bottom-left)
        top_left = corners_refined[0][0]                      # First corner (top-left)
        top_right = corners_refined[CHECKERBOARD[1]-1][0]     # Last corner in the first row (top-right)
        bottom_right = corners_refined[-1][0]                 # Last corner (bottom-right)
        bottom_left = corners_refined[-CHECKERBOARD[1]][0]    # First corner in the last row (bottom-left)

        # Calculate the size of the checkerboard in pixels (width)
        checkerboard_width = top_right[0] - top_left[0]
        
        # Print the corner coordinates and the calculated size
        print("Top-left corner:", top_left)
        print("Top-right corner:", top_right)
        print("Bottom-right corner:", bottom_right)
        print("Bottom-left corner:", bottom_left)
        print(f"Checkerboard width (in pixels): {checkerboard_width:.2f}")
        
        # Calculate the center of the checkerboard as the average of the corner coordinates
        center_x = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4
        center_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4

        # Center of the checkerboard
        center = (center_x, center_y)

        # Interpolate the center coordinates to the specified range
        mapped_center_x, mapped_center_y = interpolate_coordinates(center_x, center_y, img_width=600, img_height=400)

        # Print the center coordinates in both pixel and mapped ranges
        print(f"Center of the checkerboard (pixels): ({center_x:.2f}, {center_y:.2f})")
        print(f"Center of the checkerboard (mapped): ({mapped_center_x:.2f}, {mapped_center_y:.2f})")

        # Draw the center point on the frame
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)  # Green dot for the center
        
        # Save the frame (image) to a file
        cv2.imwrite("checkerboard_image.png", frame)

        # Save the corner coordinates to a file
        corner_coords = np.array([top_left, top_right, bottom_right, bottom_left])
        np.savetxt("corner_coordinates.txt", corner_coords, fmt="%.2f", header="Top-left, Top-right, Bottom-right, Bottom-left")
        print("Image and corner coordinates saved.")

    # Convert frame from BGR to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update the image data
    image_plot.set_data(frame_rgb)
    
    return image_plot,

# Create the animation, which updates the plot with the new frame
ani = FuncAnimation(fig, update_frame, interval=50)

# Show the plot
plt.show()

# Release the webcam after closing the plot
cap.release()
