# Require libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from spatialmath.base import *
from roboticstoolbox import models
from machinevisiontoolbox import CentralCamera
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Useful variables
from math import pi

# -----------------------------------------------------------------------------------#
def VisualServoingTest():
    plt.close('all')

    ## 1.1 Definitions
    # Create image target (points in the image plane)
    p_star = np.array([[662, 362, 362, 662],
                       [362, 362, 662, 662]])

    # Create 3D points
    p = np.array([[1.8, 1.8, 1.8, 1.8],
                  [-0.25, 0.25, 0.25, -0.25],
                  [1.25, 1.25, 0.75, 0.75]])
    
    # Make a UR5
    r = models.DH.UR10()

    # Initial pose
    q0 = [pi/2, -pi/3, -pi/3, -pi/6, 0, 0]
    workspace = [-2,2,-2,2,0,2] 

    # Add the camera
    # Mod Camera parameter as required
    cam = CentralCamera(f= 0.08, rho = 10e-5, imagesize = [1024, 1024], 
                        pp = [512, 512], name = 'UR5camera')
    
    # frame rate
    fps = 25

    # Define values
    # gain of the controller
    lambda_ = 0.6
    # depth of the IBVS
    depth = np.mean(p[0,:])

    # ---------------------------------------------------------------------------------------#
    ## 1.2 Initialise simulation (Display in 3D)

    # Display UR5
    r.q = q0
    Tc0 = r.fkine(q0)
    fig = r.plot(q0, limits= workspace)
    fig.ax.set_box_aspect([workspace[i+1] - workspace[i] for i in range(0,len(workspace),2)])

    # Plot camera and points
    cam.pose = Tc0

    # Display points in 3D and the camera
    cam.plot(scale= 0.25, solid= True, alpha= 0.8, ax = fig.ax)
    [plot_sphere(radius=0.05, centre=p[:, i], color='b') for i in range(p.shape[1])]

    # ---------------------------------------------------------------------------------------#
    ## 1.3 Initialise Simulation (Display in Image view)

    # Project points to the image
    # Camera view and plotting
    cam.clf()
    cam.plot_point(p_star, '*', pose = Tc0)       # create the camera view
    cam.plot_point(p, 'o', pose= Tc0)             # create the camera view

    # Initialise display array
    hist = dict(uv = [], vel = [], e = [], en = [], jcond = [], Tcam = [], vel_p = [], uv_p = [], qp = [], q = [])
    
    # ---------------------------------------------------------------------------------------#
    # 1.4 Loop
    # loop of the visual servoing
    ksteps = 0
    input('Enter to continue\n')
    while True:
        # compute the view of the camera
        uv = cam.plot_point(p, pose = cam.pose, alpha = 0.25)
        
        # compute image plane error as a column
        e = p_star - uv # feature error
        e = e.reshape(-1, order = 'F')
        zest = []

        # compute the Jacobian
        if not np.any(depth):
            # exact depth from simulation (not possible in practice)
            pt = homtrans(linalg.inv(cam.pose), p)
            J = cam.visjac_p(uv, pt[2,:])
        elif np.any(zest):
            J = cam.visjac_p(uv, zest)
        else:
            J = cam.visjac_p(uv, depth)
        
        # compute the velocity of camera in camera frame
        try:
            v = lambda_ * linalg.pinv(J) @ e
        except:
            return None
        
        # compute robot's Jacobian and inverse
        J2 = r.jacobe(r.q)
        Jinv = linalg.pinv(J2)
        # get joint velocities
        qp = Jinv @ v

        # Maximum angular velocity cannot exceed 180 degrees/s
        qp = np.clip(qp, -pi, pi)
        v = J2 @ qp
        print(f'v: [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}, {v[3]:.3f}, {v[4]:.3f}, {v[5]:.3f}]')

        # Update joints
        r.q += 1/fps * qp

        # Get camera location
        cam.pose = r.fkine(r.q)

        fig.step(1/fps)
        cam_redraw(cam, fig.ax) # should be used with care
        ksteps += 1

        # update the history variables
        hist['uv'].append(uv.reshape(-1, order = 'F'))
        hist['vel'].append(v)
        hist['e'].append(e)
        hist['en'].append(linalg.norm(e))
        hist['jcond'].append(np.linalg.cond(J))
        hist['Tcam'].append(cam.pose.copy())
        hist['vel_p'].append(v)
        hist['uv_p'].append(uv)
        hist['qp'].append(qp)
        hist['q'].append(r.q.copy())

        if ksteps >= 200:
            break
    
    # Modify history data
    for key in hist:
        hist[key] = np.column_stack(hist[key])
    
    # ---------------------------------------------------------------------------------------#
    ## 1.5 Plot results
    plot_p(hist, cam, p_star)
    plot_vel(hist)
    plot_camera(hist)
    plot_robjointvel(hist)
    plot_robjointpos(hist)
    input('Enter to continue\n')

# ---------------------------------------------------------------------------------------#
## Functions for plotting
def plot_p(history, camera, uv_star = None):
    """
    VisualServo.plot_p Plot feature trajectory

    VS.plot_p() plots the feature values versus time.

    See also VS.plot_vel, VS.plot_error, VS.plot_camera,
    VS.plot_jcond, VS.plot_z, VS.plot_error, VS.plot_all.
    """
    plt.figure('Feature trajectory')
    plt.cla()
    ax = plt.subplot()
    ax.set_xlim(0, camera.nu)
    ax.set_ylim(0, camera.nv)

    if not np.any(history):
        return None
    # image plane trajectory
    uv = history['uv']
    # result is a vector with row per time step, each row is u1, v1, u2, v2 ...
    for i in range(0, uv.shape[0]-1, 2):
        ax.plot(uv[i,:], uv[i+1,:], linewidth = 2)
    
    plot_polygon(uv[:,0].reshape(-1,2).T, 'o--', ax= ax, close= True)

    if uv_star is not None:
        plot_polygon(uv_star, '*--', ax = ax, close = True)
    else:
        plot_polygon(uv[:,-1].reshape(-1,2), 'rd--', ax = ax, close= True)
    
    ax.invert_yaxis()
    ax.set_xlabel('u (pixels)')
    ax.set_ylabel('v (pixels)')
    ax.legend([f'point {i+1}' for i in range(int(uv.shape[0]/2))] + ['start points', 'target points'])

# ---------------------------------------------------------------------------------------#
def plot_vel(history):
    """
    VisualServo.plot_vel Plot camera velocity

    VS.plot_vel() plots the camera velocity versus time.

    See also VS.plot_p, VS.plot_error, VS.plot_camera,
    VS.plot_jcond, VS.plot_z, VS.plot_error, VS.plot_all.
    """
    plt.figure('Camera trajectory')
    plt.cla()
    ax = plt.subplot()

    if not np.any(history):
        return None
    vel = history['vel'].T
    ax.plot(vel[:,:3], '-', linewidth = 1)
    ax.plot(vel[:,3:], '--', linewidth = 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cartesian velocity')
    ax.legend(['vx', 'vy', 'vz', '\u03C9x', '\u03C9y', '\u03C9z'])

# ---------------------------------------------------------------------------------------#
def plot_camera(history):
    """
    VisualServo.plot_camera Plot camera pose

    VS.plot_camera() plots the camera pose versus time.

    See also VS.plot_p, VS.plot_error, VS.plot_vel,
    VS.plot_jcond, VS.plot_z, VS.plot_error, VS.plot_all.
    """
    plt.figure('Camera Pose')
    plt.clf()
    if not np.any(history):
        return None
    # Cartesian camera position vs time
    T = history['Tcam'].reshape(4,-1,4)

    plt.subplot(211)
    plt.plot(T[:3,:,3].T, linewidth = 1)
    plt.ylabel('camera position')
    plt.legend(['X', 'Y', 'Z'])

    plt.subplot(212)
    plt.plot(np.row_stack([tr2rpy(T[:3,i,:3], order= 'xyz') for i in range(T.shape[1])]), linewidth = 1)
    plt.ylabel('camera orientation')
    plt.xlabel('time')
    plt.legend(['R', 'P', 'Y'])

# ---------------------------------------------------------------------------------------#
def plot_robjointvel(history):
    plt.figure('Joint Velocity')
    plt.cla()
    if not np.any(history):
        return None
    vel = history['qp'].T
    plt.plot(vel, '-', linewidth = 1)
    plt.ylabel('Joint velocity')
    plt.xlabel('Time')
    plt.legend([f'\u03C9{i+1}' for i in range(vel.shape[1])])

# ---------------------------------------------------------------------------------------#
def plot_robjointpos(history):
    plt.figure('Joint Angle')
    plt.cla()
    if not np.any(history):
        return None
    pos = history['q'].T
    plt.plot(pos, '-', linewidth = 1)
    plt.ylabel('Joint angle')
    plt.xlabel('Time')
    plt.legend([f'\u03C9{i+1}' for i in range(pos.shape[1])])

# ---------------------------------------------------------------------------------------#
def cam_redraw(cam, ax):
    """
    Input: 
    - CentralCamera object and the current axes it is plotted in
    
    This function should be used with care as it 
    deletes all objects of type poly3dcollection.
    It is used to update the camera visualisation 
    because the plot function of the CentralCamera 
    class does not return the plotted object.
    """
    for artist in ax.get_children():
        if isinstance(artist, Poly3DCollection):
            artist.remove()       
    cam.plot(scale= 0.25, solid= True, alpha= 0.8, ax = ax)

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
   VisualServoingTest()