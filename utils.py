"""Misc functions for the replay"""
import numpy as np
import threading
from Logger import Logger
import pybullet as p
#import matplotlib.pyplot as plt
import os


def get_input():
    keystrk = input()
    # thread doesn't continue until key is pressed
    # and so it remains alive

def put_on_the_floor(device, params, q_init, tau_init):
    """Make the robot go to the default initial position and wait for the user
    to press the Enter key to start the main control loop

    Args:
        device: wrapper to communicate with the robot
        params: store control parameters
        q_init: default position of the robot
        tau_init: default torque after calibration
    """

    print("PUT ON THE FLOOR.")

    Kp_pos = 5.
    Kd_pos = 0.3

    device.joints.set_position_gains(Kp_pos * np.ones(12))
    device.joints.set_velocity_gains(Kd_pos * np.ones(12))
    device.joints.set_desired_positions(q_init)
    device.joints.set_desired_velocities(np.zeros(12))
    device.joints.set_torques(np.zeros(12))

    print("Init")
    print(q_init)

    i = threading.Thread(target=get_input)
    i.start()
    print("Put the robot on the floor and press Enter")

    while i.is_alive():
        device.parse_sensor_data()
        device.send_command_and_wait_end_of_cycle(params.dt)

    # Slow increase till we reach full starting torques
    duration_increase = 2.0  # in seconds
    steps = int(duration_increase / params.dt)
    for i in range(steps):
        device.joints.set_torques(tau_init * i / steps)
        device.parse_sensor_data()
        device.send_command_and_wait_end_of_cycle(params.dt)

    print("Start the motion.")

def initialize(params, Nobs, q_init, tau_init, N):
    """
    Initialize the connection with the robot or the simulation

    Args:
        params: store control parameters
        q_init: default position after calibration
        tau_init: default torque after calibration
        N: number of samples in the replay file
    """

    if params.SIMULATION:
        from PyBulletSimulator import PyBulletSimulator
    else:
        import libodri_control_interface_pywrap as oci
        # from qualisysClient import QualisysClient

    # Create the robot wrapper to communicate with it
    if params.SIMULATION:
        device = PyBulletSimulator()
        qc = None
    else:
        device = oci.robot_from_yaml_file(params.config_file)
        # qc = QualisysClient(ip="140.93.16.160", body_id=0)
        qc = None

    # If we want to log or plot, create logger
    if params.LOGGING or params.PLOTTING:
        logger = Logger(Nobs, device, qualisys=qc, logSize=N, SIMULATION=params.SIMULATION)
    else:
        logger = None

    # Initiate communication with the device and calibrate encoders
    if params.SIMULATION:
        device.Init(calibrateEncoders=True, q_init=q_init, envID=0,
                    terrain_type=params.terrain_type, sampling_interval=params.custom_sampling_interval,
                      enable_pyb_GUI=params.PYB_GUI, dt=params.dt, alpha=params.alpha)
    else:
        # Initialize the communication and the session.
        device.initialize(q_init[:])
        device.joints.set_zero_commands()

        device.parse_sensor_data()

        # Wait for Enter input before starting the control loop
        put_on_the_floor(device, params, q_init, tau_init)
    
    return device, logger, qc

def damping(device, params):
    """
    Apply damping to the actuators to slowly shut down on the floor

    Args:
        device: wrapper to communicate with the robot
        params: store control parameters
    """

    t = 0.0
    t_max = 2.5
    print("DAMPING")
    while ((not device.is_timeout) and (t < t_max)):

        device.parse_sensor_data()  # Retrieve data from IMU and Motion capture

        # Set desired quantities for the actuators
        device.joints.set_position_gains(np.zeros(12))
        device.joints.set_velocity_gains(0.1 * np.ones(12))
        device.joints.set_desired_positions(np.zeros(12))
        device.joints.set_desired_velocities(np.zeros(12))
        device.joints.set_torques(np.zeros(12))

        # Send command to the robot
        device.send_command_and_wait_end_of_cycle(params.dt)
        if (t % 1) < 5e-5:
            print('IMU attitude:', device.imu.attitude_euler)
            print('joint pos   :', device.joints.positions)
            print('joint vel   :', device.joints.velocities)
            device.robot_interface.PrintStats()

        t += params.dt

def shutdown(device, params, replay=None, logger=None):
    """
    Shut down the connection with the robot and plot/log if necessary

    Args:
        device: wrapper to communicate with the robot
        params: store control parameters
        logger: logger for the data
    """

    # Whatever happened we send 0 torques to the motors.
    device.joints.set_torques(np.zeros(12))
    device.send_command_and_wait_end_of_cycle(params.dt)

    if device.is_timeout:
        print("Masterboard timeout detected.")
        print("Either the masterboard has been shut down or there has been a connection issue with the cable/wifi.")

    # Save the logs of the Logger object
    if logger is not None and params.LOGGING:
        logger.saveAll()
        print("Log saved")

    # Plot recorded data
    if logger is not None and replay is not None and params.PLOTTING:
        logger.plotAll(params, replay)

    if params.SIMULATION:
        # Disconnect the PyBullet server (also close the GUI)
        device.Stop()

    print("End of script")

def save_frame_video(fid, save_dir):
    img = p.getCameraImage(width=1920, height=1080, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    plt.imsave(os.path.join(save_dir, 'frame_' + '0'*(6- len(str(fid))) + str(fid) + '.png'), img[2])
