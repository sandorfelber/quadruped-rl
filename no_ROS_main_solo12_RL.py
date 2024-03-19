# coding: utf8

import os
import numpy as np
from Params import RLParams
from utils import *
import time
#from cpuMLP import PolicyMLP3, StateEstMLP2
# from numpy_mlp import MLP2
# from cpuMLP import Interface
from ControllerRL import ControllerRL
import pinocchio as pin
from Joystick import Joystick
from solo_local_height_maps import SoloLocalHeightMaps
#import rospy
#from geometry_msgs.msg import TransformStamped
#import tf.transformations
import matplotlib.pyplot as plt

PROFILER = False

np.set_printoptions(precision=3, linewidth=400)

class SoloRLDevice:

    def __init__(self, policy: ControllerRL, params: RLParams, run_name: str):
        self.policy = policy
        self.params = params
        self.run_name = run_name
        self.dummyPos = np.zeros(3)
        if params.measure_height:
            x, y = np.meshgrid(params.measure_x, params.measure_y, indexing="ij")
            self.measure_points = np.stack((x.flatten(), y.flatten()), axis=-1)
        self._pre_init()
        #self.listener()
        #LOAD FROM LAST PYBULLET SAVE
        self.pyb_global_height_map = np.loadtxt("height_map.txt", delimiter=",")
        self.pyb_sampling_bounds = np.loadtxt("sampling_bounds.txt", delimiter=",")
        
        #height_map_reshaped = self.pyb_global_height_map.reshape(387, 387)  # Reshape to 21 rows x 33 columns

        # Visualize the reshaped data
        # plt.imshow(height_map_reshaped, cmap='viridis', origin='lower', interpolation='none')
        # plt.colorbar(label='Height')
        # plt.title('Height Map Visualization')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.show()


    def _pre_init(self):
        if not self.params.SIMULATION:
            # this must be called once before connecting to the robot
            # because the first call to PyTorch takes too much time
            # (causes desync with the robot)
            self.policy.update_observation(np.zeros(3),
                                np.zeros((12,1)),
                                np.zeros((12,1)),
                                np.zeros((4,1)),
                                np.zeros((3,1)))
            self.policy.forward()

        params = self.params

        self.solo_local_height_map = SoloLocalHeightMaps()
        

        # Define joystick
        if params.USE_JOYSTICK:
            self.joystick = Joystick()
            self.joystick.update_v_ref()

        if params.USE_PREDEFINED:
            params.USE_JOYSTICK = False
            self.v_ref = 0.0  # Starting reference velocity
            self.alpha_v_ref = 0.03

        self.decimation = int(params.control_dt/params.dt)
        self.k = 0

    def init_robot_and_wait_floor(self):
         self.device, self.logger, _qc = initialize(self.params, self.policy._Nobs, self.params.q_init, np.zeros((12,)), 100000)

    def height_map(self):
        #heights = self.solo_local_height_map.flatbed()
        #Joystick.computeCode()
        if self.joystick.joystick_code == 1:
            #print("flatbed")
            heights = self.solo_local_height_map.flatbed()
        elif self.joystick.joystick_code == 2:
            heights = self.solo_local_height_map.trench()
            #print("TRENCH")
        elif self.joystick.joystick_code == 3:
            heights = self.solo_local_height_map.pre_trench()
            #print("pre_trench")
        elif self.joystick.joystick_code == 4:
            heights = self.solo_local_height_map.starting_descent()
        else:
            heights = self.solo_local_height_map.flatbed()
        heights_ = self.device.terrain_height(self.measure_points, heights)
        return self.device.dummyPos[2] - 0.215 - heights
    
    #OG WORKING FUNCTION
    # def height_map(self):
    #     if self.params.SIMULATION:
    #         heights = self.device.terrain_height(self.measure_points)
    #         height_map = np.zeros((33, 21))
    #         print(heights.shape)
    #         print(height_map.shape)
    #         print(height_map.flatten().shape)
    #         exit(0)
    #     else:
    #         heights = np.zeros(self.measure_points.shape[0]) # not implemented on real robot
    #     return self.device.dummyPos[2] - 0.215 - heights
    
    #Trying to implement height map query
    # def height_map(self):
    #     if not self.params.SIMULATION:
    #         heights = self.device.terrain_height(self.measure_points)
    #         #heights_real = self.terrain_height_real_robot(self.measure_points)
    #         #DEBUGGING:
    #         #height_difference = heights - heights_real
    #         #print("Height difference: ", height_difference)
    #         height_map_final = self.device.dummyPos[2] - 0.215 - heights

    #         #return self.vicon_positions[2] - 0.215 - heights_real
    #     else:
    #         #heights = np.zeros(self.measure_points.shape[0]) # not implemented on real robot
    #         heights_real = self.terrain_height_real_robot(self.measure_points)
    #         height_map_final = self.vicon_positions[2] - 0.215 - heights_real
    #         return height_map_final


    # def height_map(self):
    #     if self.params.SIMULATION:
    #         #heights = self.device.terrain_height(self.measure_points)
    #         heights = np.zeros(self.measure_points.shape[0]) # not implemented on real robot
    #         return self.device.dummyPos[2] - 0.215 - heights
    #     else:
    #         heights = np.zeros(self.measure_points.shape[0]) # np.zeros(self.measure_points.shape[0]) # not implemented on real robot
    #     return heights #- 0.215 #np.zeros(3)[2] - 0.215 - heights
    #     #return self.device.dummyPos[2] - 0.215 - heights

    def terrain_height_real_robot(self, points, robot_frame=True):
        """Returns the terrain's sampled height at the specified location(s).
        
        points (numpy array of shape [..., 2]): an array of 2D coordinates, either in
                                                the robot's frame or the world frame,
                                                depending on the value of parameter robot_frame
        robot_frame (bool): whether the coordinates are expressed in the robot's frame

        Returns:
            a numpy array with the same shape as 'points' (except the last dimension is removed)
        """
        #euler_attitude = 0
        if robot_frame:
            points = np.concatenate((points, np.zeros_like(points[..., 0:1])), axis=-1)[..., None]
            points = pin.rpy.rpyToMatrix(np.array([0, 0, self.vicon_attitude_euler[2]]))[None] @ points
            points = (points.squeeze(-1) + self.vicon_positions)[..., :2] #device.dummyPos)[..., :2]

        _c = points.copy()
        #print(self.device.pyb_sim.sampling_bounds[:2])
        #exit(0)
        points = (points - np.array(self.device.pyb_sim.sampling_bounds[:2])) / self.device.pyb_sim.sampling_bounds[4]
        points = points.astype(int)

        coords = np.ravel_multi_index((points[..., 1], points[..., 0]), self.device.pyb_sim.sampling_bounds[2:4], mode="clip")
        
        return self.pyb_global_height_map[coords]

    def damping_and_shutdown(self):     
        device = self.device

        # DAMPING TO GET ON THE GROUND PROGRESSIVELY *********************
        damping(device, self.params)

        # FINAL SHUTDOWN *************************************************
        shutdown(device, self.params)

    def _parse_joystick_cmd(self):
        joystick = self.joystick
        joystick.update_v_ref()
        vx = joystick.v_ref[0,0]
        vx = 0 if abs(vx) < 0.3 else vx
        wz = joystick.v_ref[-1,0]
        wz = 0 if abs(wz) < 0.3 else wz
        vy = joystick.v_ref[1,0] * int(self.params.enable_lateral_vel)
        vy = 0 if abs(vy) < 0.3 else vy
        return np.array([vx, vy, wz])
    
    def parse_file_loc_policy():
        file_loc_policy = "./tmp_checkpoints/policy_Feb25_21_38_13_ROBUST_TUNNELS.pt"
        #file_loc_policy = "/home/thomas_cbrs/Desktop/quadruped-rl/tmp_checkpoints/policy_Feb26.pt"
        return file_loc_policy
    
    # def _predefined_vel_cmd(self):
    #     t_rise = 100  # rising time to max vel
    #     t_duration = 500  # in number of iterations
    #     if self.k < t_rise + t_duration:
    #         v_max = 0.3 # in m/s
    #         v_gp = np.min([v_max * (self.k / t_rise), v_max])
    #     else:
    #         self.alpha_v_ref = 0.1
    #         v_gp = 0.0  # Stop the robot
    #     self.v_ref = self.alpha_v_ref * v_gp + (1 - self.alpha_v_ref) * self.v_ref  # Low-pass filter
    #     return np.array([self.v_ref, 0, 0]) 

    def _predefined_vel_cmd(self):
        t_rise = 100  # rising time to max velocity
        t_vel_duration = 100  # duration of max velocity
        t_ramp_down_duration = 100  # duration of ramping down
        t_stop = 200
        cycle_duration = t_rise + t_vel_duration + t_ramp_down_duration + t_stop # total duration of one cycle (walking + stopping)
        v_max = 0.0 # in m/s
        # Restart the cycle after every 'cycle_duration' iterations
        cycle_iteration = self.k % cycle_duration

        if cycle_iteration < t_rise:
            # Rising phase
            v_gp = v_max * (cycle_iteration / t_rise)
        elif cycle_iteration < t_rise + t_vel_duration:
            # Constant velocity phase
            v_gp = v_max
        elif cycle_iteration < t_rise + t_vel_duration:
            # Waiting phase before ramping down
            v_gp = v_max
        elif cycle_iteration < t_rise + t_vel_duration + t_ramp_down_duration:
            # Ramping down phase
            v_gp = v_max * (1 - (cycle_iteration - (t_rise + t_vel_duration)) / t_ramp_down_duration)
        else:
            # Stopped phase
            self.alpha_v_ref = 0.1
            v_gp = 0.0  # Stop the robot

        self.v_ref = self.alpha_v_ref * v_gp + (1 - self.alpha_v_ref) * self.v_ref  # Low-pass filter
        return np.array([self.v_ref, 0, 0])

    
    def _random_cmd(self):
        vx = np.random.uniform(-0.5 , 1.5)
        vx = 0 if abs(vx) < 0.3 else vx
        wz = np.random.uniform(-1, 1)
        wz = 0 if abs(wz) < 0.3 else wz
        return np.array([vx, 0, wz])

    def update_vel_command(self):
        params = self.params

        if params.USE_JOYSTICK:
            self.policy.vel_command = self._parse_joystick_cmd()
        
        elif params.USE_PREDEFINED:
            self.policy.vel_command = self._predefined_vel_cmd()
        
        elif self.k > 0 and self.k % 300 ==0:
            self.policy.vel_command = self._random_cmd()

    def run_cycle(self):
        policy = self.policy
        device = self.device
        decimation = self.decimation
        params = self.params

        baseVel = np.array(device.baseVel[0]) if params.SIMULATION else np.zeros(3)
        # import IPython
        # IPython.embed()
        policy.update_observation(baseVel,
                                  device.joints.positions.reshape((-1, 1)),
                                  device.joints.velocities.reshape((-1, 1)),
                                  device.imu.attitude_quaternion.reshape((-1, 1)),
                                  device.imu.gyroscope.reshape((-1, 1)),
                                  self.height_map())
        policy.forward()
        #policy.forward()
        # Set desired quantities for the actuators
        device.joints.set_position_gains(policy.P)
        device.joints.set_velocity_gains(policy.D)
        device.joints.set_desired_positions(policy.q_des)

        # if self.init_cycles <= 5:
            
        #     device.joints.set_desired_positions(policy.q_init)
        #     self.init_cycles = self.init_cycles +1

        # else:
        #     device.joints.set_desired_positions(policy.q_des)

        device.joints.set_desired_velocities(np.zeros((12,)))
        device.joints.set_torques(np.zeros((12,)))

        # Send command to the robot
        for j in range(decimation):
            if params.USE_JOYSTICK:
                self.joystick.update_v_ref()

            device.parse_sensor_data()
            device.send_command_and_wait_end_of_cycle(params.dt)

            if params.LOGGING or params.PLOTTING:
                self.logger.sample(policy, policy.q_des, policy.vel_command,
                                    policy.get_observation(), policy.get_computation_time())
                
        self.post_cycle_callback()
       
    def post_cycle_callback(self):
        self.k += 1
        if self.params.record_video and self.k % 10==0:
            save_frame_video(int(self.k//10), './recordings/')

    def save_plot_logs(self):
        params = self.params
        if params.LOGGING:
            self.logger.saveAll(suffix = "_" + self.run_name)
            print("log saved")

        if params.PLOTTING:
            self.logger.plotAll(params.dt, None)

    def control_loop(self):
        self.init_robot_and_wait_floor()
        while (not self.device.is_timeout and self.k < self.params.max_steps/self.decimation):
            self.run_cycle()
            self.update_vel_command()     
        
        if self.device.is_timeout:
            print("Time out detected..............")

        self.damping_and_shutdown()

    # def quaternion_callback(self, msg):
    #     # Extract the position (translation)
    #     position = np.array([
    #     msg.transform.translation.x,
    #     msg.transform.translation.y,
    #     msg.transform.translation.z
    #     ])

    #     # Assuming the quaternion is part of a geometry_msgs/TransformStamped message
    #     quaternion = (
    #         msg.transform.rotation.x,
    #         msg.transform.rotation.y,
    #         msg.transform.rotation.z,
    #         msg.transform.rotation.w
    #     )
    #     # X, Y, Z positions
    #     self.vicon_positions = position
    #     self.vicon_positions = [0, 0, 0.22]
    #     #print(self.vicon_positions)
    #     # Update the roll (x), pitch (y), and yaw (z) angles
    #     self.vicon_attitude_euler = tf.transformations.euler_from_quaternion(quaternion)  # Convert to Euler angles
        

    # def listener(self):
    #     if not self.params.SIMULATION or self.params.SIMULATION:
    #         rospy.init_node('quaternion_listener', anonymous=True)  # Initialize the ROS node
    #         rospy.Subscriber("/vicon/Solo/Solo", TransformStamped, self.quaternion_callback)  # Subscribe to the quaternion topic

            

def main():
    """
    Main function
    """

    params = RLParams()
    if not params.SIMULATION:  # When running on the real robot
        os.nice(-20)  #  Set the process to highest priority (from -20 highest to +20 lowest)

    # q_init = np.array([  0., 0.9, -1.64,
    #              0., 0.9, -1.64,

    #              0., -0.9 , 1.64,
    #              0., -0.9  , 1.64 ])
        
    q_init = np.array([
                0.3, 0.9, -1.64,
                -0.3, 0.9, -1.64,
                0.3, -0.9 , 1.64,
                -0.3, -0.9  , 1.64 ])
    params.q_init = q_init
    
    policy = ControllerRL(SoloRLDevice.parse_file_loc_policy(), q_init, params.measure_height)
    
    device = SoloRLDevice(policy, params, "solo")
    
    device.control_loop()
    device.save_plot_logs()
    
    quit()
    

if __name__ == "__main__":
    if PROFILER:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()

    main()
    #rospy.spin()

    if PROFILER:
        profiler.disable()
        profiler.print_stats("calls")
