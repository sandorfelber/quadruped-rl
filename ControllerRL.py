import torch
import numpy as np
import pinocchio as pin
from time import time as clock

# from isaac gym utils
# @torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class ControllerRL():
    def __init__(self, filename, q_init, measure_height):
        """
        Load a RL policy previously saved using torch.jit.save for cpu.
        Args:
            - filename (str) : path of the torch.jit model.
            - q_init (np.array(12)) : initial joint position
        """
        # Control gains
        self.P =  np.array([3.0, 3.0, 3.0]*4)
        self.D = np.array([0.2, 0.2, 0.2]*4)
        # self.P =  np.array([0.0, 0.0, 0.0]*4)
        # self.D = np.array([0., 0., 0.]*4)

        # Size
        self._Nobs = 741
        self._Nact = 12

        # Observation
        self.obs = np.zeros(self._Nobs)
        self.obs_torch = torch.zeros(1, self._Nobs)
        self.act = np.zeros(self._Nact)
        self.vel_command = np.array([0.,0.,0.])
        self.joints_pos = np.zeros(self._Nact)
        self.joints_vel = np.zeros(self._Nact)
        
        self.measure_height = measure_height
        if measure_height:
            self.height_map = np.zeros(self._Nobs - 48)

        self.orientation_quat = torch.zeros(1, 4)
        self.projected_gravity = np.zeros(3)
        self.base_speed = np.zeros(3)
        self.base_ang_vel = np.zeros(3)

        # Scales 
        self.scale_actions = 0.3
        self.scale_cmd_lin_vel = 2.0
        self.scale_cmd_ang_vel = 0.25
        self.scale_cmd = np.array([self.scale_cmd_lin_vel,self.scale_cmd_lin_vel,self.scale_cmd_ang_vel])
        self.scale_lin_vel = 2
        self.scale_ang_vel = 0.25
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 0.05
        self.scale_height_map = 5
        
        self.clip_observations = 100.
        self.clip_actions = 8.
        self.clip_height_map = 1

        # Load model
        self.model = torch.jit.load(filename)
        self.model.eval()

        # Initial joint position
        self.q_init = q_init
        self.q_des = q_init.copy()

        # timings
        self._t0 = 0
        self._t1 = 0
        self._t2 = 0
        self._t3 = 0

        self.init_cycles = True

        
    def forward(self):
          
        self._t2 = clock()

        self.act[:] = self.model(self.obs_torch).detach()[0]
        if self.init_cycles == True:  
            self.act[:] = np.clip(self.act, -self.clip_actions/4, self.clip_actions/4)
            self.init_cycles = False
        else:    
            self.act[:] = np.clip(self.act, -self.clip_actions, self.clip_actions)
        self._t3 = clock()
        self.q_des[:] = self.act * self.scale_actions + self.q_init

        # if self.init_cycles == True:    
        #     self.q_des[:] = self.q_init
        #     self.init_cycles = False
        # else:
        #     pass

        return self.q_des #self.q_init
    
    def update_observation(self, base_speed, joints_pos, joints_vel, orientation_quat, imu_gyro, height_map=None):
        self._t0 = clock()

        self.joints_pos[:] = joints_pos[:,0]
        self.joints_vel[:] = joints_vel[:,0]

        # Project gravity vector in the robot world frame
        self.orientation_quat[:] = torch.from_numpy(orientation_quat.flatten())
        self.projected_gravity[:] = quat_rotate_inverse(self.orientation_quat, torch.tensor([[0.0, 0.0, -1.0]]).float())

        # self.projected_gravity[:] = imu_ori.flatten()
        self.base_speed[:] = base_speed.flatten()
        self.base_ang_vel[:] = imu_gyro.flatten()
        
        self.obs[:48] = np.hstack([self.base_speed * self.scale_lin_vel,
                                  self.base_ang_vel * self.scale_ang_vel,
                                  self.projected_gravity,
                                  self.vel_command * self.scale_cmd,
                                  (self.joints_pos - self.q_init) * self.scale_dof_pos,
                                  self.joints_vel * self.scale_dof_vel,
                                  self.act
                                  ])
        
        if height_map is not None:
            self.height_map[:] = height_map.clip(-self.clip_height_map, self.clip_height_map)
            self.obs[48:] = self.height_map * self.scale_height_map

        self.obs[:] = np.clip(self.obs, -self.clip_observations, self.clip_observations)
        self.obs_torch[:] = torch.from_numpy(self.obs)

        self._t1 = clock()

    def get_observation(self):
        return self.obs
    
      
    def get_computation_time(self):
        # Computation time in us
        return (self._t3 - self._t2) * 1e6