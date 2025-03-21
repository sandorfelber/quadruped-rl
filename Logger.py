'''This class will log 1d array in Nd matrix from device and qualisys object'''
import numpy as np
from datetime import datetime as datetime
from time import time
import pinocchio as pin

class Logger():
    def __init__(self, Nobs, device=None, qualisys=None, logSize=60e3, SIMULATION=False):
        logSize = int(logSize)
        self.logSize = logSize
        nb_motors = 12
        self.k = 0
        self.device = device

        # Allocate the data:
        # IMU and actuators:
        self.q_mes = np.zeros([logSize, nb_motors])
        self.P = np.zeros([logSize, nb_motors])
        self.D = np.zeros([logSize, nb_motors])

        self.v_mes = np.zeros([logSize, nb_motors])
        self.torquesFromCurrentMeasurment = np.zeros([logSize, nb_motors])
        self.torquesRef = np.zeros([logSize, nb_motors])
        self.torquesP = np.zeros([logSize, nb_motors])
        self.torquesD = np.zeros([logSize, nb_motors])
        self.baseOrientation = np.zeros([logSize, 3])
        self.baseOrientationQuat = np.zeros([logSize, 4])
        self.baseAngularVelocity = np.zeros([logSize, 3])
        self.baseLinearAcceleration = np.zeros([logSize, 3])
        self.baseAccelerometer = np.zeros([logSize, 3])

        self.current = np.zeros(logSize)
        self.voltage = np.zeros(logSize)
        self.energy = np.zeros(logSize)

        self.q_des = np.zeros([logSize, nb_motors])
        self.v_des = np.zeros([logSize, 3])

        # Observation by neural network
        self.observation = np.zeros([logSize, Nobs])
        self.computation_time = np.zeros(logSize)

        # Motion capture:
        self.qualisys = qualisys
        self.mocapPosition = np.zeros([logSize, 3])
        self.mocapVelocity = np.zeros([logSize, 3])
        self.mocapAngularVelocity = np.zeros([logSize, 3])
        self.mocapOrientationMat9 = np.zeros([logSize, 3, 3])
        self.mocapOrientationQuat = np.zeros([logSize, 4])

        self.has_powerboard = device is not None and hasattr(device, "powerboard")

        # Timestamps
        self.tstamps = np.zeros(logSize)

        # Simulation
        self.SIMULATION = SIMULATION

    def sample(self, policy, qdes, vdes, obs, ctime):

        device = self.device

        # Logging from the device (data coming from the robot)
        self.q_mes[self.k] = device.joints.positions
        self.q_des[self.k] = qdes
        self.v_des[self.k] = vdes
        self.P[self.k] = policy.P
        self.D[self.k] = policy.D
        self.v_mes[self.k] = device.joints.velocities
        self.baseOrientation[self.k] = device.imu.attitude_euler
        self.baseOrientationQuat[self.k] = device.imu.attitude_quaternion
        self.baseAngularVelocity[self.k] = device.imu.gyroscope
        self.baseLinearAcceleration[self.k] = device.imu.linear_acceleration
        self.baseAccelerometer[self.k] = device.imu.accelerometer
        self.torquesFromCurrentMeasurment[self.k] = device.joints.measured_torques

        # Logging torques.
        self.torquesRef[self.k] = self.P[self.k] * (self.q_des[self.k] - self.q_mes[self.k]) - self.D[self.k] * self.v_mes[self.k] 
        self.torquesP[self.k] = self.P[self.k] * (self.q_des[self.k] - self.q_mes[self.k])
        self.torquesD[self.k] = - self.D[self.k] * self.v_mes[self.k]

        if self.has_powerboard:
            self.current[self.k] = device.powerboard.current
            self.voltage[self.k] = device.powerboard.voltage
            self.energy[self.k] = device.powerboard.energy

        # Logging observation of neural network
        self.observation[self.k] = obs
        self.computation_time[self.k] = ctime

        # Logging from qualisys (motion -- End of script --capture)
        qualisys = self.qualisys
        if qualisys is not None:
            self.mocapPosition[self.k] = qualisys.getPosition()
            self.mocapVelocity[self.k] = qualisys.getVelocity()
            self.mocapAngularVelocity[self.k] = qualisys.getAngularVelocity()
            self.mocapOrientationMat9[self.k] = qualisys.getOrientationMat9()
            self.mocapOrientationQuat[self.k] = qualisys.getOrientationQuat()
        elif self.SIMULATION:  # Logging from PyBullet simulator through fake device
            self.mocapPosition[self.k] = device.baseState[0]
            self.mocapVelocity[self.k] = device.baseVel[0]
            self.mocapAngularVelocity[self.k] = device.baseVel[1]
            self.mocapOrientationMat9[self.k] = device.rot_oMb
            self.mocapOrientationQuat[self.k] = device.baseState[1]

        # Logging timestamp
        self.tstamps[self.k] = time()

        self.k += 1

    def saveAll(self, fileName="dataSensors", suffix=""):
        date_str = datetime.now().strftime('_%Y_%m_%d_%H_%M')

        np.savez_compressed(fileName + date_str + suffix + ".npz",
                            q_mes=self.q_mes,
                            q_des=self.q_des,
                            P=self.P,
                            D=self.D,
                            v_mes=self.v_mes,
                            v_des=self.v_des,
                            baseOrientation=self.baseOrientation,
                            baseOrientationQuat=self.baseOrientationQuat,
                            baseAngularVelocity=self.baseAngularVelocity,
                            baseLinearAcceleration=self.baseLinearAcceleration,
                            baseAccelerometer=self.baseAccelerometer,
                            torquesFromCurrentMeasurment=self.torquesFromCurrentMeasurment,
                            torquesRef = self.torquesRef,
                            torquesD = self.torquesD,
                            torquesP = self.torquesP,
                            current=self.current,
                            voltage=self.voltage,
                            energy=self.energy,
                            observation=self.observation,
                            computation_time=self.computation_time,
                            mocapPosition=self.mocapPosition,
                            mocapVelocity=self.mocapVelocity,
                            mocapAngularVelocity=self.mocapAngularVelocity,
                            mocapOrientationMat9=self.mocapOrientationMat9,
                            mocapOrientationQuat=self.mocapOrientationQuat,
                            tstamps=self.tstamps)

    def loadAll(self, fileName):

        data = np.load(fileName)

        self.q_mes = data["q_mes"]
        self.q_des = data["q_des"]
        self.P = data["P"]
        self.D = data["D"]
        self.v_mes = data["v_mes"]
        self.v_des = data["v_des"] if "v_des" in data else self.v_des
        self.baseOrientation = data["baseOrientation"]
        self.baseOrientationQuat = data["baseOrientationQuat"]
        self.baseAngularVelocity = data["baseAngularVelocity"]
        self.baseLinearAcceleration = data["baseLinearAcceleration"]
        self.baseAccelerometer = data["baseAccelerometer"]
        self.torquesFromCurrentMeasurment = data["torquesFromCurrentMeasurment"]
        self.torquesRef = data["torquesRef"]
        self.torquesP = data["torquesP"]
        self.torquesD = data["torquesD"]
        self.current = data["current"]
        self.voltage = data["voltage"]
        self.energy = data["energy"]
        self.observation = data["observation"]
        self.computation_time = data["computation_time"]
        self.mocapPosition = data["mocapPosition"]
        self.mocapVelocity = data["mocapVelocity"]
        self.mocapAngularVelocity = data["mocapAngularVelocity"]
        self.mocapOrientationMat9 = data["mocapOrientationMat9"]
        self.mocapOrientationQuat = data["mocapOrientationQuat"]
        self.tstamps = data["tstamps"]

    def processMocap(self):

        self.mocap_pos = np.zeros([self.logSize, 3])
        self.mocap_h_v = np.zeros([self.logSize, 3])
        self.mocap_b_w = np.zeros([self.logSize, 3])
        self.mocap_RPY = np.zeros([self.logSize, 3])
   
        for i in range(self.logSize):
            self.mocap_RPY[i] = pin.rpy.matrixToRpy(pin.Quaternion(self.mocapOrientationQuat[i]).toRotationMatrix())

        # Robot world to Mocap initial translation and rotation
        mTo = np.array([self.mocapPosition[0, 0], self.mocapPosition[0, 1], 0.0])  
        mRo = pin.rpy.rpyToMatrix(0.0, 0.0, self.mocap_RPY[0, 2])

        for i in range(self.logSize):
            oRb = self.mocapOrientationMat9[i]

            oRh = pin.rpy.rpyToMatrix(0.0, 0.0, self.mocap_RPY[i, 2] - self.mocap_RPY[0, 2])

            self.mocap_h_v[i] = (oRh.transpose() @ mRo.transpose() @ self.mocapVelocity[i].reshape((3, 1))).ravel()
            self.mocap_b_w[i] = (oRb.transpose() @ self.mocapAngularVelocity[i].reshape((3, 1))).ravel()
            self.mocap_pos[i] = (mRo.transpose() @ (self.mocapPosition[i, :] - mTo).reshape((3, 1))).ravel()

    def custom_suptitle(self, name):
        from matplotlib import pyplot as plt

        fig = plt.gcf()
        fig.suptitle(name)
        fig.canvas.manager.set_window_title(name)

    def plot(self, title, nrows, ncols, t_range, legends, observations, references=None, mocaps=None, **kwargs):
        from matplotlib import pyplot as plt
        plt.figure()
        for i, obs in enumerate(observations):
            if i == 0:
                ax0 = plt.subplot(nrows, ncols, 1)
            else:
                plt.subplot(nrows, ncols, ncols * (i % nrows) + (i // nrows) + 1, sharex=ax0)
           
            if obs is not None:
                plt.plot(t_range[:self.k+1], obs[:self.k+1], color='b', linewidth=3, label='Observation', **kwargs)
            if references is not None and references[i] is not None:
                plt.plot(t_range[:self.k+1], references[i][:self.k+1], color='r', linewidth=3, label='Reference', **kwargs)
            if (self.SIMULATION or self.qualisys is not None) and mocaps is not None and mocaps[i] is not None:
                plt.plot(t_range[:self.k+1], mocaps[i][:self.k+1], color='g', linewidth=3, label='Motion capture', **kwargs)

            plt.xlabel("Time [s]")
            plt.ylabel(legends(i))
            plt.legend()

        self.custom_suptitle(title)

    def plot_PD(self, title, nrows, ncols, t_range, legends, observations, torquesRef, torquesP, torquesD, **kwargs):
        from matplotlib import pyplot as plt
        plt.figure()
        for i, obs in enumerate(observations):
            if i == 0:
                ax0 = plt.subplot(nrows, ncols, 1)
            else:
                plt.subplot(nrows, ncols, ncols * (i % nrows) + (i // nrows) + 1, sharex=ax0)
           
            if obs is not None:
                plt.plot(t_range[:self.k+1], obs[:self.k+1], color='b', linewidth=3, label='Observation', **kwargs)
            
            plt.plot(t_range[:self.k+1], torquesRef[i][:self.k+1], color='r', linewidth=3, label='Ref', **kwargs)
            plt.plot(t_range[:self.k+1], torquesP[i][:self.k+1], color='g', linewidth=3, label='P', **kwargs)
            plt.plot(t_range[:self.k+1], torquesD[i][:self.k+1], color='k', linewidth=3, label='D', **kwargs)
        

            plt.xlabel("Time [s]")
            plt.ylabel(legends(i))
            plt.legend() 


    def plotAll(self, dt, replay):
        from matplotlib import pyplot as plt

        t_range = np.array([k*dt for k in range(self.logSize)])

        self.processMocap()

        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        lgd = lambda unit: lambda i: "%s %s [%s]" % (lgd1[i % 3], lgd2[i//3],  unit)

        ####
        # Desired & Measured actuator positions
        ####     
        self.plot("Actuator positions", 3, 4, t_range, lgd("rad"), self.q_mes.T, self.q_des.T)
       
        ####
        # Desired & Measured actuator velocity
        ####
        self.plot("Actuator velocities", 3, 4, t_range, lgd("rad/s"), self.v_mes.T)
    
        ####
        # FF torques & FB torques & Sent torques & Meas torques
        ####
        self.plot("Torques", 3, 4, t_range[:-1], lgd("Nm"), self.torquesFromCurrentMeasurment[1:].T)

        ####
        # Custom FF torques & FB torques & Sent torques & Meas torques
        ####
        self.plot_PD("Torques_PD", 3, 4, t_range[:-1], lgd("Nm"), self.torquesFromCurrentMeasurment[1:].T,self.torquesRef[1:].T,
                     self.torquesP[1:].T, self.torquesD[1:].T )

        if self.has_powerboard:
            ####
            # Power supply profile
            ####
            plt.figure()
            for i in range(4):
                if i == 0:
                    ax0 = plt.subplot(4, 1, i+1)
                else:
                    plt.subplot(4, 1, i+1, sharex=ax0)

                if i == 0:
                    plt.plot(t_range, self.current[:], linewidth=2)
                    plt.ylabel("Bus current [A]")
                elif i == 1:
                    plt.plot(t_range, self.voltage[:], linewidth=2)
                    plt.ylabel("Bus voltage [V]")
                elif i == 2:
                    plt.plot(t_range, self.energy[:], linewidth=2)
                    plt.ylabel("Bus energy [J]")
                else:
                    power = self.current[:] * self.voltage[:]
                    plt.plot(t_range, power, linewidth=2)
                    N = 10
                    plt.plot(t_range[(N-1):], np.convolve(power, np.ones(N)/N, mode='valid'), linewidth=2)
                    plt.legend(["Raw", "Averaged 10 ms"])
                    plt.ylabel("Bus power [W]")
                    plt.xlabel("Time [s]")
            self.custom_suptitle("Energy profiles")

        ####
        # Measured & Reference position and orientation (ideal world frame)
        ####
        lgd = ["Pos X", "Pos Y", "Pos Z", "Roll", "Pitch", "Yaw"]
        obs = [None] * 3 + list(self.baseOrientation.T)
        mocap = list(self.mocap_pos.T) + list(self.mocap_RPY.T)
        self.plot("Position and orientation", 3, 2, t_range, lambda i: lgd[i], obs, mocaps=mocap)

        ####
        # Measured & Reference linear and angular velocities (horizontal frame)
        ####
        lgd = ["Linear vel X", "Linear vel Y", "Linear vel Z",
               "Angular vel Roll", "Angular vel Pitch", "Angular vel Yaw"]
        
        obs = [None] * 3 + list(self.baseAngularVelocity[:, :3].T)
        mocaps = list(self.mocap_h_v.T) + list(self.mocap_b_w.T)
        references = list(self.v_des[:, :2].T) + [None] * 3 + [self.v_des[:, 2]]

        self.plot("Position and orientation", 3, 2, t_range, lambda i: lgd[i], obs, references, mocaps)

        ####
        # Duration of each loop
        ####

        plt.figure()
        plt.plot(t_range[:-1], np.clip(np.diff(self.tstamps), 0, None))
        self.custom_suptitle("Duration of each loop [s]")

        ####
        # Duration of each loop
        ####

        plt.figure()
        plt.plot(t_range, self.computation_time)
        self.custom_suptitle("Computation time of the neural network [us]")

        ###############################
        # Display all graphs and wait #
        ###############################
        plt.show(block=True)

def analyse_consumption():
    import matplotlib.pyplot as plt
    import numpy as np
    import glob
    files = np.sort(glob.glob('test_policies_2022_08_05/*.npz'))[:-2]

    for file in files:
        data = np.load(file)
        current = data['current']
        voltage = data['voltage']
        power = data['current'] * data['voltage']
        mocap_v = data['mocapVelocity']
        mocap_oRh = data["mocapOrientationMat9"]
        mocap_h_v = np.zeros(mocap_v.shape)
        for k in range(mocap_v.shape[0]):
            mocap_h_v[k, :] = (mocap_oRh[k].transpose() @ mocap_v[k].reshape((-1, 1))).ravel()
        joystick = data['observation'][:, 9 ]
        mask = joystick>0.99
        # plt.plot(power)
        # plt.plot(joystick)
        # plt.plot(mask)
        # plt.figure()
        # plt.plot(power[mask])
        # plt.show()
        print("{} \t Average power going {:.3f} m/s : {:.3f} W ".format((file.split("_")[-1]).split(sep=".")[0],
                                                                        mocap_h_v[mask, 0].mean(), power[mask].mean()))

if __name__ == "__main__":

    import sys
    import os
    from sys import argv
    sys.path.insert(0, os.getcwd()) # adds current directory to python path

    # Data file name to load
    file_name = "dataSensors_2022_08_04_18_07_w3.npz"

    # Retrieve length of log file
    N = np.load(file_name)["tstamps"].shape[0]

    # Create loggers
    logger = Logger(logSize=N)

    # Load data from .npz file
    logger.loadAll(file_name)

    # Call all ploting functions
    logger.plotAll(0.001, None)
