import numpy as np
import pybullet as pyb  # Pybullet server
import pybullet_data
import time as time
import sys
import pinocchio as pin
from Paths import SOLO_URDF


class pybullet_simulator:
    """Wrapper for the PyBullet simulator to initialize the simulation, interact with it
    and use various utility functions

    Args:
        q_init (array): the default position of the robot
        envID (int): identifier of the current environment to be able to handle different scenarios
        terrain (string) see PyBulletSimulator.Init below
        sampling_interval (float) see PyBulletSimulator.Init below
        enable_pyb_GUI (bool): to display PyBullet GUI or not
        dt (float): time step of the inverse dynamics
    """

    def __init__(self, q_init, envID, terrain, sampling_interval, enable_pyb_GUI, dt=0.001):

        self.applied_force = np.zeros(3)

        # Start the client for PyBullet
        if enable_pyb_GUI:
            pyb.connect(pyb.GUI)
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
            pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 0)

        else:
            pyb.connect(pyb.DIRECT)
        # p.GUI for graphical version
        # p.DIRECT for non-graphical version

        # Load horizontal plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Roll and Pitch angle of the ground
        p_roll = 0.0 / 57.3  # Roll angle of ground
        p_pitch = 0.0 / 57.3  # Pitch angle of ground

        # Either use a flat ground or a rough terrain
        
        terrain_type = terrain.split(":")[0]
        if terrain_type == "plane" or terrain_type == "custom":
            self.planeId = pyb.loadURDF("plane.urdf")  # Flat plane
            self.planeIdbis = pyb.loadURDF("plane.urdf")  # Flat plane

            # Tune position and orientation of plane
            pyb.resetBasePositionAndOrientation(self.planeId, [0, 0, 0.0], pin.Quaternion(pin.rpy.rpyToMatrix(p_roll, p_pitch, 0.0)).coeffs())
            pyb.resetBasePositionAndOrientation(self.planeIdbis, [200.0, 0, -100.0 * np.sin(p_pitch)], pin.Quaternion(pin.rpy.rpyToMatrix(p_roll, p_pitch, 0.0)).coeffs())
        
            self.height_map = np.zeros((1,))
            self.sampling_bounds = [0, 0, 1, 1, 1]

        elif terrain_type == "rough":
            import random
            random.seed(41)
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
            heightPerturbationRange = 0.05

            numHeightfieldRows = 256*2
            numHeightfieldColumns = 256*2
            heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns
            height_prev = 0.0
            for j in range(int(numHeightfieldColumns/2)):
                for i in range(int(numHeightfieldRows/2)):
                    # uniform distribution
                    height = random.uniform(0, heightPerturbationRange)
                    # height = 0.25*np.sin(2*np.pi*(i-128)/46)  # sinus pattern
                    heightfieldData[2*i+2*j * numHeightfieldRows] = (height + height_prev) * 0.5
                    heightfieldData[2*i+1+2*j*numHeightfieldRows] = height
                    heightfieldData[2*i+(2*j+1) * numHeightfieldRows] = (height + height_prev) * 0.5
                    heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows] = height
                    height_prev = height

            # Create the collision shape based on the height field
            terrainShape = pyb.createCollisionShape(shapeType=pyb.GEOM_HEIGHTFIELD, meshScale=[0.05, 0.05, 1],
                                                    heightfieldTextureScaling=(numHeightfieldRows-1)/2,
                                                    heightfieldData=heightfieldData,
                                                    numHeightfieldRows=numHeightfieldRows,
                                                    numHeightfieldColumns=numHeightfieldColumns)
            self.planeId = pyb.createMultiBody(0, terrainShape)
            pyb.resetBasePositionAndOrientation(self.planeId, [0, 0, 0], [0, 0, 0, 1])
            pyb.changeVisualShape(self.planeId, -1, rgbaColor=[1, 1, 1, 1])

            self.height_map = np.zeros((1,)) # np.array(heightfieldData).reshape(numHeightfieldRows, numHeightfieldColumns) # not sure this is the correct way
            self.sampling_bounds = [0, 0, 1, 1, 1]

        if terrain_type == "custom":
            terrain = terrain.split(":")[1]
            import os
            
            minx, maxx, miny, maxy, minz, maxz = 0, sampling_interval, 0, sampling_interval, 0, 1
            self.terrain_objs = {}

            objects = os.listdir(terrain)
            for obj in objects:
                if len(obj) > 0 and obj[0] == "_":
                    continue

                urdf = os.path.join(terrain, obj, obj + ".urdf")
                print("Loading terrain object:", urdf, end="")
                self.terrain_objs[obj] = pyb.loadURDF(urdf)
                pyb.resetBasePositionAndOrientation(self.terrain_objs[obj], [0, 0, 0.001], [0, 0, 0, 1])                
                print(" DONE")
            
                a, b = pyb.getAABB(self.terrain_objs[obj])
                minx = min(minx, a[0])
                miny = min(miny, a[1])
                minz = min(minz, a[2])

                maxx = max(maxx, b[0])
                maxy = max(maxy, b[1])                
                maxz = max(maxz, b[2])

            minx -= sampling_interval
            maxx += sampling_interval
            miny -= sampling_interval
            maxy += sampling_interval

            x_pts = np.arange(minx, maxx+sampling_interval, sampling_interval)
            y_pts = np.arange(miny, maxy+sampling_interval, sampling_interval)
            x, y = np.meshgrid(x_pts, y_pts)

            print("Sampling height map...", end="")
            self.height_map = np.zeros_like(x)

            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    self.height_map[i,j] = pyb.rayTest([x[i,j], y[i,j], maxz+1], [x[i,j], y[i,j], minz-1])[0][3][2]
             
            self.sampling_bounds = [minx, miny, *self.height_map.shape[0:2], sampling_interval]
            self.height_map = self.height_map.flatten()

            print(" DONE")
            
        if envID == 1:

            # Add stairs with platform and bridge
            self.stairsId = pyb.loadURDF("bauzil_stairs.urdf")  # ,
            """basePosition=[-1.25, 3.5, -0.1],
                                 baseOrientation=pyb.getQuaternionFromEuler([0.0, 0.0, 3.1415]))"""
            pyb.changeDynamics(self.stairsId, -1, lateralFriction=1.0)

            # Create the red steps to act as small perturbations
            mesh_scale = [1.0, 0.1, 0.02]
            visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                  fileName="cube.obj",
                                                  halfExtents=[0.5, 0.5, 0.1],
                                                  rgbaColor=[1.0, 0.0, 0.0, 1.0],
                                                  specularColor=[0.4, .4, 0],
                                                  visualFramePosition=[0.0, 0.0, 0.0],
                                                  meshScale=mesh_scale)

            collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                        fileName="cube.obj",
                                                        collisionFramePosition=[0.0, 0.0, 0.0],
                                                        meshScale=mesh_scale)
            for i in range(4):
                tmpId = pyb.createMultiBody(baseMass=0.0,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0.0, 0.5+0.2*i, 0.01],
                                            useMaximalCoordinates=True)
                pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

            tmpId = pyb.createMultiBody(baseMass=0.0,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseCollisionShapeIndex=collisionShapeId,
                                        baseVisualShapeIndex=visualShapeId,
                                        basePosition=[0.5, 0.5+0.2*4, 0.01],
                                        useMaximalCoordinates=True)
            pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

            tmpId = pyb.createMultiBody(baseMass=0.0,
                                        baseInertialFramePosition=[0, 0, 0],
                                        baseCollisionShapeIndex=collisionShapeId,
                                        baseVisualShapeIndex=visualShapeId,
                                        basePosition=[0.5, 0.5+0.2*5, 0.01],
                                        useMaximalCoordinates=True)
            pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

            # Create the green steps to act as bigger perturbations
            mesh_scale = [0.2, 0.1, 0.01]
            visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                  fileName="cube.obj",
                                                  halfExtents=[0.5, 0.5, 0.1],
                                                  rgbaColor=[0.0, 1.0, 0.0, 1.0],
                                                  specularColor=[0.4, .4, 0],
                                                  visualFramePosition=[0.0, 0.0, 0.0],
                                                  meshScale=mesh_scale)

            collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                        fileName="cube.obj",
                                                        collisionFramePosition=[0.0, 0.0, 0.0],
                                                        meshScale=mesh_scale)

            for i in range(3):
                tmpId = pyb.createMultiBody(baseMass=0.0,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0.15 * (-1)**i, 0.9+0.2*i, 0.025],
                                            useMaximalCoordinates=True)
                pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

            # Create sphere obstacles that will be thrown toward the quadruped
            mesh_scale = [0.1, 0.1, 0.1]
            visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                  fileName="sphere_smooth.obj",
                                                  halfExtents=[0.5, 0.5, 0.1],
                                                  rgbaColor=[1.0, 0.0, 0.0, 1.0],
                                                  specularColor=[0.4, .4, 0],
                                                  visualFramePosition=[0.0, 0.0, 0.0],
                                                  meshScale=mesh_scale)

            collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                        fileName="sphere_smooth.obj",
                                                        collisionFramePosition=[0.0, 0.0, 0.0],
                                                        meshScale=mesh_scale)

            self.sphereId1 = pyb.createMultiBody(baseMass=0.4,
                                                 baseInertialFramePosition=[0, 0, 0],
                                                 baseCollisionShapeIndex=collisionShapeId,
                                                 baseVisualShapeIndex=visualShapeId,
                                                 basePosition=[-0.6, 0.9, 0.1],
                                                 useMaximalCoordinates=True)

            self.sphereId2 = pyb.createMultiBody(baseMass=0.4,
                                                 baseInertialFramePosition=[0, 0, 0],
                                                 baseCollisionShapeIndex=collisionShapeId,
                                                 baseVisualShapeIndex=visualShapeId,
                                                 basePosition=[0.6, 1.1, 0.1],
                                                 useMaximalCoordinates=True)

            # Flag to launch the two spheres in the environment toward the robot
            self.flag_sphere1 = True
            self.flag_sphere2 = True

        # Create blue spheres without collision box for debug purpose
        mesh_scale = [0.015, 0.015, 0.015]
        visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                              fileName="sphere_smooth.obj",
                                              halfExtents=[0.5, 0.5, 0.1],
                                              rgbaColor=[0.0, 0.0, 1.0, 1.0],
                                              specularColor=[0.4, .4, 0],
                                              visualFramePosition=[0.0, 0.0, 0.0],
                                              meshScale=mesh_scale)

        self.ftps_Ids = np.zeros((4, 5), dtype=int)
        for i in range(4):
            for j in range(5):
                self.ftps_Ids[i, j] = pyb.createMultiBody(baseMass=0.0,
                                                          baseInertialFramePosition=[0, 0, 0],
                                                          baseVisualShapeIndex=visualShapeId,
                                                          basePosition=[0.0, 0.0, -0.1],
                                                          useMaximalCoordinates=True)

        # Create green spheres without collision box for debug purpose
        visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                              fileName="sphere_smooth.obj",
                                              halfExtents=[0.5, 0.5, 0.1],
                                              rgbaColor=[0.0, 1.0, 0.0, 1.0],
                                              specularColor=[0.4, .4, 0],
                                              visualFramePosition=[0.0, 0.0, 0.0],
                                              meshScale=mesh_scale)
        self.ftps_Ids_deb = [0] * 4
        for i in range(4):
            self.ftps_Ids_deb[i] = pyb.createMultiBody(baseMass=0.0,
                                                       baseInertialFramePosition=[0, 0, 0],
                                                       baseVisualShapeIndex=visualShapeId,
                                                       basePosition=[0.0, 0.0, -0.1],
                                                       useMaximalCoordinates=True)

        # Create a red line for debug purpose
        self.lineId_red = []

        # Create a blue line for debug purpose
        self.lineId_blue = []

        """cubeStartPos = [0.0, 0.45, 0.0]
        cubeStartOrientation = pyb.getQuaternionFromEuler([0, 0, 0])
        self.cubeId = pyb.loadURDF("cube_small.urdf",
                                   cubeStartPos, cubeStartOrientation)
        pyb.changeDynamics(self.cubeId, -1, mass=0.5)
        print("Mass of cube:", pyb.getDynamicsInfo(self.cubeId, -1)[0])"""

        # Set the gravity
        pyb.setGravity(0, 0, -9.81)

        # Load Quadruped robot
        robotStartPos = [0, 0, 0.0]
        robotStartOrientation = pyb.getQuaternionFromEuler([0.0, 0.0, 0.0])  # -np.pi/2
        
        pyb.setAdditionalSearchPath(SOLO_URDF)
        self.robotId = pyb.loadURDF("solo12.urdf", robotStartPos, robotStartOrientation)

        # Disable default motor control for revolute joints
        self.revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        pyb.setJointMotorControlArray(self.robotId, jointIndices=self.revoluteJointIndices,
                                      controlMode=pyb.VELOCITY_CONTROL,
                                      targetVelocities=[0.0 for m in self.revoluteJointIndices],
                                      forces=[0.0 for m in self.revoluteJointIndices])

        # Initialize the robot in a specific configuration
        self.q_init = np.array([q_init]).transpose()
        pyb.resetJointStatesMultiDof(self.robotId, self.revoluteJointIndices, self.q_init)  # q0[7:])

        # Enable torque control for revolute joints
        jointTorques = [0.0 for m in self.revoluteJointIndices]
        pyb.setJointMotorControlArray(self.robotId, self.revoluteJointIndices,
                                      controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

        # Get position of feet in world frame with base at (0, 0, 0)
        feetLinksID = [3, 7, 11, 15]
        linkStates = pyb.getLinkStates(self.robotId, feetLinksID)

        # Get minimum height of feet (they are in the ground since base is at 0, 0, 0)
        z_min = linkStates[0][4][2]
        i_min = 0
        i = 1
        for link in linkStates[1:]:
            if link[4][2] < z_min:
                z_min = link[4][2]
                i_min = i
            i += 1

        # Set base at (0, 0, -z_min) so that the lowest foot is at z = 0
        pyb.resetBasePositionAndOrientation(self.robotId, [0.0, 0.0, -z_min], pin.Quaternion(pin.rpy.rpyToMatrix(p_roll, p_pitch, 0.0)).coeffs())

        # Progressively raise the base to achieve proper contact (take into account radius of the foot)
        while (pyb.getClosestPoints(self.robotId, self.planeId, distance=0.005,
                                    linkIndexA=feetLinksID[i_min]))[0][8] < -0.001:
            z_min -= 0.001
            pyb.resetBasePositionAndOrientation(self.robotId, [0.0, 0.0, -z_min], pin.Quaternion(pin.rpy.rpyToMatrix(p_roll, p_pitch, 0.0)).coeffs())

        # Fix the base in the world frame
        # pyb.createConstraint(self.robotId, -1, -1, -1, pyb.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, robotStartPos[2]])

        # Set time step for the simulation
        pyb.setTimeStep(dt)

        # Change camera position
        pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=45, cameraPitch=-39.9,
                                       cameraTargetPosition=[0.0, 0.0, robotStartPos[2]-0.2])

    def check_pyb_env(self, k, envID, velID, qmes12):
        """Check the state of the robot to trigger events and update camera

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            envID (int): Identifier of the current environment to be able to handle different scenarios
            velID (int): Identifier of the current velocity profile to be able to handle different scenarios
            qmes12 (19x1 array): the position/orientation of the trunk and angular position of actuators

        """

        # If spheres are loaded
        if envID == 1:
            # Check if the robot is in front of the first sphere to trigger it
            if self.flag_sphere1 and (qmes12[1, 0] >= 0.9):
                pyb.resetBaseVelocity(self.sphereId1, linearVelocity=[2.5, 0.0, 2.0])
                self.flag_sphere1 = False

            # Check if the robot is in front of the second sphere to trigger it
            if self.flag_sphere2 and (qmes12[1, 0] >= 1.1):
                pyb.resetBaseVelocity(self.sphereId2, linearVelocity=[-2.5, 0.0, 2.0])
                self.flag_sphere2 = False

            # Create the red steps to act as small perturbations
            """if k == 10:
                pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

                mesh_scale = [2.0, 2.0, 0.3]
                visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                      fileName="cube.obj",
                                                      halfExtents=[0.5, 0.5, 0.1],
                                                      rgbaColor=[0.0, 0.0, 1.0, 1.0],
                                                      specularColor=[0.4, .4, 0],
                                                      visualFramePosition=[0.0, 0.0, 0.0],
                                                      meshScale=mesh_scale)

                collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                            fileName="cube.obj",
                                                            collisionFramePosition=[0.0, 0.0, 0.0],
                                                            meshScale=mesh_scale)

                tmpId = pyb.createMultiBody(baseMass=0.0,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0.0, 0.0, 0.15],
                                            useMaximalCoordinates=True)
                pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)

                pyb.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.5], [0, 0, 0, 1])"""
            if k == 10:
                pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

                mesh_scale = [0.1, 0.1, 0.04]
                visualShapeId = pyb.createVisualShape(shapeType=pyb.GEOM_MESH,
                                                      fileName="cube.obj",
                                                      halfExtents=[0.5, 0.5, 0.1],
                                                      rgbaColor=[0.0, 0.0, 1.0, 1.0],
                                                      specularColor=[0.4, .4, 0],
                                                      visualFramePosition=[0.0, 0.0, 0.0],
                                                      meshScale=mesh_scale)

                collisionShapeId = pyb.createCollisionShape(shapeType=pyb.GEOM_MESH,
                                                            fileName="cube.obj",
                                                            collisionFramePosition=[0.0, 0.0, 0.0],
                                                            meshScale=mesh_scale)

                tmpId = pyb.createMultiBody(baseMass=0.0,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0.19, 0.15005, 0.02],
                                            useMaximalCoordinates=True)
                pyb.changeDynamics(tmpId, -1, lateralFriction=1.0)
                pyb.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.25], [0, 0, 0, 1])

        # Apply perturbations by directly applying external forces on the robot trunk
        if velID == 4:
            self.apply_external_force(k, 4250, 500, np.array([0.0, 0.0, -3.0]), np.zeros((3,)))
            self.apply_external_force(k, 5250, 500, np.array([0.0, +3.0, 0.0]), np.zeros((3,)))
        """if velID == 0:
            self.apply_external_force(k, 1000, 1000, np.array([0.0, 0.0, -6.0]), np.zeros((3,)))
            self.apply_external_force(k, 2000, 1000, np.array([0.0, +12.0, 0.0]), np.zeros((3,)))"""

        # Update the PyBullet camera on the robot position to do as if it was attached to the robot
        """pyb.resetDebugVisualizerCamera(cameraDistance=0.75, cameraYaw=+50, cameraPitch=-35,
                                       cameraTargetPosition=[qmes12[0, 0], qmes12[1, 0] + 0.0, 0.0])"""

        # Get the orientation of the robot to change the orientation of the camera with the rotation of the robot
        oMb_tmp = pin.SE3(pin.Quaternion(qmes12[3:7]), np.array([0.0, 0.0, 0.0]))
        RPY = pin.rpy.matrixToRpy(oMb_tmp.rotation)

        # Update the PyBullet camera on the robot position to do as if it was attached to the robot
        pyb.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=(0.0*RPY[2]*(180/3.1415)+45), cameraPitch=-39.9,
                                       cameraTargetPosition=[qmes12[0,0], qmes12[1,0] + 0.0, 0.0])

        return 0

    def retrieve_pyb_data(self):
        """Retrieve the position and orientation of the base in world frame as well as its linear and angular velocities
        """

        # Retrieve data from the simulation
        self.jointStates = pyb.getJointStates(self.robotId, self.revoluteJointIndices)  # State of all joints
        self.baseState = pyb.getBasePositionAndOrientation(self.robotId)  # Position and orientation of the trunk
        self.baseVel = pyb.getBaseVelocity(self.robotId)  # Velocity of the trunk

        # Joints configuration and velocity vector for free-flyer + 12 actuators
        self.qmes12 = np.vstack((np.array([self.baseState[0]]).T, np.array([self.baseState[1]]).T,
                                 np.array([[state[0] for state in self.jointStates]]).T))
        self.vmes12 = np.vstack((np.array([self.baseVel[0]]).T, np.array([self.baseVel[1]]).T,
                                 np.array([[state[1] for state in self.jointStates]]).T))

        """robotVirtualOrientation = pyb.getQuaternionFromEuler([0, 0, np.pi / 4])
        self.qmes12[3:7, 0] = robotVirtualOrientation"""

        # Add uncertainty to feedback from PyBullet to mimic imperfect measurements
        """tmp = np.array([pyb.getQuaternionFromEuler(pin.rpy.matrixToRpy(
            pin.Quaternion(self.qmes12[3:7, 0:1]).toRotationMatrix())
            + np.random.normal(0, 0.03, (3,)))])
        self.qmes12[3:7, 0] = tmp[0, :]
        self.vmes12[0:6, 0] += np.random.normal(0, 0.01, (6,))"""

        return 0

    def apply_external_force(self, k, start, duration, F, loc):
        """Apply an external force/momentum to the robot
        4-th order polynomial: zero force and force velocity at start and end
        (bell-like force trajectory)

        Args:
            k (int): numero of the current iteration of the simulation
            start (int): numero of the iteration for which the force should start to be applied
            duration (int): number of iterations the force should last
            F (3x array): components of the force in PyBullet world frame
            loc (3x array): position on the link where the force is applied
        """

        if ((k < start) or (k > (start+duration))):
            return 0.0
        """if k == start:
            print("Applying [", F[0], ", ", F[1], ", ", F[2], "]")"""

        ev = k - start
        t1 = duration
        A4 = 16 / (t1**4)
        A3 = - 2 * t1 * A4
        A2 = t1**2 * A4
        alpha = A2*ev**2 + A3*ev**3 + A4*ev**4

        self.applied_force[:] = alpha * F

        pyb.applyExternalForce(self.robotId, -1, alpha * F, loc, pyb.LINK_FRAME)

        return 0.0

    def get_to_default_position(self, qtarget):
        """Controler that tries to get the robot back to a default angular positions
        of its 12 actuators using polynomials to generate trajectories and a PD controller
        to make the actuators follow them

        Args:
            qtarget (12x1 array): the target position for the 12 actuators
        """

        qmes = np.zeros((12, 1))
        vmes = np.zeros((12, 1))

        # Retrieve angular position and velocity of actuators
        jointStates = pyb.getJointStates(self.robotId, self.revoluteJointIndices)
        qmes[:, 0] = [state[0] for state in jointStates]
        vmes[:, 0] = [state[1] for state in jointStates]

        # Create trajectory
        dt_traj = 0.0020
        t1 = 4.0  # seconds
        cpt = 0

        # PD settings
        P = 1.0 * 3.0
        D = 0.05 * np.array([[1.0, 0.3, 0.3, 1.0, 0.3, 0.3,
                              1.0, 0.3, 0.3, 1.0, 0.3, 0.3]]).transpose()

        while True or np.max(np.abs(qtarget - qmes)) > 0.1:

            time_loop = time.time()

            # Retrieve angular position and velocity of actuators
            jointStates = pyb.getJointStates(self.robotId, self.revoluteJointIndices)
            qmes[:, 0] = [state[0] for state in jointStates]
            vmes[:, 0] = [state[1] for state in jointStates]

            # Torque PD controller
            if (cpt * dt_traj < t1):
                ev = dt_traj * cpt
                A3 = 2 * (qmes - qtarget) / t1**3
                A2 = (-3/2) * t1 * A3
                qdes = qmes + A2*(ev**2) + A3*(ev**3)
                vdes = 2*A2*ev + 3*A3*(ev**2)
            jointTorques = P * (qdes - qmes) + D * (vdes - vmes)

            # Saturation to limit the maximal torque
            t_max = 2.5
            jointTorques[jointTorques > t_max] = t_max
            jointTorques[jointTorques < -t_max] = -t_max

            # Set control torque for all joints
            pyb.setJointMotorControlArray(self.robotId, self.revoluteJointIndices,
                                          controlMode=pyb.TORQUE_CONTROL, forces=jointTorques)

            # Compute one step of simulation
            pyb.stepSimulation()

            # Increment loop counter
            cpt += 1

            while (time.time() - time_loop) < dt_traj:
                pass


class Hardware():
    """Dummy class that simulates the Hardware class used to communicate with the real masterboard"""

    def __init__(self):

        self.is_timeout = False

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def IsTimeout(self):

        return self.is_timeout

    def Stop(self):

        return 0

    def imu_data_attitude(self, i):
        if i == 0:
            return self.roll
        elif i == 1:
            return self.pitch
        elif i == 2:
            return self.yaw

class IMU():
    """Dummy class that simulates the IMU class used to communicate with the real masterboard"""

    def __init__(self):
        self.linear_acceleration = np.zeros((3, ))
        self.accelerometer = np.zeros((3, ))
        self.gyroscope = np.zeros((3, ))
        self.attitude_euler = np.zeros((3, ))
        self.attitude_quaternion = np.array([[0.0, 0.0, 0.0, 1.0]])

class Powerboard():
    """Dummy class that simulates the Powerboard class used to communicate with the real masterboard"""

    def __init__(self):
        self.current = 0.0
        self.voltage = 0.0
        self.energy = 0.0

class Joints():
    """Dummy class that simulates the Joints class used to communicate with the real masterboard"""

    def __init__(self, parent_class):
        self.parent = parent_class
        self.positions = np.zeros((12, ))
        self.velocities = np.zeros((12, ))
        self.measured_torques = np.zeros((12, ))

    def set_torques(self, torques):
        """Set desired joint torques

        Args:
            torques (12 x 0): desired articular feedforward torques
        """
        # Save desired torques in a storage array
        self.parent.tau_ff = torques.copy()

        return

    def set_position_gains(self, P):
        """Set desired P gains for articular low level control

        Args:
            P (12 x 0 array): desired position gains
        """
        self.parent.P = P

    def set_velocity_gains(self, D):
        """Set desired D gains for articular low level control

        Args:
            D (12 x 0 array): desired velocity gains
        """
        self.parent.D = D

    def set_desired_positions(self, q_des):
        """Set desired joint positions

        Args:
            q_des (12 x 0 array): desired articular positions
        """
        self.parent.q_des = q_des

    def set_desired_velocities(self, v_des):
        """Set desired joint velocities

        Args:
            v_des (12 x 0 array): desired articular velocities
        """
        self.parent.v_des = v_des

class RobotInterface():
    """Dummy class that simulates the robot_interface class used to communicate with the real masterboard"""

    def __init__(self):
        pass

    def PrintStats(self):
        pass

class PyBulletSimulator():
    """Class that wraps a PyBullet simulation environment to seamlessly switch between the real robot or
    simulation by having the same interface in both cases (calling the same functions/variables)
    """

    def __init__(self):

        self.cpt = 0
        self.nb_motors = 12
        self.jointTorques = np.zeros(self.nb_motors)
        self.revoluteJointIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        self.is_timeout = False
        self.powerboard = Powerboard()
        self.imu = IMU()
        self.joints = Joints(self)
        self.robot_interface = RobotInterface()

        self.dummyPos = np.zeros(3)
        self.baseVel = np.zeros(3)
        self.debugPoints = -1

        # Measured data
        self.o_baseVel = np.zeros((3, 1))
        self.o_imuVel = np.zeros((3, 1))
        self.prev_o_imuVel = np.zeros((3, 1))

        # PD+ quantities
        self.P = 0.0
        self.D = 0.0
        self.q_des = np.zeros(12)
        self.v_des = np.zeros(12)
        self.tau_ff = np.zeros(12)

        # LPF parameters
        self._alpha = 0.0
        self.filterTorques = np.zeros(self.nb_motors)

    def Init(self, calibrateEncoders=False, q_init=None, envID=0, terrain_type="plane", sampling_interval=0.01, enable_pyb_GUI=False, dt=0.002, alpha=0.0):
        """Initialize the PyBullet simultor with a given environment and a given state of the robot

        Args:
            calibrateEncoders (bool): dummy variable, not used for simulation but used for real robot
            q_init (12 x 0 array): initial angular positions of the joints of the robot
            envID (int): which environment should be loaded in the simulation
            terrain_type (string): "plane", "rough" or "custom:/path/to/terrain"
            sampling_interval (float): interval in meters between each point in the global sampled height map
                                        (currently only supported with a custom terrain)
            enable_pyb_GUI (bool): to display PyBullet GUI or not
            dt (float): time step of the simulation
        """

        # Initialisation of the PyBullet simulator
        self.pyb_sim = pybullet_simulator(q_init, envID, terrain_type, sampling_interval, enable_pyb_GUI, dt)
        self.q_init = q_init
        self.joints.positions[:] = q_init
        self.dt = dt
        self.time_loop = time.time()

        self._alpha = alpha

        return
    
    def terrain_height(self, points, robot_frame=True):
        """Returns the terrain's sampled height at the specified location(s). Only for simulation.
        This is only implemented for custom terrains and will always return 0-filled arrays with
        plane or rough terrains.
        
        points (numpy array of shape [..., 2]): an array of 2D coordinates, either in
                                                the robot's frame or the world frame,
                                                depending on the value of parameter robot_frame
        robot_frame (bool): whether the coordinates are expressed in the robot's frame

        Returns:
            a numpy array with the same shape as 'points' (except the last dimension is removed)
        """
        if robot_frame:
            points = np.concatenate((points, np.zeros_like(points[..., 0:1])), axis=-1)[..., None]
            points = pin.rpy.rpyToMatrix(np.array([0, 0, self.imu.attitude_euler[2]]))[None] @ points
            points = (points.squeeze(-1) + self.dummyPos)[..., :2]

        _c = points.copy()
    
        points = (points - np.array(self.pyb_sim.sampling_bounds[:2])) / self.pyb_sim.sampling_bounds[4]
        points = points.astype(int)

        coords = np.ravel_multi_index((points[..., 1], points[..., 0]), self.pyb_sim.sampling_bounds[2:4], mode="clip")
       
       # UNCOMMENT TO SHOW MEASURED POINTS
      #  if self.debugPoints >= 0:
     #       pyb.removeUserDebugItem(self.debugPoints)
      #  self.debugPoints = pyb.addUserDebugPoints(np.concatenate((_c, self.pyb_sim.height_map[coords][..., None]), axis=-1),
       #                         np.tile(np.array([[1, 0, 0]]), (187, 1)),
      #                          pointSize=5,
       #                         lifeTime=0.1
       #                         )

        return self.pyb_sim.height_map[coords]

    def cross3(self, left, right):
        """Numpy is inefficient for this

        Args:
            left (3x0 array): left term of the cross product
            right (3x0 array): right term of the cross product
        """
        return np.array([[left[1] * right[2] - left[2] * right[1]],
                         [left[2] * right[0] - left[0] * right[2]],
                         [left[0] * right[1] - left[1] * right[0]]])

    def parse_sensor_data(self):
        """Retrieve data about the robot from the simulation to mimic what the masterboard does
        """

        # Position and velocity of actuators
        jointStates = pyb.getJointStates(self.pyb_sim.robotId, self.revoluteJointIndices)  # State of all joints
        self.joints.positions[:] = np.array([state[0] for state in jointStates])
        self.joints.velocities[:] = np.array([state[1] for state in jointStates])

        # Measured torques
        self.joints.measured_torques[:] = self.jointTorques[:].ravel()

        # Position and orientation of the trunk (PyBullet world frame)
        self.baseState = pyb.getBasePositionAndOrientation(self.pyb_sim.robotId)
        self.dummyHeight = np.array(self.baseState[0])
        self.dummyHeight[2] = 0.20
        self.dummyPos = np.array(self.baseState[0])

        # Linear and angular velocity of the trunk (PyBullet world frame)
        self.baseVel = pyb.getBaseVelocity(self.pyb_sim.robotId)
        # print("baseVel: ", np.array([self.baseVel[0]]))

        # Orientation of the base (quaternion)
        self.imu.attitude_quaternion[0,:] = np.array(self.baseState[1])
        self.rot_oMb = pin.Quaternion(self.imu.attitude_quaternion.T).toRotationMatrix()
        self.imu.attitude_euler[:] = pin.rpy.matrixToRpy(self.rot_oMb)
        self.oMb = pin.SE3(self.rot_oMb, np.array([self.dummyHeight]).transpose())

        # Angular velocities of the base
        self.imu.gyroscope[:] = (self.oMb.rotation.transpose() @ np.array([self.baseVel[1]]).transpose()).ravel()

        # Linear Acceleration of the base
        self.o_baseVel = np.array([self.baseVel[0]]).transpose()
        self.b_baseVel = (self.oMb.rotation.transpose() @ self.o_baseVel).ravel()

        self.o_imuVel = self.o_baseVel + self.oMb.rotation @ self.cross3(np.array([0.1163, 0.0, 0.02]), self.imu.gyroscope[:])

        self.imu.linear_acceleration[:] = (self.oMb.rotation.transpose() @ (self.o_imuVel - self.prev_o_imuVel)).ravel() / self.dt
        self.prev_o_imuVel[:, 0:1] = self.o_imuVel
        self.imu.accelerometer[:] = self.imu.linear_acceleration + \
            (self.oMb.rotation.transpose() @ np.array([[0.0], [0.0], [-9.81]])).ravel()

        return

    def send_command_and_wait_end_of_cycle(self, WaitEndOfCycle=True):
        """Send control commands to the robot

        Args:
            WaitEndOfCycle (bool): wait to have simulation time = real time
        """

        # Position and velocity of actuators
        jointStates = pyb.getJointStates(self.pyb_sim.robotId, self.revoluteJointIndices)  # State of all joints
        self.joints.positions[:] = np.array([state[0] for state in jointStates])
        self.joints.velocities[:] = np.array([state[1] for state in jointStates])

        # Compute PD torques
        tau_pd = self.P * (self.q_des - self.joints.positions) + self.D * (self.v_des - self.joints.velocities)

        # Save desired torques in a storage array
        self.jointTorques = (tau_pd + self.tau_ff).clip(-3,3)

        # Low pass filter the torques
        self.filterTorques = self._alpha * self.filterTorques + (1 - self._alpha) * self.jointTorques

        # Set control torque for all joints
        pyb.setJointMotorControlArray(self.pyb_sim.robotId, self.pyb_sim.revoluteJointIndices,
                                      controlMode=pyb.TORQUE_CONTROL, forces=self.filterTorques)

        # Compute one step of simulation
        pyb.stepSimulation()

        pyb.resetDebugVisualizerCamera(cameraDistance=0.9, 
                                       cameraYaw=(0.0*self.imu.attitude_euler[2]*(180/3.1415)-0),
                                       cameraPitch=-39.9,
                                       cameraTargetPosition=[self.baseState[0][0], self.baseState[0][1] + 0.0, 0.0])

        # Wait to have simulation time = real time
        if WaitEndOfCycle:
            while (time.time() - self.time_loop) < self.dt:
                pass
            self.cpt += 1

        self.time_loop = time.time()

        return

    def Print(self):
        """Print simulation parameters in the console"""

        np.set_printoptions(precision=2)
        # print(chr(27) + "[2J")
        print("#######")
        print("q_mes = ", self.joints.positions)
        print("v_mes = ", self.joints.velocities)
        print("torques = ", self.joints.measured_torques)
        print("orientation = ", self.imu.attitude_quaternion)
        print("lin acc = ", self.imu.linear_acceleration)
        print("ang vel = ", self.imu.gyroscope)
        sys.stdout.flush()

    def Stop(self):
        """Stop the simulation environment"""

        # Disconnect the PyBullet server (also close the GUI)
        pyb.disconnect()
