from PyBulletSimulator import PyBulletSimulator
from Params import RLParams
import numpy as np
import pybullet as pyb  # Pybullet server
import os
import ctypes
import pinocchio as pin
from time import perf_counter as clock
import pybullet_data

try:
    from tqdm import tqdm
except ImportError:
    import warnings
    warnings.warn("Use tqdm library for bar plot in terminal.", ImportWarning)

params = RLParams()
device = PyBulletSimulator()
qc = None
logger = None

# q_init = np.array([
#                 0.3, 0.9, -1.64,
#                 -0.3, 0.9, -1.64,
#                 0.3, -0.9 , 1.64,
#                 -0.3, -0.9  , 1.64 ])
# params.q_init = q_init
# device.Init(calibrateEncoders=True, q_init=q_init, envID=0,
#                     terrain_type=params.terrain_type, sampling_interval=params.custom_sampling_interval,
#                       enable_pyb_GUI=params.PYB_GUI, dt=params.dt, alpha=params.alpha)


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

class MapHeader(ctypes.Structure):
    _fields_ = [
        ("size_x", ctypes.c_int),
        ("size_y", ctypes.c_int),
        ("x_init", ctypes.c_double),
        ("x_end", ctypes.c_double),
        ("y_init", ctypes.c_double),
        ("y_end", ctypes.c_double),
    ]


class Heightmap:
    def __init__(self):
        self.map_ = {"size_x": 0,
                     "size_y": 0,
                     "x_init": 0.0,
                     "x_end": 0.0,
                     "y_init": 0.0,
                     "y_end": 0.0}

        self.x_ = None
        self.y_ = None
        self.z_ = None
        self.dx_ = None
        self.dy_ = None


    def load_binary(self,filename):
        with open(filename, "rb") as f:
            # Read the MapHeader
            header_size = ctypes.sizeof(MapHeader)
            header_data = f.read(header_size)
            header = MapHeader.from_buffer_copy(header_data)

            # Read the heightmap data
            arr_bytes = f.read()
            heightmap_data = np.frombuffer(arr_bytes, dtype=np.float64).reshape(header.size_x, header.size_y)

            # Reconstruct Heightmap object
            self.z_ = heightmap_data
            self.map_['size_x'] = header.size_x
            self.map_['size_y'] = header.size_y
            self.map_['x_init'] = header.x_init
            self.map_['x_end'] = header.x_end
            self.map_['y_init'] = header.y_init
            self.map_['y_end'] = header.y_end

            self.dx_ = abs((header.x_init - header.x_end) / (header.size_x - 1))
            self.dy_ = abs((header.y_init - header.y_end) / (header.size_y - 1))


    def save_binary(self, filename):
        """
        Save heightmap matrix as binary file.
        Args :
        - filename (str) : name of the file saved.
        """
        arr_bytes = self.z_.astype(ctypes.c_double).tobytes()
        h = MapHeader(self.map_['size_x'],
                    self.map_['size_y'],
                    self.map_['x_init'],
                    self.map_['x_end'],
                    self.map_['y_init'],
                    self.map_['y_end'])
        h_bytes = bytearray(h)

        with open(filename, "ab") as f:
            f.truncate(0)
            f.write(h_bytes)
            f.write(arr_bytes)


    def create_heightmap(self):
        # Connect to PyBullet physics server
        pyb.connect(pyb.DIRECT)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 0)
        n_x = 100
        n_y = 100
        minx = -4
        maxx = 4
        miny = -4
        maxy = 4
        sampling_interval = 0.005

        minx, maxx, miny, maxy, minz, maxz = 0, sampling_interval, 0, sampling_interval, 0, 1
        terrain_objs = {}


        terrain = params.terrain_type.split(":")[1]
        objects = os.listdir(terrain)
        for obj in objects:
            if len(obj) > 0 and obj[0] == "_":
                continue

            urdf = os.path.join(terrain, obj)#, obj + ".urdf")
            print("Loading terrain object:", urdf, end="")
            terrain_objs[obj] = pyb.loadURDF(urdf)
            pyb.resetBasePositionAndOrientation(terrain_objs[obj], [0, 0, 0.001], [0, 0, 0, 1])
            print(" DONE")

            a, b = pyb.getAABB(terrain_objs[obj])
            minx = min(minx, a[0])
            miny = min(miny, a[1])
            minz = min(minz, a[2])

            maxx = max(maxx, b[0])
            maxy = max(maxy, b[1])
            maxz = max(maxz, b[2])



        self.x_ = np.arange(minx, maxx+sampling_interval, sampling_interval)
        self.y_ = np.arange(miny, maxy+sampling_interval, sampling_interval)
        self.z_ = np.zeros((len(self.x_), len(self.y_)))
        self.dx_ = sampling_interval
        self.dy_ = sampling_interval

        total_iterations = self.z_.size
        # Create progress bar
        progress_bar = tqdm(total=total_iterations, desc="Computing Heightmap")

        for i in range(len(self.x_)):
            for j in range(len(self.y_)):
                self.z_[i,j] = pyb.rayTest([self.x_[i], self.y_[j], maxz+1], [self.x_[i], self.y_[j], minz-1])[0][3][2]
                # Update progress bar
                progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        # Update the dictionnary containing main info about the heightmap.
        self.map_['size_x'] = len(self.x_)
        self.map_['size_y'] = len(self.y_)
        self.map_['x_init'] = self.x_[0]
        self.map_['x_end'] = self.x_[-1]
        self.map_['y_init'] = self.y_[0]
        self.map_['y_end'] = self.y_[-1]

        pyb.disconnect()

    def xIndex(self, x):
        return -1 if x < self.map_['x_init'] or x > self.map_['x_end'] else int(round((x - self.map_['x_init']) / self.dx_))

    def yIndex(self, y):
        return -1 if y < self.map_['y_init'] or y > self.map_['y_end'] else int(round((y - self.map_['y_init']) / self.dy_))

    def getHeight(self, x, y):
        iX = self.xIndex(x)
        iY = self.yIndex(y)
        return 0. if iX == -1 or iY == -1 else self.z_[iX, iY]

    def plot(self,alpha=1.0, ax=None):
        """
        Plot the heightmap
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        plt.ion()

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        xv, yv = np.meshgrid(self.x_, self.y_, sparse=False, indexing="ij")
        surf = ax.plot_surface(xv, yv, self.z_, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)


        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_zlim([np.min(self.z_), np.max(self.z_) + 1.0])

        return ax

    def plot_contour(self,alpha=1.0, ax=None):
        """
        Plot the heightmap using a contour plot
        """
        import matplotlib.pyplot as plt
        plt.ion()

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Create contour plot
        contour = ax.contourf(self.z_, cmap='viridis', alpha=alpha)

        # Add colorbar
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        return ax

def create_sphere():
    rgba = [0.41, 1.0, 0.0, 1.0]

    mesh_scale = [0.025, 0.025, 0.025]
    visualShapeId = pyb.createVisualShape(
        shapeType=pyb.GEOM_MESH,
        fileName="sphere_smooth.obj",
        halfExtents=[0.5, 0.5, 0.1],
        rgbaColor=rgba,
        specularColor=[0.4, 0.4, 0],
        visualFramePosition=[0.0, 0.0, 0.0],
        meshScale=mesh_scale,
    )
    Nx_sphere = 6
    Ny_sphere = 6
    sphereIds = np.zeros((Nx_sphere,Ny_sphere))
    for i in range(Nx_sphere):
        for j in range(Ny_sphere):
            # Get the heightmap value for the corresponding position
            sphereIds[i,j] = pyb.createMultiBody(
                baseMass=0.0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=visualShapeId,
                basePosition=[0., 0., -0.5],
                useMaximalCoordinates=True,
            )

    return sphereIds

def update_spheres(q, sphereIds):

    R = pin.rpy.rpyToMatrix(np.array([0.,0.,q[5]]))[:2,:2]
    T = q[:2]

    # Create spheres and update their positions based on the heightmap
    Nx_sphere = sphereIds.shape[0]
    Ny_sphere = sphereIds.shape[1]
    for i in range(Nx_sphere):
        for j in range(Ny_sphere):
            # Get the heightmap value for the corresponding position
            ii = int(i*Nx/Nx_sphere)
            jj = int(j*Ny/Ny_sphere)
            height = z_local[ii, jj]
            pos_l = np.array([x[ii],y[jj]])
            pos_w = R.T @ pos_l + T
            # Create a sphere at the calculated position with height as z-coordinate
            print("pos_w[0] : ", pos_w[0])
            print("pos_w[1] : ", pos_w[1])
            print("height : ", height)
            pyb.resetBasePositionAndOrientation(int(sphereIds[i,j]), [pos_w[0], pos_w[1], height], [0, 0, 0, 1])



if __name__ == "__main__":
    heightmap = Heightmap()
    heightmap.create_heightmap()
    heightmap.save_binary("heightmap.bin")
    heightmap = None
    heightmap = Heightmap()
    heightmap.load_binary("heightmap.bin")


    pyb.connect(pyb.GUI)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf")
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 0)
    terrain_objs = {}
    terrain = params.terrain_type.split(":")[1]
    objects = os.listdir(terrain)
    for obj in objects:
        if len(obj) > 0 and obj[0] == "_":
            continue

        urdf = os.path.join(terrain, obj)#, obj + ".urdf")
        print("Loading terrain object:", urdf, end="")
        terrain_objs[obj] = pyb.loadURDF(urdf)

    sphereIds = create_sphere()

    # Create height map loccally around the robot.

    # Pose of the robot : [x,y,z,roll,pitch,yaw]
    q = np.array([0.,0.,0.3,0.,0.,0.])

    Nx = 26 # Height, nb points pointing forward
    Ny = 21 # Width
    height = 0.5 # size of heightmap [m]
    width = 0.4 # size of heightmap [m]
    dx = 0.2 # shift forward in along x axis.

    # x and y are defined in robot frame.
    x = np.linspace(- (height/2 - dx) , height/2 + dx, Nx)
    y = np.linspace(-width/2, width/2, Ny)
    z_local = np.zeros((Nx,Ny))

    # Get the local heightmap.
    t0 = clock()
    R = pin.rpy.rpyToMatrix(np.array([0.,0.,q[5]]))[:2,:2]
    T = q[:2]
    for i in range(len(x)):
        for j in range(len(y)):
            p_world = R.T @ np.array([x[i],y[j]]) + T
            z_local[i,j] = heightmap.getHeight(p_world[0], p_world[1])

    t1 = clock()
    print("Step function [ms] : ", 1000 * (t1 - t0))

    update_spheres(q,sphereIds)