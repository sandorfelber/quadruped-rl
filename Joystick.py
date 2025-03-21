# coding: utf8

import numpy as np
import gamepadClient as gC

LAASGAMEPAD=0

class Joystick:
    """Joystick-like controller that outputs the reference velocity in local frame

    Args:
        predefined (bool): use either a predefined velocity profile (True) or a gamepad (False)
    """

    def __init__(self):

        # Reference velocity in local frame
        self.v_ref = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
        self.v_gp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        self.reduced = False
        self.stop = False

        self.alpha = 0.003  # Coefficient to low pass the joystick velocity

        # Joystick variables (linear and angular velocity and their scaling for the joystick)
        self.vX = 0.
        self.vY = 0.
        self.vYaw = 0.
        self.vZ = 0.
        self.VxScale = 0.35
        self.VyScale = 0.45
        self.vYawScale = 1.
        self.vZScale = 0.3
        # self.VxScale = 0.15
        # self.VyScale = 0.225
        # self.vYawScale = 0.5
        # self.vZScale = 0.15
    
        self.Vx_ref = 0.0
        self.Vy_ref = 0.0
        self.Vw_ref = 0.0

        # Y, B, A and X buttons (in that order)
        self.northButton = False
        self.eastButton = False
        self.southButton = False
        self.westButton = False
        self.joystick_code = 0  # Code to carry information about pressed buttons

        self.reset = False
        self.client_init = False

    def update_v_ref(self):
        """Update the reference velocity of the robot along X, Y and Yaw in local frame by
        listening to a gamepad handled by an independent thread

        Args:
            k_loop (int): numero of the current iteration
        """

        # Create the gamepad client
        if LAASGAMEPAD:
            if not self.client_init:                
                self.gp = gC.GamepadClient()

                self.gp.leftJoystickX.value = 0.00390625
                self.gp.leftJoystickY.value = 0.00390625
                self.gp.rightJoystickX.value = 0.00390625
                self.client_init = True

            self.vX = (self.gp.leftJoystickX.value / 0.00390625 - 1 ) * self.VxScale
            self.vY = (self.gp.leftJoystickY.value / 0.00390625 - 1 ) * self.VyScale
            self.vYaw = (self.gp.rightJoystickX.value / 0.00390625 - 1 ) * self.vYawScale

        else:
            if not self.client_init:
                self.gp = gC.GamepadClient()
                self.gp.leftJoystickX.value = 0.
                self.gp.leftJoystickY.value = 0.
                self.gp.rightJoystickX.value = 0.
                self.client_init = True

            self.vX = self.gp.leftJoystickX.value * self.VxScale
            self.vY = self.gp.leftJoystickY.value * self.VyScale
            self.vYaw = self.gp.rightJoystickX.value * self.vYawScale


        self.v_gp = np.array([[- self.vY, - self.vX, 0.0, 0.0, 0.0, - self.vYaw]]).T

        if self.gp.startButton.value:
            #self.reduced = not self.reduced
            self.reset = True

        # Switch to safety controller if the Back key is pressed
        if self.gp.backButton.value:
            self.stop = True

        # Switch gaits
        if self.gp.northButton.value:
            self.northButton = True
            self.eastButton = False
            self.southButton = False
            self.westButton = False
        elif self.gp.eastButton.value:
            self.northButton = False
            self.eastButton = True
            self.southButton = False
            self.westButton = False
        elif self.gp.southButton.value:
            self.northButton = False
            self.eastButton = False
            self.southButton = True
            self.westButton = False
        elif self.gp.westButton.value:
            self.northButton = False
            self.eastButton = False
            self.southButton = False
            self.westButton = True

        # Low pass filter to slow down the changes of velocity when moving the joysticks
        self.v_gp[(self.v_gp < 0.3) & (self.v_gp > -0.3)] = 0.0
        self.v_ref = self.alpha * self.v_gp + (1-self.alpha) * self.v_ref

        # Update joystick code depending on which buttons are pressed
        self.computeCode()

        return 0

    def computeCode(self):
        # Check joystick buttons to trigger a change of gait type
        self.joystick_code = 0
        if self.southButton:
            self.joystick_code = 1
            self.southButton = False
        elif self.eastButton:
            self.joystick_code = 2
            self.eastButton = False
        elif self.westButton:
            self.joystick_code = 3
            self.westButton = False
        elif self.northButton:
            self.joystick_code = 4
            self.northButton = False

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from time import time as clock

    joystick = Joystick()
    joystick.update_v_ref()
    k = 0
    vx = [0.0] * 1000
    vy = [0.0] * 1000
    vyaw = [0.0] * 1000

    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim([-2.5, 2.5])
    h1, = plt.plot(np.linspace(0.001, 1.0, 1000), vx, "b", linewidth=2)  # forwards/backwards
    h2, = plt.plot(np.linspace(0.001, 1.0, 1000), vy, "r", linewidth=2) # left/right
    h3, = plt.plot(np.linspace(0.001, 1.0, 1000), vyaw, "g", linewidth=2) # left/right
    plt.xlabel("Time [s]")
    plt.ylabel("Forward reference velocity [m/s]")
    plt.show(block=False)
    
    print("Start")
    while True:
        # Update the reference velocity coming from the gamepad
        joystick.update_v_ref()
        vx.pop(0)
        vx.append(joystick.v_ref[0, 0])
        vy.pop(0)
        vy.append(joystick.v_ref[1, 0])
        vyaw.pop(0)
        vyaw.append(joystick.v_ref[5, 0])

        if k % 50 == 0:
            h1.set_ydata(vx)
            h2.set_ydata(vy)
            h3.set_ydata(vyaw)
            print("Joystick raw x:        ", joystick.v_gp[0, 0])
            print("Joystick filtered x:   ", joystick.v_ref[0, 0])
            print("Joystick raw y:        ", joystick.v_gp[1, 0])
            print("Joystick filtered y:   ", joystick.v_ref[1, 0])
            print("Joystick raw yaw:      ", joystick.v_gp[5, 0])
            print("Joystick filtered yaw: ", joystick.v_ref[5, 0])
            plt.pause(0.0001)

        k += 1
