'''
Simple python script to get Asyncronous gamepad inputs
Thomas FLAYOLS - LAAS CNRS
From https://github.com/thomasfla/solopython

Use:
To display data, run "python gamepadClient.py"
'''
import inputs
import time
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
from ctypes import c_double, c_bool
from pynput import keyboard


class GamepadClient():
    def __init__(self):
        self.running = Value(c_bool, lock=True)
        self.startButton = Value(c_bool, lock=True)
        self.backButton = Value(c_bool, lock=True)
        self.northButton = Value(c_bool, lock=True)
        self.eastButton = Value(c_bool, lock=True)
        self.southButton = Value(c_bool, lock=True)
        self.westButton = Value(c_bool, lock=True)
        self.leftJoystickX = Value(c_double, lock=True)
        self.leftJoystickY = Value(c_double, lock=True)
        self.rightJoystickX = Value(c_double, lock=True)
        self.rightJoystickY = Value(c_double, lock=True)
        self.R1Button = Value(c_bool, lock=True)
        self.L1Button = Value(c_bool, lock=True)

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()

        self.startButton.value = False
        self.backButton.value = False
        self.northButton.value = False
        self.eastButton.value = False
        self.southButton.value = False
        self.westButton.value = False
        self.leftJoystickX.value = 0.0
        self.leftJoystickY.value = 0.0
        self.rightJoystickX.value = 0.0
        self.rightJoystickY.value = 0.0
        self.R1Button.value = False
        self.L1Button.value = False

        args = (self.running, self.startButton, self.backButton,
                self.northButton, self.eastButton, self.southButton, self.westButton, self.leftJoystickX,
                self.leftJoystickY, self.rightJoystickX, self.rightJoystickY, self.R1Button, self.L1Button)
        self.process = Process(target=self.run, args=args)
        self.process.start()
        time.sleep(0.2)

    def run(self, running, startButton, backButton, northButton, eastButton, southButton, westButton, leftJoystickX, leftJoystickY, rightJoystickX, rightJoystickY, R1Button, L1Button):
        running.value = True
        while(running.value):
            try:
                events = inputs.get_gamepad()
                for event in events:
                    # print(event.ev_type, event.code, event.state)
                    if event.ev_type == 'Absolute':
                        if event.code == 'ABS_X':
                            leftJoystickX.value = event.state / 32768.0
                        if event.code == 'ABS_Y':
                            leftJoystickY.value = event.state / 32768.0
                        if event.code == 'ABS_RX':
                            rightJoystickX.value = event.state / 32768.0
                        if event.code == 'ABS_RY':
                            rightJoystickY.value = event.state / 32768.0
                    if (event.ev_type == 'Key'):
                        if event.code == 'BTN_START':
                            startButton.value = event.state
                        elif event.code == 'BTN_TR':
                            R1Button.value = event.state
                        elif event.code == 'BTN_TL':
                            L1Button.value = event.state
                        elif event.code == 'BTN_SELECT':
                            backButton.value = event.state
                        elif event.code == 'BTN_NORTH':
                            northButton.value = event.state
                        elif event.code == 'BTN_EAST':
                            eastButton.value = event.state
                        elif event.code == 'BTN_SOUTH':
                            southButton.value = event.state
                        elif event.code == 'BTN_WEST':
                            westButton.value = event.state
            except inputs.UnpluggedError:
                pass  # No gamepad found, ignore

    def on_press(self, key):
        try:
            if key == keyboard.Key.f2:
                self.rightJoystickX.value = -1.0  # Yaw left
            elif key == keyboard.Key.f3:
                self.rightJoystickX.value = 1.0  # Yaw right
            elif key == keyboard.Key.left:
                self.leftJoystickX.value = -1.0
            elif key == keyboard.Key.right:
                self.leftJoystickX.value = 1.0
            elif key == keyboard.Key.up:
                self.leftJoystickY.value = -1.0
            elif key == keyboard.Key.down:
                self.leftJoystickY.value = 1.0

            elif key == keyboard.Key.f4:
                self.southButton.value = 1.0 # Flat
            elif key == keyboard.Key.f5:
                self.eastButton.value = 1.0 # Ascent
            elif key == keyboard.Key.f6:
                self.westButton.value = 1.0 # Descent
            elif key == keyboard.Key.f7:
                self.northButton.value = 1.0 # Trench
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key in [keyboard.Key.f2, keyboard.Key.f3]:
                self.rightJoystickX.value = 0.0  # Stop yaw movement
            elif key in [keyboard.Key.left, keyboard.Key.right]:
                self.leftJoystickX.value = 0.0
                self.rightJoystickX.value = 0.0
            elif key in [keyboard.Key.up, keyboard.Key.down]:
                self.leftJoystickY.value = 0.0
            elif key in [keyboard.Key.f4, keyboard.Key.f5, keyboard.Key.f6, keyboard.Key.f7]:
                pass
                # self.southButton.value = 0.0
                # self.eastButton.value = 0.0
                # self.westButton.value = 0.0
                # self.northButton.value = 0.0
        except AttributeError:
            pass

    def stop(self):
        self.running.value = False
        self.keyboard_listener.stop()
        self.process.terminate()
        self.process.join()


if __name__ == "__main__":
    gp = GamepadClient()
    try:
        for i in range(1000):
            print("LX = ", gp.leftJoystickX.value, end=" ; ")
            print("LY = ", gp.leftJoystickY.value, end=" ; ")
            print("RX = ", gp.rightJoystickX.value, end=" ; ")
            print("RY = ", gp.rightJoystickY.value, end=" ; ")
            print("start = ",gp.startButton.value)
            print("back = ",gp.backButton.value)
            print("R1 = ",gp.R1Button.value)
            print("L1 = ",gp.L1Button.value)
            time.sleep(0.1)
    except KeyboardInterrupt:
        gp.stop()
