#
import sys
import genesis as gs

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from envs.genesis_env import GenesisSim


GS = GenesisSim()

# create franka keyboard operator
from controllers.keyboard_controller import KeyboardController
controller = KeyboardController()

# from sensors.wrist_camera import WristCamera
# wrist_camera = WristCamera()

from robots.franka import Franka
franka = Franka(name="franka",sensors=[], backends=[controller])

from robots.robot import Robot
satellite = Robot(name="satellite")

GS.scene.link_entities(satellite.robot, franka.robot, "attachment", "panda_link0")


satellite_part = Robot(name="satellite_part")


# 
GS.start()
franka.initialize()
satellite.initialize()

while True:
    franka.step()
    GS.step()
    if not GS.viewer.is_alive(): # 
        print("Viewer window has been closed.")
        break

GS.stop()