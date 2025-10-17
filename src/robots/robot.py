
import sys
import torch
import genesis as gs

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from envs.genesis_env import GenesisSim
from configs.asset_configs import ASSETS
from utils.singlelink_state import SingleLinkState


class Robot:
    def __init__(
        self,
        name="franka",
        sensors=[],
        backends=[],
    ):
        # Get the current scene 
        self._robot_name = name
        self._scene = GenesisSim().scene
        self.device = GenesisSim().device

        # entity, config = parse_asset_config(ASSETS, name)
        self.robot = self._scene.add_entity(**ASSETS[name])

        # Initialize the base's position and orientation
        self._base_state = SingleLinkState(device=self.device)
        # baselink_state = self._base_state.global_state

        # --------------------------------------------------------------------
        # -------------------- Add sensors to the robot --------------------
        # --------------------------------------------------------------------
        self._sensors = sensors

        for sensor in self._sensors:
            sensor.initialize(self)

        # --------------------------------------------------------------------
        # -------------------- Add control backends to the robot -----------
        # --------------------------------------------------------------------
        self._backends = backends

        # Initialize the backends
        for backend in self._backends:
            backend.initialize(self)


    """
    Properties
    """
    
    @property
    def base_state(self):
        """The state of the robot.

        Returns:
            State: The current state of the robot, i.e., position, orientation, linear and angular velocities...
        """
        return self._base_state
    
    @property
    def name(self) -> str:
        """Robot name.

        Returns:
            Robot name (str): 
        """
        return self._robot_name
    
    """
    Operations
    """
    def initialize(self):
        # Initialize robot's configuration
        # self.set_config(self.config)
        return

    def update_base_state(self):
        # Initialize robot's base coordinate
        self._base_state.update_from_global_frame(
            position=self.robot.get_pos(),
            quat=self.robot.get_quat(),  # quaternion: [qw, qx, qy, qz]
            # linear_velocity=torch.tensor([0.1, 0.2, 0.3]),
            # angular_velocity=torch.tensor([0.01, 0.02, 0.03])
        )

    def step(self):
        return
    
    def stop(self):
        for sensor in self._sensors:
            sensor.stop()
        for backend in self._backends:
            backend.stop()
        return

    def reset(self):
        return

    def show_info(self):
        print([link.name for link in self.robot.links])

    def apply_force(self, force, pos=[0.0, 0.0, 0.0]):
        """
        Method that will apply a force on the rigidbody, on the part specified in the 'body_part' at its relative position
        given by 'pos' (following a FLU) convention. 

        Args:
            force (list): A 3-dimensional vector of floats with the force [Fx, Fy, Fz] on the body axis of the vehicle according to a FLU convention.
            pos (list): _description_. Defaults to [0.0, 0.0, 0.0].
            body_part (str): . Defaults to "/body".
        """

        # Get the handle of the rigidbody that we will apply the force to
        return