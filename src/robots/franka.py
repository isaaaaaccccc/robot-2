
import sys
import torch
import copy
import numpy as np
import genesis as gs

# Extension APIs
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))
from robots.robot import Robot
from envs.genesis_env import GenesisSim
from utils.multilink_state import MultiLinkState
from utils.utils import convert_dict_to_tensors, angle_difference, angular_velocity
from configs.asset_configs import FRANKA_CONFIG
from controllers.smooth_IK_solver import SmoothIKSolver

class Franka(Robot):
    def __init__(
        self,
        name="franka",
        sensors=[],
        backends=[]
    ):
        # Initialize the Robot object, and add robot to backends
        super().__init__(
            name="franka", 
            sensors=sensors,
            backends=backends
            )
        
        # Get the current world at which we want to spawn the Robot
        self.franka_name = name
        self._scene = GenesisSim().scene
        
        self.end_effector = self.robot.get_link("panda_hand")
        self.config = convert_dict_to_tensors(FRANKA_CONFIG, torch.float32, self.device)

        baselink_state = self._base_state.global_state
        self.ee_state = MultiLinkState(baselink_state, device=self.device) # body state是相对于机械臂baselink

        self.joints_name = (
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint1",
            "panda_finger_joint2",
        )

        motors_dof_idx = [self.robot.get_joint(name).dofs_idx_local[0] for name in self.joints_name]
        self.motors_dof = motors_dof_idx[:7]
        self.fingers_dof = motors_dof_idx[7:9]
        self.finger_open = torch.tensor([0.04, 0.04],dtype=torch.float32,device=self.device)
        self.finger_close = torch.tensor([0.0,0.0],dtype=torch.float32,device=self.device)
        
        
    """
    Properties
    """
    
    @property
    def state(self):
        """The state of the robot.

        Returns:
            State: The current state of the robot, i.e., position, orientation, linear and angular velocities...
        """
        return self.ee_state
    
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
        """After the scene is built."""
        # Initialize Franka robot's configuration
        self.set_config(self.config)

        # Initialize base and end effector's position and attitude
        self.update_state()
        # print(self.ee_state)

        self.IK = SmoothIKSolver(
            robot=self.robot, 
            end_effector=self.end_effector,
            smooth_factor=0.3, 
            max_joint_change=0.05,
            pos_tolerance=1e-4,
            quat_tolerance=1e-4
        )

        # Initialize target's position and attitude
        self.target_pos, self.target_quat = None, None
        self.target_handopen = True
        self.duration_time = 1  # velocity duration time /s
        
        keyboardControl = self._backends[0]
        keyboardControl.update_state(self.ee_state.body_state)
    
    def set_config(self, config):
        if config is None:
            return
        # Set controller gains
        if 'control' in config:
            self.robot.set_dofs_kp(config['control']['kp'])
            self.robot.set_dofs_kv(config['control']['kv'])
            self.robot.set_dofs_force_range(
                config['control']['force_range_min'],
                config['control']['force_range_max']
            )
            # print("set controller gains!")
        # Set initial dofs positions
        if 'initial_dofs' in config:
            self.robot.set_dofs_position(config['initial_dofs'], self.motors_dof+self.fingers_dof)

    def update_state(self):
        # Update base's position and attitude
        self.update_base_state()

        # Transmit the updated base to ee
        baselink_state = self._base_state.global_state
        self.ee_state.update_baselink(baselink_state)

        # Update robot's end effector state
        self.ee_state.update_from_global_frame(
            position=self.end_effector.get_pos(),
            quat=self.end_effector.get_quat() # [qw,qx,qy,qz]
        )


    def step(self):
        # Update base state and then ee state of franka
        self.update_state()
        # Update latest state to manual controller 
        # self.ee_state.position_body 
        keyboardControl = self._backends[0]
        # keyboardControl.update_state(self.ee_state.body_state)
        print("ee body state: ", self.ee_state.body_state)
        # print("ee global state: ", self.ee_state.global_state)
        # Controller changes the state and send back to ee state to set target poses
        if keyboardControl.running:
            self.target_pos, self.target_quat, self.target_handopen = keyboardControl.step(None)
        # print()
        print("Keyboard command: ", self.target_pos, self.target_quat)
        finger_state = self.finger_open if self.target_handopen else self.finger_close

        # Compute joints' angles under body frame
        # qpos = self.robot.inverse_kinematics(
        #     link=self.end_effector,
        #     pos=self.target_pos, 
        #     quat=self.target_quat,
        # )
        qpos = self.IK.solve(self.target_pos,self.target_quat)
        print("Target command: ", qpos)
        # if torch.isnan(qpos).any():
        #     return

        # Control joints' dofs
        self.robot.set_qpos(qpos[:-2], self.motors_dof)

        qpos_fb = self.robot.get_qpos()
        print("Current command: ", qpos_fb)
        # q_velocity = angular_velocity(qpos[:-2], qpos_fb[:-2], self.duration_time)
        # self.robot.control_dofs_velocity(q_velocity, self.motors_dof)
        
        self.robot.control_dofs_position(finger_state, self.fingers_dof)

        # Call the update methods in sensors
        for sensor in self._sensors:
            sensor.step()
        

    def show_info(self):
        print([link.name for link in self.robot.links])
