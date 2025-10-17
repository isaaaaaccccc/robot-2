import numpy as np
import torch
from scipy.spatial.transform import Rotation
from typing import Optional, Union


class StateVectors:
    """
    Stores and manages the state vectors of a given robot, including position, orientation and velocity.
    Automatically synchronizes all orientation variables.
    Supports device management for GPU computation.
    """

    def __init__(self, euler_order: str = "XYZ", datatype = torch.float32, device: str = "cpu"):
        """
        Initialize the State object.
        
        Args:
            euler_order: Euler angle rotation order (default: "XYZ" for roll-pitch-yaw)
        """
        self.euler_order = euler_order
        self.device = torch.device(device)
        self.datatype = datatype

        # Initialize all state variables
        self._initialize_states()

    def _initialize_states(self):
        """Initialize all state variables with zeros"""
        # Global frame states
        self._position = torch.tensor([0.0, 0.0, 0.0], dtype=self.datatype, device=self.device)
        self._quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=self.datatype, device=self.device)  # [qx, qy, qz, qw]
        self._orient = torch.tensor([0.0, 0.0, 0.0], dtype=self.datatype, device=self.device)
        self._R_mat = torch.eye(3, dtype=self.datatype, device=self.device)

        self._linear_velocity = torch.tensor([0.0, 0.0, 0.0], dtype=self.datatype, device=self.device)
        self._angular_velocity = torch.tensor([0.0, 0.0, 0.0], dtype=self.datatype, device=self.device)

    @property
    def position(self):
        return self._position.clone()
    
    @property
    def quat(self):
        return self._quat.clone()
    
    @property
    def orient(self):
        return self._orient.clone()
    
    @property
    def linear_velocity(self):
        return self._linear_velocity.clone()
    
    @property
    def angular_velocity(self):
        return self._angular_velocity.clone()

    def update(self, 
               position: Optional[torch.Tensor] = None,
               orient: Optional[torch.Tensor] = None,
               quat: Optional[torch.Tensor] = None,
               linear_velocity: Optional[torch.Tensor] = None,
               angular_velocity: Optional[torch.Tensor] = None):
        """
        Update state variables with automatic synchronization of orientation representations.
                
        Args:
            position: Position vector [x, y, z] in global frame
            orient: Euler angles [roll, pitch, yaw] in global frame
            quat: Quaternion [qx, qy, qz, qw] in global frame
            linear_velocity: Linear velocity vector [vx, vy, vz] in global frame
            angular_velocity: Angular velocity vector [ωx, ωy, ωz] in global frame
            
        Raises:
            ValueError: If both orient and quat are provided simultaneously
        """
        # Update position if provided
        if position is not None:
            self._position = position.to(dtype=self.datatype, device=self.device)
        
        # Update orientation representations with priority handling
        if orient is not None and quat is not None:
            raise ValueError("Cannot provide both orient and quat simultaneously. "
                           "Please provide only one orientation representation.")
        
        if orient is not None:
            # Update from Euler angles
            self._update_rotation_matrices_from_euler(orient)
        elif quat is not None:
            # Update from quaternion
            self._update_rotation_matrices_from_quat(quat)
        
        # Update velocities if provided
        if linear_velocity is not None:
            self._linear_velocity = linear_velocity.to(dtype=self.datatype, device=self.device)
        
        if angular_velocity is not None:
            self._angular_velocity = angular_velocity.to(dtype=self.datatype, device=self.device)

    def _update_rotation_matrices_from_quat(self, quat: torch.Tensor):
        """Update rotation matrices from global quaternion"""
        self._quat = quat.to(dtype=self.datatype, device=self.device)

        # Update orientation angles
        R = Rotation.from_quat(quat.cpu().numpy())  # [qx, qy, qz, qw]
        self._orient = torch.tensor(R.as_euler(self.euler_order.lower(), degrees=False), dtype=self.datatype)
        self._R_mat = torch.tensor(R.as_matrix(), dtype=self.datatype, device=self.device) # transfer to this frame

    def _update_rotation_matrices_from_euler(self, euler: torch.Tensor):
        """Update rotation matrices from global Euler angles"""
        self._orient = euler.to(dtype=self.datatype, device=self.device)

        # Update quaternions
        R = Rotation.from_euler(self.euler_order.lower(), euler.cpu().numpy(), degrees=False)
        self._quat = torch.tensor(R.as_quat(), dtype=self.datatype)
        self._R_mat = torch.tensor(R.as_matrix(), dtype=self.datatype, device=self.device) # transfer to this frame

    def to(self, device: Union[str, torch.device]):
        """Move all state tensors to the specified device"""
        device = torch.device(device)
        if device == self.device:
            return self
        
        self.device = device
        
        # Move all tensors to the new device
        for attr_name in dir(self):
            if attr_name.startswith('_') and not attr_name.startswith('__'):
                attr = getattr(self, attr_name)
                if isinstance(attr, torch.Tensor):
                    setattr(self, attr_name, attr.to(device))
        
        return self

    def __repr__(self):
        return (f"State(euler_order='{self.euler_order}')\n"
                f"Position: {self.position}\n"
                f"Orientation: {self.orient}\n"
                f"Linear_Velocity: {self.linear_velocity}\n"
                f"Angular_Velocity: {self.angular_velocity}\n")