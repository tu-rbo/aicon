from abc import ABCMeta
from typing import Dict, Union, Callable

import torch
import yaml

# from hidden_ctrl_interface.controller.constraint_following_controller import Constraint_Following_Controller
from aicon.base_classes.components import ActionComponent
from aicon.base_classes.connections import ActiveInterconnection
# from aicon.sensors.ros_tf_sensor import ROSTFSensor


class BasicPandaControlAction(ActionComponent, metaclass=ABCMeta):

    def __init__(self, name: str, connections: Dict[str, ActiveInterconnection], goals : Union[None,Dict[str, Callable]] = None, cf_config_file_path:str=None, dtype:torch.dtype=torch.get_default_dtype(), device:torch.device=torch.get_default_device(), mockbuild : bool = False):
        super().__init__(name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild)
        if mockbuild:
            return
        if cf_config_file_path is None:
            cf_config_file_path = "PATH TO CF CONFIG"
        config = yaml.safe_load(open(cf_config_file_path, "r"))
        # self.cf_controller = Constraint_Following_Controller(config=config, gripper=False,
        #                                                      motion_force_patterns=None,
        #                                                      experiment_dir=config["experiment_path"] + "/" + config['experiment_name'])
        self.v_max_trans = 0.06
        self.v_max_rot = 0.15
        self.comp_time = 0.2
        # self.pose_sensor = ROSTFSensor("InternalEEPoseSensor", connections={}, source_frame="panda_link0",
        #                                target_frame="panda_link8", device=device,
        #                                quantity_name="pose_measured_ee_internal", dtype=dtype, mockbuild=mockbuild)
