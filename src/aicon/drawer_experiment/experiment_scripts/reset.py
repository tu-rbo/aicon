import time

import rospy
import yaml

from hidden_ctrl_interface.controller.constraint_following_controller import Constraint_Following_Controller


def perform_reset():
    dq = [0.73786533, -1.37039304, 1.20071924, -1.83920097, -0.20790383, 1.43937683,
         1.55907142]
    move_to_dq(dq)


def move_to_dq(dq):
    cf_config_file_path = "/home/vito/ws_aicon/src/lockbox_solving_system/config/constraint_following_config.yaml"
    config = yaml.safe_load(open(cf_config_file_path, "r"))
    cf_controller = Constraint_Following_Controller(config=config, gripper=False,
                                                    motion_force_patterns=None,
                                                    experiment_dir=config["experiment_path"] + "/" + config[
                                                        'experiment_name'])
    time.sleep(2)
    print("Initial Joint State")
    print(cf_controller.q_queue[-1])
    # go to initial position
    cf_controller.move_to_joint_position(
        dq, completion_time=7, print=False)
    time.sleep(1)
    print("Final Joint State")
    print(cf_controller.q_queue[-1])


if __name__ == "__main__":
    rospy.init_node("RESET", anonymous=False)
    perform_reset()
    rospy.signal_shutdown("Main Process Received Shutdown!")