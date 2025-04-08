import rospy
from std_msgs.msg import Float64MultiArray

from aicon.middleware.util_ros import ros_multiarray_to_torch_tensor, torch_tensor_to_ros_multiarray

subscriber = None
publisher = None


def register_ros_action_callback(action_take_func, dtype, device):
    global subscriber

    def callback(msg):
        action_value = ros_multiarray_to_torch_tensor(msg, dtype=dtype, device=device)
        action_take_func(action_value)
        return

    subscriber = rospy.Subscriber("blocksworld_action",
                                           Float64MultiArray, callback, queue_size=10)


def get_action_send_func():
    global publisher

    publisher = rospy.Publisher("blocksworld_action", Float64MultiArray,
                                          queue_size=10)

    def send_action_func(action):
        msg = torch_tensor_to_ros_multiarray(action, Float64MultiArray())
        publisher.publish(msg)

    return send_action_func