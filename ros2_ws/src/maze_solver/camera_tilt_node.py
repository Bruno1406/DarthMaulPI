#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ros_robot_controller_msgs.msg import PWMServoState, SetPWMServoState


class CameraTiltNode(Node):
    def __init__(self):
        super().__init__('camera_tilt_node')
        self.declare_parameter('servo_id', 1)
        self.declare_parameter('position', 2000)
        self.declare_parameter('duration', 0.5)
        self.declare_parameter('publish_period_s', 0.25)
        self.declare_parameter('publish_for_s', 3.0)

        self.servo_id = int(self.get_parameter('servo_id').value)
        self.position = int(self.get_parameter('position').value)
        self.duration = float(self.get_parameter('duration').value)
        publish_period_s = float(self.get_parameter('publish_period_s').value)
        publish_for_s = float(self.get_parameter('publish_for_s').value)

        self.publisher = self.create_publisher(
            SetPWMServoState,
            'ros_robot_controller/pwm_servo/set_state',
            1,
        )
        self.stop_at = self.get_clock().now().nanoseconds + int(publish_for_s * 1e9)
        self.timer = self.create_timer(publish_period_s, self.publish_tilt_command)
        self.publish_tilt_command()

    def publish_tilt_command(self):
        servo_state = PWMServoState()
        servo_state.id = [self.servo_id]
        servo_state.position = [self.position]

        msg = SetPWMServoState()
        msg.state = [servo_state]
        msg.duration = self.duration
        self.publisher.publish(msg)

        if self.get_clock().now().nanoseconds >= self.stop_at:
            self.get_logger().info(
                f'Set camera servo {self.servo_id} to position {self.position}'
            )
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CameraTiltNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
