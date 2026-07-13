#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2023/08/28
# stm32 ros2 package

import os
import math
import time
import rclpy
import threading
import yaml  # 已导入 PyYAML
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Imu, Joy
from std_msgs.msg import UInt16, Bool
from ros_robot_controller.ros_robot_controller_sdk import Board, PacketReportKeyEvents
from ros_robot_controller_msgs.srv import GetBusServoState, GetPWMServoState
from ros_robot_controller_msgs.msg import (
    ButtonState, BuzzerState, MotorsState, BusServoState, LedState,
    SetBusServoState, ServosPosition, SetPWMServoState, Sbus, OLEDState,
    RGBStates, PWMServoState
)

class RosRobotController(Node):
    gravity = 9.80665
    MOTOR_IDS = (1, 2, 3, 4)

    def __init__(self, name):
        super().__init__(name)

        self.running = True
        self._reception_enabled = True

        self._motor_lock = threading.RLock()
        self._last_motor_command_monotonic = time.monotonic()
        self._last_motor_command_nonzero = False
        self._motor_watchdog_tripped = False

        self.board = Board()
        self.board.enable_reception(True)

        self.declare_parameter('imu_frame', 'imu_link')
        self.declare_parameter('init_finish', False)
        self.declare_parameter('motor_command_timeout_sec', 0.50)
        self.declare_parameter('motor_watchdog_period_sec', 0.05)
        self.IMU_FRAME = self.get_parameter('imu_frame').value
        self.motor_command_timeout_sec = max(
            0.10,
            float(self.get_parameter('motor_command_timeout_sec').value),
        )
        self.motor_watchdog_period_sec = max(
            0.02,
            float(self.get_parameter('motor_watchdog_period_sec').value),
        )

        self.imu_pub = self.create_publisher(Imu, '~/imu_raw', 1)
        self.joy_pub = self.create_publisher(Joy, '~/joy', 1)
        self.sbus_pub = self.create_publisher(Sbus, '~/sbus', 1)
        self.button_pub = self.create_publisher(ButtonState, '~/button', 1)
        self.battery_pub = self.create_publisher(UInt16, '~/battery', 1)
        self.create_subscription(LedState, '~/set_led', self.set_led_state, 5)
        self.create_subscription(BuzzerState, '~/set_buzzer', self.set_buzzer_state, 5)
        self.create_subscription(OLEDState, '~/set_oled', self.set_oled_state, 5)
        self.create_subscription(MotorsState, '~/set_motor', self.set_motor_state, 10)
        self.create_subscription(
            Bool,
            '~/enable_reception',
            self._enable_reception_callback,
            1,
        )
        self.create_subscription(SetBusServoState, '~/bus_servo/set_state', self.set_bus_servo_state, 10)
        self.create_subscription(ServosPosition, '~/bus_servo/set_position', self.set_bus_servo_position, 10)
        self.create_subscription(SetPWMServoState, '~/pwm_servo/set_state', self.set_pwm_servo_state, 10)
        self.create_service(GetBusServoState, '~/bus_servo/get_state', self.get_bus_servo_state)
        self.create_service(GetPWMServoState, '~/pwm_servo/get_state', self.get_pwm_servo_state)
        self.create_subscription(RGBStates, '~/set_rgb', self.set_rgb_states, 10)

        # 加载并设置舵机偏移量从 YAML 文件
        self.load_servo_offsets()

        # 初始化电机速度
        self._safe_stop_motors(
            'startup',
            repeat=3,
            delay_sec=0.02,
        )

        self.clock = self.get_clock()
        threading.Thread(target=self.pub_callback, daemon=True).start()
        threading.Thread(
            target=self._motor_watchdog_loop,
            daemon=True,
        ).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def load_servo_offsets(self):
        """
        从 YAML 文件中读取舵机偏差设置。
        """
        config_path = os.path.join(os.environ['HOME'], 'workspace/ros2_ws/src/driver/controller/config/servo_config.yaml')
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # 确保config是字典
            if not isinstance(config, dict):
                self.get_logger().error(f"YAML 配置文件格式错误: {config_path}，应为字典格式。")
                return

            # 遍历ID1到ID4并设置偏移量
            for servo_id in range(1, 5):
                offset = config.get(servo_id, 0)  # 如果未找到，默认偏移量为0
                try:
                    self.board.pwm_servo_set_offset(servo_id, offset)
                    self.get_logger().info(f"已设置舵机 {servo_id} 的偏移量为 {offset}")
                except Exception as e:
                    self.get_logger().error(f"设置舵机 {servo_id} 偏移量时出错: {e}")

        except FileNotFoundError:
            self.get_logger().error(f"配置文件未找到: {config_path}")
        except yaml.YAMLError as e:
            self.get_logger().error(f"解析 YAML 文件时出错: {e}")
        except Exception as e:
            self.get_logger().error(f"读取配置文件时出错: {e}")

    def get_node_state(self, request, response):
        response.success = True
        return response

    def pub_callback(self):
        while self.running and rclpy.ok():
            if self._reception_enabled:
                self.pub_button_data(self.button_pub)
                self.pub_joy_data(self.joy_pub)
                self.pub_imu_data(self.imu_pub)
                self.pub_sbus_data(self.sbus_pub)
                self.pub_battery_data(self.battery_pub)

            time.sleep(0.02)

    def _enable_reception_callback(self, msg):
        self._reception_enabled = bool(msg.data)
        self.get_logger().info(
            '\033[1;32m%s\033[0m'
            % ('enable_reception ' + str(self._reception_enabled))
        )
        self.board.enable_reception(self._reception_enabled)

    def set_led_state(self, msg):
        self.board.set_led(msg.on_time, msg.off_time, msg.repeat, msg.id)

    def set_buzzer_state(self, msg):
        self.board.set_buzzer(msg.freq, msg.on_time, msg.off_time, msg.repeat)

    def set_rgb_states(self, msg):
        pixels = []
        for state in msg.states:
            pixels.append((state.index, state.red, state.green, state.blue))
        self.board.set_rgb(pixels)

    @classmethod
    def _zero_motor_data(cls):
        return [
            [motor_id, 0.0]
            for motor_id in cls.MOTOR_IDS
        ]

    def _write_motor_speeds(self, data, *, context):
        try:
            with self._motor_lock:
                self.board.set_motor_speed(data)
            return True
        except Exception as exc:
            self.get_logger().error(
                f'Motor serial write failed during {context}: {exc}'
            )
            return False

    def _safe_stop_motors(
        self,
        reason,
        *,
        repeat=1,
        delay_sec=0.0,
    ):
        stopped = False

        for _ in range(max(1, int(repeat))):
            stopped = (
                self._write_motor_speeds(
                    self._zero_motor_data(),
                    context=f'safe stop: {reason}',
                )
                or stopped
            )

            if delay_sec > 0.0:
                time.sleep(float(delay_sec))

        if stopped:
            with self._motor_lock:
                self._last_motor_command_nonzero = False
                self._motor_watchdog_tripped = False

        return stopped

    def _motor_watchdog_loop(self):
        while self.running:
            time.sleep(self.motor_watchdog_period_sec)

            with self._motor_lock:
                command_nonzero = self._last_motor_command_nonzero
                command_age = (
                    time.monotonic()
                    - self._last_motor_command_monotonic
                )

            if not command_nonzero:
                continue

            if command_age <= self.motor_command_timeout_sec:
                continue

            if not self._motor_watchdog_tripped:
                self.get_logger().error(
                    'Motor command watchdog expired: '
                    f'age={command_age:.3f}s > '
                    f'{self.motor_command_timeout_sec:.3f}s; '
                    'forcing all motors to zero'
                )
                self._motor_watchdog_tripped = True

            self._safe_stop_motors(
                'motor command watchdog',
                repeat=3,
                delay_sec=0.01,
            )

    def set_motor_state(self, msg):
        # Always send all four channels. A missing channel must become zero,
        # not retain an older speed on the expansion board.
        speeds = {
            motor_id: 0.0
            for motor_id in self.MOTOR_IDS
        }

        for item in msg.data:
            motor_id = int(item.id)
            speed = float(item.rps)

            if motor_id not in speeds:
                self.get_logger().warning(
                    f'Ignoring invalid motor id={motor_id}'
                )
                continue

            if not math.isfinite(speed):
                self.get_logger().error(
                    'Ignoring non-finite motor speed: '
                    f'id={motor_id}, rps={speed}'
                )
                continue

            speeds[motor_id] = speed

        data = [
            [motor_id, speeds[motor_id]]
            for motor_id in self.MOTOR_IDS
        ]

        if not self._write_motor_speeds(
            data,
            context='motor command callback',
        ):
            return

        with self._motor_lock:
            self._last_motor_command_monotonic = time.monotonic()
            self._last_motor_command_nonzero = any(
                abs(speed) > 1e-6
                for speed in speeds.values()
            )
            self._motor_watchdog_tripped = False

    def set_oled_state(self, msg):
        self.board.set_oled_text(int(msg.index), msg.text)

    def set_pwm_servo_state(self, msg):
        data = []
        for i in msg.state:
            if i.id and i.position:
                data.extend([[i.id[0], i.position[0]]])
            if i.id and i.offset:
                self.board.pwm_servo_set_offset(i.id[0], i.offset[0])

        if data != []:
            self.board.pwm_servo_set_position(msg.duration, data)

    def get_pwm_servo_state(self, msg):
        states = []
        for i in msg.cmd:
            data = PWMServoState()
            if i.get_position:
                state = self.board.pwm_servo_read_position(i.id)
                if state is not None:
                    data.position = state
            if i.get_offset:
                state = self.board.pwm_servo_read_offset(i.id)
                if state is not None:
                    data.offset = state
            states.append(data)
        return [True, states]

    def set_bus_servo_position(self, msg):
        self.get_logger().info("In bus servo positon callback method")
        data = []
        for i in msg.position:
            data.extend([[i.id, i.position]])
        if data:
            self.board.bus_servo_set_position(msg.duration, data)

    def set_bus_servo_state(self, msg):
        data = []
        servo_id = []
        for i in msg.state:
            if i.present_id:
                if i.present_id[0]:
                    if i.target_id:
                        if i.target_id[0]:
                            self.board.bus_servo_set_id(i.present_id[1], i.target_id[1])
                    if i.position:
                        if i.position[0]:
                            data.extend([[i.present_id[1], i.position[1]]])
                    if i.offset:
                        if i.offset[0]:
                            self.board.bus_servo_set_offset(i.present_id[1], i.offset[1])
                    if i.position_limit:
                        if i.position_limit[0]:
                            self.board.bus_servo_set_angle_limit(i.present_id[1], i.position_limit[1:])
                    if i.voltage_limit:
                        if i.voltage_limit[0]:
                            self.board.bus_servo_set_vin_limit(i.present_id[1], i.voltage_limit[1:])
                    if i.max_temperature_limit:
                        if i.max_temperature_limit[0]:
                            self.board.bus_servo_set_temp_limit(i.present_id[1], i.max_temperature_limit[1])
                    if i.enable_torque:
                        if i.enable_torque[0]:
                            self.board.bus_servo_enable_torque(i.present_id[1], i.enable_torque[1])
                    if i.save_offset:
                        if i.save_offset[0]:
                            self.board.bus_servo_save_offset(i.present_id[1])
                    if i.stop:
                        if i.stop[0]:
                            servo_id.append(i.present_id[1])
        if data != []:
            self.board.bus_servo_set_position(msg.duration, data)
        if servo_id != []:
            self.board.bus_servo_stop(servo_id)

    def get_bus_servo_state(self, request, response):
        states = []
        for i in request.cmd:
            data = BusServoState()
            if i.get_id:
                state = self.board.bus_servo_read_id(i.id)
                if state is not None:
                    i.id = state[0]
                    data.present_id = state
            if i.get_position:
                state = self.board.bus_servo_read_position(i.id)
                if state is not None:
                    data.position = state
            if i.get_offset:
                state = self.board.bus_servo_read_offset(i.id)
                if state is not None:
                    data.offset = state
            if i.get_voltage:
                state = self.board.bus_servo_read_voltage(i.id)
                if state is not None:
                    data.voltage = state
            if i.get_temperature:
                state = self.board.bus_servo_read_temp(i.id)
                if state is not None:
                    data.temperature = state
            if i.get_position_limit:
                state = self.board.bus_servo_read_angle_limit(i.id)
                if state is not None:
                    data.position_limit = state
            if i.get_voltage_limit:
                state = self.board.bus_servo_read_vin_limit(i.id)
                if state is not None:
                    data.voltage_limit = state
            if i.get_max_temperature_limit:
                state = self.board.bus_servo_read_temp_limit(i.id)
                if state is not None:
                    data.max_temperature_limit = state
            if i.get_torque_state:
                state = self.board.bus_servo_read_torque(i.id)
                if state is not None:
                    data.enable_torque = state
            states.append(data)
        response.state = states
        response.success = True
        return response

    def pub_battery_data(self, pub):
        data = self.board.get_battery()
        if data is not None:
            msg = UInt16()
            msg.data = data
            pub.publish(msg)

    def pub_button_data(self, pub):
        data = self.board.get_button()
        if data is not None:
            key_id, key_event = data
            state_map = {
                PacketReportKeyEvents.KEY_EVENT_PRESSED: 1,
                PacketReportKeyEvents.KEY_EVENT_LONGPRESS: 2,
                PacketReportKeyEvents.KEY_EVENT_LONGPRESS_REPEAT: 3,
                PacketReportKeyEvents.KEY_EVENT_RELEASE_FROM_LP: 4,
                PacketReportKeyEvents.KEY_EVENT_RELEASE_FROM_SP: 0,
                PacketReportKeyEvents.KEY_EVENT_CLICK: 5,
                PacketReportKeyEvents.KEY_EVENT_DOUBLE_CLICK: 6,
                PacketReportKeyEvents.KEY_EVENT_TRIPLE_CLICK: 7,
            }
            state = state_map.get(key_event, -1)

            if state != -1:
                msg = ButtonState()
                msg.id = key_id
                msg.state = state
                pub.publish(msg)
            else:
                self.get_logger().error(f"Unhandled button event: {key_event}")

    def pub_joy_data(self, pub):
        data = self.board.get_gamepad()
        if data is not None:
            msg = Joy()
            msg.axes = data[0]
            msg.buttons = data[1]
            msg.header.stamp = self.clock.now().to_msg()
            pub.publish(msg)

    def pub_sbus_data(self, pub):
        data = self.board.get_sbus()
        if data is not None:
            msg = Sbus()
            msg.channel = data
            msg.header.stamp = self.clock.now().to_msg()
            pub.publish(msg)

    def pub_imu_data(self, pub):
        data = self.board.get_imu()
        if data is not None:
            ax, ay, az, gx, gy, gz = data
            msg = Imu()
            msg.header.frame_id = self.IMU_FRAME
            msg.header.stamp = self.clock.now().to_msg()

            msg.orientation.w = 0.0
            msg.orientation.x = 0.0
            msg.orientation.y = 0.0
            msg.orientation.z = 0.0

            msg.linear_acceleration.x = ax * self.gravity
            msg.linear_acceleration.y = ay * self.gravity
            msg.linear_acceleration.z = az * self.gravity

            msg.angular_velocity.x = math.radians(gx)
            msg.angular_velocity.y = math.radians(gy)
            msg.angular_velocity.z = math.radians(gz)

            msg.orientation_covariance = [0.01, 0.0, 0.0,
                                          0.0, 0.01, 0.0,
                                          0.0, 0.0, 0.01]
            msg.angular_velocity_covariance = [0.01, 0.0, 0.0,
                                              0.0, 0.01, 0.0,
                                              0.0, 0.0, 0.01]
            msg.linear_acceleration_covariance = [0.0004, 0.0, 0.0,
                                                 0.0, 0.0004, 0.0,
                                                 0.0, 0.0, 0.004]
            pub.publish(msg)

    def destroy_node(self):
        self.running = False

        # Direct SDK write. This does not depend on another ROS node receiving
        # a final message during shutdown.
        self._safe_stop_motors(
            'node destruction',
            repeat=5,
            delay_sec=0.03,
        )

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = RosRobotController(
            'ros_robot_controller'
        )
        rclpy.spin(node)
    except (
        KeyboardInterrupt,
        ExternalShutdownException,
    ):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception as exc:
                print(
                    'Failed to destroy '
                    f'ros_robot_controller cleanly: {exc}'
                )

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
