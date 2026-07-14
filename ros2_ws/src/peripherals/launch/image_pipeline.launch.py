from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import (
    ComposableNodeContainer,
    Node,
)
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    usb_camera_parameters = PathJoinSubstitution([
        FindPackageShare('peripherals'),
        'config',
        'usb_cam_param.yaml',
    ])

    camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        output='screen',
        name='usb_cam',
        parameters=[
            usb_camera_parameters
        ],
        remappings=[
            (
                'image_raw',
                '/ascamera/camera_publisher/rgb0/image',
            ),
            (
                'image_raw/compressed',
                '/ascamera/camera_publisher/'
                'rgb0/image_compressed',
            ),
            (
                'image_raw/compressedDepth',
                '/ascamera/camera_publisher/'
                'rgb0/compressedDepth',
            ),
            (
                'image_raw/theora',
                '/ascamera/camera_publisher/'
                'rgb0/image_raw/theora',
            ),
            (
                'camera_info',
                '/ascamera/camera_publisher/'
                'rgb0/camera_info',
            ),
        ],
    )

    image_proc_container = ComposableNodeContainer(
        name='image_proc_container',
        namespace='ascamera/camera_publisher/rgb0',
        package='rclcpp_components',
        executable='component_container',
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                name='rectify_color_node',
                remappings=[
                    (
                        'image',
                        '/ascamera/camera_publisher/'
                        'rgb0/image',
                    ),
                    (
                        'camera_info',
                        '/ascamera/camera_publisher/'
                        'rgb0/camera_info',
                    ),
                    (
                        'image_rect',
                        '/ascamera/camera_publisher/'
                        'rgb0/image_rect',
                    ),
                ],
            ),
        ],
    )

    return LaunchDescription([
        camera_node,
        image_proc_container,
    ])
