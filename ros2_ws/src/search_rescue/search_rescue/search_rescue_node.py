import rclpy
from rclpy.node import Node
from math import sqrt
from search_rescue.vision import detect_color
from search_rescue.cube_tracker import CubeTracker
 
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from apriltag_msgs.msg import AprilTagDetectionArray

from maze_interface.srv import GradeCubes


class SearchRescueNode(Node):
    def __init__(self):
        super().__init__("search_rescue_node")
        self.tracker = CubeTracker()
        self.bridge = CvBridge()
        self.latest_image = None
        self.received_first_image = False
        self.camera_info = None
        self.tag_size_m = 0.0162
        self.submitted_count = 0

        self.image_subscriber = self.create_subscription(Image, "/ascamera/camera_publisher/rgb0/image", self.callback_image, 10)
        self.apriltag_subscriber = self.create_subscription(AprilTagDetectionArray, "/apriltag_detections", self.callback_apriltag, 10)
        self.camera_info_subscriber = self.create_subscription(CameraInfo,"/ascamera/camera_publisher/rgb0/camera_info", self.callback_camera_info, 10)

        # ToDo: self.create_client...

# self.image_subscriber
    def callback_image(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
        self.latest_image=cv_image
        if self.received_first_image == False:
            self.get_logger().info("received image")
            self.received_first_image = True

# self.apriltag_subscriber
    def callback_apriltag(self, msg):
        if not msg.detections:
            self.get_logger().info("no tag yet")
            return

        if self.latest_image is None:
            self.get_logger().info("no image yet")
            return
        
        detection = max(msg.detections, key=lambda d: d.centre.y)

        tag_id = detection.id
        centre_x = detection.centre.x
        centre_y = detection.centre.y
            
        relative_position = self.estimate_cube_relative_position(detection)

        if relative_position is None:
            return

        x_cm, y_cm = relative_position
        # ToDo：Determine the absolute position of the cube based on the robot’s position and the cube’s position relative to the robot
        crop = self.crop_cube_from_detection(self.latest_image, detection)

        if crop is None:
            self.get_logger().info("error: crop failed")
            return

        self.process_cube_detection(tag_id, x_cm, y_cm, crop)

    def crop_cube_from_detection(self, image, detection):
        if image is None:
            return None

        height, width = image.shape[:2]

        cx = int(detection.centre.x)
        cy = int(detection.centre.y)

        corners = detection.corners

        xs = [corner.x for corner in corners]
        ys = [corner.y for corner in corners]

        tag_width = max(xs) - min(xs)
        tag_height = max(ys) - min(ys)

        tag_size = max(tag_width, tag_height)

        if tag_size <= 0:
            return None

        scale = 1.8
        crop_size = int(tag_size * scale)
        half = crop_size // 2

        x1 = max(0, cx - half)
        x2 = min(width, cx + half)
        y1 = max(0, cy - half)
        y2 = min(height, cy + half)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        return crop
    
    def process_cube_detection(self,tag_id,x_cm,y_cm,image):
        if image is None:
            self.get_logger().info("error: no image")
            return
        color_id = detect_color(image)

        if color_id is None:
            self.get_logger().info("error: no color")
            return
        
        is_new_cube = self.tracker.add_cube(tag_id, x_cm, y_cm, color_id)
        if is_new_cube:
            n,xs,ys,colors = self.tracker.export_for_service()
            self.get_logger().info(
                f"new cube: n={n}, x={round(x_cm)}, y={round(y_cm)}, color={color_id}"
            )

        

# self.camera_info_subscriber
    def callback_camera_info(self, msg):
        if self.camera_info is not None:
            return

        self.camera_info = msg
        self.get_logger().info(f"camera info received: k={msg.k}")

    def estimate_cube_relative_position(self, detection):
        if self.camera_info is None:
            self.get_logger().info("no camera info yet")
            return None
        
        fx = self.camera_info.k[0]
        cx = self.camera_info.k[2]

        corners = detection.corners

        side_lengths = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            side_lengths.append(sqrt(dx * dx + dy * dy))

        tag_width_px = sum(side_lengths) / len(side_lengths)

        if tag_width_px <= 0:
            return None

        distance_m = self.tag_size_m * fx / tag_width_px

        image_offset_x = detection.centre.x - cx
        lateral_m = image_offset_x * distance_m / fx

        forward_cm = distance_m * 100
        left_cm = -lateral_m * 100

        return forward_cm, left_cm
    

    
    def maybe_submit_cubes(self, is_new_cube):
        if not is_new_cube:
           return

        n, _, _, _ = self.tracker.export_for_service()

        if n > self.submitted_count and n <= 4:
            self.submit_cubes()
            self.submitted_count = n
    
    def submit_cubes(self):
        n,xs,ys,colors = self.tracker.export_for_service()

        request = GradeCubes.Request()
        request.n = n
        request.x = xs
        request.y = ys
        request.color = colors

        future = self.client.call_async(request)
        future.add_done_callback(self.handle_grade_response)
        self.get_logger().info("submitted cubes")

    def handle_grade_response(self, future):
        response = future.result()
        self.get_logger().info(f"score: {response.score}")      


def main(args=None):
    rclpy.init(args=args)
    node = SearchRescueNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()