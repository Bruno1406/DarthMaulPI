import rclpy
import cv2
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

        self.latest_rect_image = None
        self.latest_raw_image = None
        self.received_first_image = False
        self.camera_info = None

        self.tag_size_m = 0.0162
        self.submitted_count = 0

        # image_rect gives more accurate geometry for distance estimation.
        # raw image keeps more natural colors for color detection. 
        self.rect_image_subscriber = self.create_subscription(Image,"/ascamera/camera_publisher/rgb0/image_rect",self.callback_rect_image,10)
        self.raw_image_subscriber = self.create_subscription(Image,"/ascamera/camera_publisher/rgb0/image",self.callback_raw_image,10)
        self.apriltag_subscriber = self.create_subscription(AprilTagDetectionArray, "/apriltag_detections", self.callback_apriltag, 10)
        self.camera_info_subscriber = self.create_subscription(CameraInfo,"/ascamera/camera_publisher/rgb0/camera_info", self.callback_camera_info, 10)

        # TODO: Enable this client only when the grading service is running.
        # During local tests, keep it disabled to avoid blocking startup.
        # self.client = self.create_client(GradeCubes, "/grade_cubes")
        # while not self.client.wait_for_service(1.0):
        #     self.get_logger().info("waiting for grade_cubes service ...")

    # -------------------------------------------------------------------------
    # Image and camera callbacks
    # -------------------------------------------------------------------------

    def callback_rect_image(self, msg):
        # Store the rectified image used together with AprilTag pixel positions.
        self.latest_rect_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.received_first_image == False:
            self.get_logger().info("received rect image")
            self.received_first_image = True


    def callback_raw_image(self, msg):
        # Store the raw image. It is often better for color detection.
        self.latest_raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # -------------------------------------------------------------------------
    # AprilTag detection pipeline
    # -------------------------------------------------------------------------

    def callback_apriltag(self, msg):
        if not msg.detections:
            return

        if self.latest_rect_image is None:
            self.get_logger().info("no rect image yet")
            return

        if self.latest_raw_image is None:
            self.get_logger().info("no raw image yet")
            return
        
        valid_detections = []
        for d in msg.detections:
            xs = [c.x for c in d.corners]
            ys = [c.y for c in d.corners]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            area = w * h
            ratio = h / w if w > 0 else 0.0

            if w < 15 or h < 15:
                continue

            if ratio < 0.6 or ratio > 1.5:
                continue

            valid_detections.append((d, area))

        if not valid_detections:
            return

        detection,area = max(valid_detections, key=lambda item: item[1])

        relative_position = self.estimate_cube_relative_position(detection)
        if relative_position is None:
            return
 
        xr_cm, yr_cm = relative_position

        
        #TODO: Convert robot-relative cube position to global map coordinates.
        rect_crop = self.crop_cube_from_detection(self.latest_rect_image, detection)
        raw_crop = self.crop_cube_from_detection(self.latest_raw_image, detection)

        if rect_crop is None or raw_crop is None:
            self.get_logger().info("error: crop failed")
            return

        cv2.imwrite("/tmp/search_rescue_rect_crop.jpg", rect_crop)
        cv2.imwrite("/tmp/search_rescue_raw_crop.jpg", raw_crop)

        # TODO: xr_cm and yr_cm are currently robot-relative coordinates.
        # The competition requires absolute maze coordinates, so convert them
        # using the robot's global pose before calling process_cube_detection().
        self.process_cube_detection(xr_cm, yr_cm, raw_crop) 

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

        scale = 2.3
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
    
    def process_cube_detection(self,x_cm,y_cm,image):
        if image is None:
            self.get_logger().info("error: no image")
            return
        color_id = detect_color(image)

        if color_id is None:
            return
        
        is_new_cube = self.tracker.add_cube(x_cm, y_cm, color_id)
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

    # -------------------------------------------------------------------------
    # Position estimation
    # -------------------------------------------------------------------------
    def estimate_cube_relative_position(self, detection):
        """
        Estimate cube position relative to the robot/camera.

        Returns:
            (forward_cm, left_cm), where forward is the camera-facing distance
            and left is the lateral offset. This is not yet the absolute maze
            coordinate required by the task.
        """
        if self.camera_info is None:
            self.get_logger().info("no camera info yet")
            return None
        
        fx = self.camera_info.p[0]
        cx = self.camera_info.p[2]

        corners = detection.corners
        side_lengths = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            side_lengths.append(sqrt(dx * dx + dy * dy))
        tag_width_px = max(side_lengths)

        if tag_width_px <= 0:
            return None

        distance_m = self.tag_size_m * fx / tag_width_px

        image_offset_x = detection.centre.x - cx
        lateral_m = image_offset_x * distance_m / fx

        forward_cm = distance_m * 100
        left_cm = -lateral_m * 100

        return forward_cm, left_cm
    

    
    # def maybe_submit_cubes(self, is_new_cube):
    #     if not is_new_cube:
    #        return

    #     n, _, _, _ = self.tracker.export_for_service()

    #     if n > self.submitted_count and n <= 4:
    #         self.submit_cubes()
    #         self.submitted_count = n
    
    # def submit_cubes(self):
    #     n,xs,ys,colors = self.tracker.export_for_service()

    #     request = GradeCubes.Request()
    #     request.n = n
    #     request.x = xs
    #     request.y = ys
    #     request.color = colors

    #     future = self.client.call_async(request)
    #     future.add_done_callback(self.handle_grade_response)
    #     self.get_logger().info("submitted cubes")

    # def handle_grade_response(self, future):
    #     response = future.result()
    #     self.get_logger().info(f"score: {response.score}")      


def main(args=None):
    rclpy.init(args=args)
    node = SearchRescueNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()