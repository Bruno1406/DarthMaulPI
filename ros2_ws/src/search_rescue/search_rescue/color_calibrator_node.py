import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from apriltag_msgs.msg import AprilTagDetectionArray


class ColorCalibratorNode(Node):
    def __init__(self):
        super().__init__("color_calibrator_node")

        self.bridge = CvBridge()
        self.latest_image = None
        self.hsv_samples = []
        self.frame_count = 0
        self.save_crop_count = 0

        self.image_subscriber = self.create_subscription(
            Image,
            "/ascamera/camera_publisher/rgb0/image",
            self.callback_image,
            10
        )

        self.apriltag_subscriber = self.create_subscription(
            AprilTagDetectionArray,
            "/apriltag_detections",
            self.callback_apriltag,
            10
        )

        self.get_logger().info("color calibrator started")

    def callback_image(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

    def callback_apriltag(self, msg):
        if self.latest_image is None:
            return

        if not msg.detections:
            return

        detection = max(msg.detections, key=lambda d: d.centre.y)

        crop = self.crop_cube_from_detection(self.latest_image, detection)

        if crop is None:
            return

        if self.save_crop_count < 20:
            filename = f"/tmp/crop_{self.save_crop_count}.jpg"
            cv2.imwrite(filename, crop)
            self.get_logger().info(f"saved crop: {filename}")
            self.save_crop_count += 1

        pixels = self.extract_valid_hsv_pixels(crop)

        if len(pixels) == 0:
            return

        self.hsv_samples.append(pixels)
        self.frame_count += 1

        if self.frame_count % 20 == 0:
            total_pixels = sum(len(sample) for sample in self.hsv_samples)
            self.get_logger().info(
                f"collected frames: {self.frame_count}, pixels: {total_pixels}"
            )

    def crop_cube_from_detection(self, image, detection):
        # (height, width, channels)
        height, width = image.shape[:2]  

        cx = int(detection.centre.x)
        cy = int(detection.centre.y)

        xs = [corner.x for corner in detection.corners]
        ys = [corner.y for corner in detection.corners]

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

        return image[y1:y2, x1:x2]

    def extract_valid_hsv_pixels(self, bgr_image):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        valid_mask = cv2.inRange(
            hsv,
            (0, 50, 40),
            (179, 255, 255)
        )

        return hsv[valid_mask > 0]

    def print_result(self):
        if not self.hsv_samples:
            print("no hsv samples collected")
            return

        pixels = np.vstack(self.hsv_samples)

        h = pixels[:, 0]
        s = pixels[:, 1]
        v = pixels[:, 2]

        h_low, h_high = np.percentile(h, [20, 80])
        s_low, s_high = np.percentile(s, [10, 90])
        v_low, v_high = np.percentile(v, [10, 90])

        print("")
        print("recommended HSV range:")
        print(f"lower = ({int(h_low)}, {int(s_low)}, {int(v_low)})")
        print(f"upper = ({int(h_high)}, {int(s_high)}, {int(v_high)})")
        print("")
        print("median:")
        print(f"H={np.median(h):.1f}, S={np.median(s):.1f}, V={np.median(v):.1f}")
        print("")
        print("mean/std:")
        print(f"H={np.mean(h):.1f} +/- {np.std(h):.1f}")
        print(f"S={np.mean(s):.1f} +/- {np.std(s):.1f}")
        print(f"V={np.mean(v):.1f} +/- {np.std(v):.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = ColorCalibratorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.print_result()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()