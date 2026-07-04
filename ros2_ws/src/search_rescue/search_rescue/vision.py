import cv2
import numpy as np

def detect_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #红色比较特殊，在下面单独处理
    color_ranges = {
        1: ((100, 80, 50), (130, 255, 255)),  # blue
        2: ((20, 80, 80), (35, 255, 255)),    # yellow
        3: ((40, 60, 50), (85, 255, 255)),    # green
        4: ((5, 100, 80), (20, 255, 255)),    # orange
        6: ((130, 50, 50), (160, 255, 255)),  # purple
        7: ((160, 50, 80), (175, 255, 255)),  # pink
        8: ((5, 40, 20), (25, 180, 160)),     # brown
    }

    best_color =None
    best_pixels = 0

    for color_id, (lower, upper) in color_ranges.items():
        lower=np.array(lower)
        upper=np.array(upper)

        mask = cv2.inRange(hsv,lower,upper)
        pixels = cv2.countNonZero(mask)

        if pixels > best_pixels:
            best_pixels = pixels
            best_color = color_id

    lower_red_1 = np.array((0, 100, 80))
    upper_red_1 = np.array((5, 255, 255))
    lower_red_2 = np.array((175, 100, 80))
    upper_red_2 = np.array((180, 255, 255))

    red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_pixels = cv2.countNonZero(red_mask_1) + cv2.countNonZero(red_mask_2)

    if red_pixels > best_pixels:
        best_color = 5
        best_pixels = red_pixels

    if best_pixels == 0:
        return None

    return best_color


# if __name__ == "__main__":
#     image = cv2.imread("test_cube.jpg")
#     color_id = detect_color(image)
#     print(color_id)