# search_rescue

Task 3 package for detecting rescue cubes, estimating their positions, identifying colors, and preparing cube data for the grading service.

## File overview

- `package.xml`: ROS2 package metadata and runtime dependencies.
- `setup.py`: Python package installation config and console script entry points.
- `setup.cfg`: ROS2 Python package script install paths.
- `resource/search_rescue`: Ament resource marker for package discovery.
- `launch/search_rescue.launch.xml`: Starts the camera pipeline, AprilTag detection, and the main search-rescue node.
- `launch/color_calibrator.launch.xml`: Starts the camera pipeline, AprilTag detection, and the HSV color calibration node.
- `search_rescue/__init__.py`: Python package marker.
- `search_rescue/search_rescue_node.py`: Main Task 3 node for image/tag handling, cube position estimation, color detection, tracking, and grading-service submission logic.
- `search_rescue/color_calibrator_node.py`: Helper node for collecting cropped cube images and HSV statistics during color calibration.
- `search_rescue/cube_tracker.py`: Tracks detected cubes by position and exports cube data in the grading-service format.
- `search_rescue/vision.py`: HSV-based cube color detection utilities.
- `search_rescue/exploration.py`: Placeholder for future Task 3 exploration integration.
- `search_rescue/test_cube.jpg`: Sample cube image used while developing color detection.
- `search_rescue/test_cube2.jpg`: Additional sample cube image used while developing color detection.
- `test/test_copyright.py`: Generated ROS2 package copyright test.
- `test/test_flake8.py`: Generated ROS2 package style test.
- `test/test_pep257.py`: Generated ROS2 package docstring style test.
