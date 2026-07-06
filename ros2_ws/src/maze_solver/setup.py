from glob import glob
from os.path import join

from setuptools import setup

package_name = 'maze_solver'

setup(
    name=package_name,
    version='0.0.1',
    py_modules=['camera_tilt_node', 'maze_solver_node', 'maze_explorer_node'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuweinan',
    maintainer_email='yuweinan0415@gmail.com',
    description='Known-maze graph solver that commands DarthMaulPI motion primitives.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_tilt_node = camera_tilt_node:main',
            'maze_solver_node = maze_solver_node:main',
            'maze_explorer_node = maze_explorer_node:main',
        ],
    },
)
