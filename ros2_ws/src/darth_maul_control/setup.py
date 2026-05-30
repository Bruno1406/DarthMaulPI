from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'darth_maul_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dtdav',
    maintainer_email='dtdav@todo.todo',
    description='Safe waypoint-following motion-control foundation for the DarthMaulPI robot.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = darth_maul_control.control_node:main',
        ],
    },
)
