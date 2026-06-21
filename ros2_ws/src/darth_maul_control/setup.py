from glob import glob
import os

from setuptools import find_packages, setup

package_name = "darth_maul_control"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dtdav",
    maintainer_email="chendavidtimothy@gmail.com",
    description="Nav2-backed motion primitive executor for DarthMaulPI.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "motion_primitive_server = darth_maul_control.motion_primitive_server:main",
        ],
    },
)
