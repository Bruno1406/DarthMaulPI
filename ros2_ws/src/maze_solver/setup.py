from setuptools import find_packages, setup

package_name = "maze_solver"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yuweinan",
    maintainer_email="yuweinan0415@gmail.com",
    description="Maze graph solver that sends motion primitive goals.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "maze_solver_node = maze_solver.maze_solver_node:main",
        ],
    },
)
