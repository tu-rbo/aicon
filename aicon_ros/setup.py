from setuptools import setup

package_name = "aicon_ros_python_wrapping"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],  # or use find_packages()
    data_files=[
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],  # or other dependencies
    zip_safe=True,
    maintainer="Vito Mengers",
    maintainer_email="v.mengers@tu-berlin.de",
    description="Python wrapping for the middleware for AICON based on ROS communication between the components."
    " An asynchronous and multi-machine alternative to the default synchronous default middleware.",
    license="MIT",
)
