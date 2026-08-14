import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'amr_aruco'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={package_name: ['*.json']},
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='joon5514@naver.com',
    description='ArUco marker detection and patrol/emergency/helmet delivery FSM for TurtleBot4 AMR',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco_detect = amr_aruco.aruco_detect:main',
            'amr_patrol_emer_helmet = amr_aruco.amr_patrol_emer_helmet:main',
        ],
    },
)
