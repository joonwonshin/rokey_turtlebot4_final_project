import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'sqlite3db'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, package_name), glob(package_name + '/*.py')),
        (os.path.join('share', package_name, package_name), glob(package_name + '/*.html')),
        (os.path.join('share', package_name, package_name), glob(package_name + '/*.png')),
    ],
    install_requires=['setuptools', 'openpyxl'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='ast04141@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'create_db = sqlite3db.create_db:main',
            'ros2_db_node = sqlite3db.ros2_db_node:main',
            'app = sqlite3db.app:main',
            'db_update = sqlite3db.db_update:main',
        ],
    },
)
