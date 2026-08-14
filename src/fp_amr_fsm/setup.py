from setuptools import find_packages, setup

package_name = 'fp_amr_fsm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 관제 웹 모니터 (rosbridge 기반) — html 과 배경 지도 이미지
        ('share/' + package_name + '/web',
            ['web/fleet_monitor.html', 'web/map.png']),
        # fleet_fsm 이 사용하는 지도 (nav2 map_server 와 동일한 pgm+yaml 쌍)
        ('share/' + package_name + '/maps',
            ['maps/final_project.pgm', 'maps/final_project.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='kings0625@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'amr_patrol_emer_helmet = fp_amr_fsm.amr_patrol_emer_helmet:main',
            'fleet_fsm = fp_amr_fsm.fleet_fsm:main',
            'safety_alert_bridge = fp_amr_fsm.safety_alert_bridge:main'
        ],
    },
)
