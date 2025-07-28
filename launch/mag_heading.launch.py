from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo

def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg="자력계 기반 절대 방향 계산 노드 시작"),
        
        Node(
            package='mag_heading',
            executable='mag_heading_node',
            name='mag_heading_node',
            output='screen',
            parameters=[],
            remappings=[
                # 기본 토픽 이름을 사용하므로 리매핑 불필요
                # ('/imu_main', '/imu_main'),
                # ('/magnetometer_main', '/magnetometer_main'),
                # ('/heading', '/heading'),
            ]
        ),
        
        LogInfo(msg="절대 방향 IMU 노드 실행 완료. /imu_absolute 토픽에서 절대 방향 IMU 확인 가능")
    ]) 