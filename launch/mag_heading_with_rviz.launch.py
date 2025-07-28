from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 패키지 경로
    package_dir = get_package_share_directory('mag_heading')
    rviz_config_file = os.path.join(package_dir, 'config', 'rviz_config.rviz')
    
    return LaunchDescription([
        LogInfo(msg="자력계 기반 절대 방향 IMU + rviz2 시각화 시작"),
        
        # 자력계 헤딩 노드
        Node(
            package='mag_heading',
            executable='mag_heading_node',
            name='mag_heading_node',
            output='screen',
            parameters=[],
        ),
        
        # rviz2 실행
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),
        
        LogInfo(msg="시각화 준비 완료! rviz2에서 IMU 움직임 확인 가능")
    ]) 