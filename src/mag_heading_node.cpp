#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <vector>
#include <cmath>

class MagHeadingNode : public rclcpp::Node
{
public:
    MagHeadingNode() : Node("mag_heading_node")
    {
        // 초기화
        calib_file_path_ = std::string(getenv("HOME")) + "/yaw_ws/src/mag_heading/config/mag_calibration.yaml";
        magnetic_declination_ = 0.122; // 서울 2025년 기준 편각 (~7도)
        last_imu_time_ = this->now(); // 시간 초기화 추가
        
        // 캘리브레이션 파라미터 로드 시도
        if (loadCalibrationParameters()) {
            is_calibrated_ = true;
            RCLCPP_INFO(this->get_logger(), "캘리브레이션 파라미터 로드 완료");
            startNormalMode();
        } else {
            RCLCPP_INFO(this->get_logger(), "캘리브레이션 파일 없음. 캘리브레이션 시작");
            RCLCPP_INFO(this->get_logger(), "IMU를 모든 방향으로 천천히 회전시키세요 (약 2-3분)");
            startCalibrationMode();
        }
        
        // 구독자 생성
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu_main", 10,
            std::bind(&MagHeadingNode::imuCallback, this, std::placeholders::_1));
        
        mag_sub_ = this->create_subscription<sensor_msgs::msg::MagneticField>(
            "/magnetometer_main", 10,
            std::bind(&MagHeadingNode::magnetometerCallback, this, std::placeholders::_1));
        
            // 발행자 생성 - 절대 방향이 포함된 IMU
        imu_absolute_pub_ = this->create_publisher<sensor_msgs::msg::Imu>(
            "/imu_absolute", 10);
        
        // TF 브로드캐스터 생성 (rviz2 시각화용)
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        
        // 타이머 생성 (캘리브레이션 진행 상황 출력용)
        timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&MagHeadingNode::timerCallback, this));
    }

private:
    // ROS2 구성요소
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::MagneticField>::SharedPtr mag_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_absolute_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // 상태 변수
    bool is_calibrated_ = false;
    sensor_msgs::msg::Imu::SharedPtr latest_imu_;
    std::string calib_file_path_;
    double magnetic_declination_;
    
    // 캘리브레이션 데이터
    std::vector<Eigen::Vector3d> calib_data_;
    static const size_t MIN_CALIB_SAMPLES = 1000;
    
    // 캘리브레이션 파라미터
    Eigen::Vector3d hard_iron_offset_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d soft_iron_matrix_ = Eigen::Matrix3d::Identity();
    
    // Complementary Filter 변수들
    double gyro_heading_ = 0.0;        // 각속도 적분 기반 heading
    double mag_heading_ = 0.0;         // 자력계 기반 heading  
    double fused_heading_ = 0.0;       // 융합된 최종 heading
    rclcpp::Time last_imu_time_;       // 이전 IMU 시간
    bool filter_initialized_ = false;
    
    // 정지 상태 감지 변수
    double prev_raw_yaw_ = 0.0;        // 이전 원시 yaw
    int stationary_count_ = 0;         // 정지 카운터
    static constexpr int MIN_STATIONARY_COUNT = 10;  // 최소 정지 카운트
    static constexpr double STATIONARY_THRESHOLD = 0.05;  // 정지 임계값 (라디안) 0.01 → 0.05 (약 3도)
    
    // 적응형 필터 파라미터
    static constexpr double BASE_GYRO_WEIGHT = 0.995;   // 기본 각속도 가중치 (99.5%)
    static constexpr double BASE_MAG_WEIGHT = 0.005;    // 기본 자력계 가중치 (0.5%)
    static constexpr double MIN_MAG_WEIGHT = 0.001;     // 최소 자력계 가중치
    static constexpr double MAX_MAG_WEIGHT = 0.02;      // 최대 자력계 가중치
    
    // 간단한 안정성 체크
    int consecutive_good_readings_ = 0;
    static constexpr int MIN_GOOD_READINGS = 3;
    
    // 자력계 안정성 추적
    double prev_mag_heading_ = 0.0;
    int mag_stable_count_ = 0;
    static constexpr int MAG_STABLE_THRESHOLD = 10;

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        // IMU 데이터 유효성 검사
        tf2::Quaternion test_quat(msg->orientation.x, msg->orientation.y, 
                                 msg->orientation.z, msg->orientation.w);
        
        // 쿼터니언 정규화 (중요!)
        test_quat.normalize();
        
        // 정규화된 쿼터니언으로 IMU 메시지 업데이트
        auto normalized_imu = std::make_shared<sensor_msgs::msg::Imu>(*msg);
        normalized_imu->orientation.x = test_quat.x();
        normalized_imu->orientation.y = test_quat.y();
        normalized_imu->orientation.z = test_quat.z();
        normalized_imu->orientation.w = test_quat.w();
        
        // 원시 IMU의 yaw 추출
        tf2::Matrix3x3 rot(test_quat);
        double roll, pitch, current_raw_yaw;
        rot.getRPY(roll, pitch, current_raw_yaw);
        
        // 이전 IMU 데이터와 비교해서 yaw 변화량 계산
        if (latest_imu_) {
            tf2::Quaternion prev_quat(latest_imu_->orientation.x, latest_imu_->orientation.y,
                                     latest_imu_->orientation.z, latest_imu_->orientation.w);
            tf2::Matrix3x3 prev_rot(prev_quat);
            double prev_roll, prev_pitch, prev_raw_yaw;
            prev_rot.getRPY(prev_roll, prev_pitch, prev_raw_yaw);
            
            // Yaw 변화량 계산 (각도 점프 처리)
            double yaw_change = current_raw_yaw - prev_raw_yaw;
            while (yaw_change > M_PI) yaw_change -= 2.0 * M_PI;
            while (yaw_change <= -M_PI) yaw_change += 2.0 * M_PI;
            
            // 정지 상태 감지
            if (abs(yaw_change) < STATIONARY_THRESHOLD) {
                stationary_count_++;
            } else {
                stationary_count_ = 0;  // 움직임 감지시 리셋
            }
            
            // 융합된 heading에 실제 yaw 변화량 적용
            if (filter_initialized_) {
                fused_heading_ += yaw_change;
                gyro_heading_ += yaw_change;
                
                // -π ~ π 정규화
                while (fused_heading_ > M_PI) fused_heading_ -= 2.0 * M_PI;
                while (fused_heading_ <= -M_PI) fused_heading_ += 2.0 * M_PI;
                while (gyro_heading_ > M_PI) gyro_heading_ -= 2.0 * M_PI;
                while (gyro_heading_ <= -M_PI) gyro_heading_ += 2.0 * M_PI;
            }
        }
        
        prev_raw_yaw_ = current_raw_yaw;  // 다음 비교를 위해 저장
        
        latest_imu_ = normalized_imu;
        last_imu_time_ = msg->header.stamp;
    }
    
    void magnetometerCallback(const sensor_msgs::msg::MagneticField::SharedPtr msg)
    {
        if (!is_calibrated_) {
            collectCalibrationData(msg);
            checkCalibrationComplete();
        } else {
            publishCorrectedHeading(msg);
        }
    }
    
    void timerCallback()
    {
        if (!is_calibrated_ && !calib_data_.empty()) {
            // 캘리브레이션 품질 체크
            double coverage = calculateCoverageQuality();
            RCLCPP_INFO(this->get_logger(), 
                "캘리브레이션 진행중... %zu/%zu 샘플 수집, 3D 커버리지: %.1f%%", 
                calib_data_.size(), MIN_CALIB_SAMPLES, coverage * 100.0);
            
            if (coverage < 0.3 && calib_data_.size() > 200) {
                RCLCPP_WARN(this->get_logger(), 
                    "3D 회전이 부족합니다! X,Y,Z축 모든 방향으로 회전시키세요");
            }
        }
    }
    
    void startCalibrationMode()
    {
        calib_data_.clear();
        hard_iron_offset_.setZero();
        soft_iron_matrix_.setIdentity();
    }
    
    void startNormalMode()
    {
        RCLCPP_INFO(this->get_logger(), "정상 모드 시작. /imu_absolute 토픽으로 절대 방향 IMU 발행");
    }
    
    void collectCalibrationData(const sensor_msgs::msg::MagneticField::SharedPtr msg)
    {
        Eigen::Vector3d mag_data(msg->magnetic_field.x, 
                                msg->magnetic_field.y, 
                                msg->magnetic_field.z);
        calib_data_.push_back(mag_data);
    }
    
    void checkCalibrationComplete()
    {
        if (calib_data_.size() >= MIN_CALIB_SAMPLES) {
            RCLCPP_INFO(this->get_logger(), "충분한 데이터 수집 완료. 캘리브레이션 계산 중...");
            calculateCalibrationParameters();
            saveParametersToFile();
            is_calibrated_ = true;
            RCLCPP_INFO(this->get_logger(), 
                "캘리브레이션 완료! 다음 실행부터는 자동으로 생략됩니다.");
            startNormalMode();
        }
    }
    
    void calculateCalibrationParameters()
    {
        if (calib_data_.size() < 100) return;
        
        // 더 정확한 sphere fitting을 통한 hard iron 계산
        // 최소제곱법으로 구심점 찾기
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (const auto& data : calib_data_) {
            centroid += data;
        }
        centroid /= calib_data_.size();
        
        // 반복적으로 중심점 개선
        Eigen::Vector3d center = centroid;
        for (int iter = 0; iter < 10; ++iter) {
            Eigen::Vector3d sum_weighted = Eigen::Vector3d::Zero();
            double sum_weights = 0.0;
            
            for (const auto& data : calib_data_) {
                double dist = (data - center).norm();
                if (dist > 1e-8) {  // 0으로 나누기 방지
                    double weight = 1.0 / dist;
                    sum_weighted += weight * data;
                    sum_weights += weight;
                }
            }
            
            if (sum_weights > 0) {
                center = sum_weighted / sum_weights;
            }
        }
        
        hard_iron_offset_ = center;
        
        // 스케일 계산 (반지름 추정)
        double avg_radius = 0.0;
        for (const auto& data : calib_data_) {
            avg_radius += (data - hard_iron_offset_).norm();
        }
        avg_radius /= calib_data_.size();
        
        // Soft iron은 단순화해서 Identity로 유지 (복잡한 ellipsoid fitting 필요)
        soft_iron_matrix_ = Eigen::Matrix3d::Identity();
        
        RCLCPP_INFO(this->get_logger(), 
            "Hard iron offset: [%.6f, %.6f, %.6f], Avg radius: %.6f",
            hard_iron_offset_.x(), hard_iron_offset_.y(), hard_iron_offset_.z(), avg_radius);
    }
    
    void publishCorrectedHeading(const sensor_msgs::msg::MagneticField::SharedPtr msg)
    {
        if (!latest_imu_) return;
        
        // 자력계 보정
        Eigen::Vector3d mag_raw(msg->magnetic_field.x, 
                               msg->magnetic_field.y, 
                               msg->magnetic_field.z);
        
        // 자력계 품질 체크 (관대하게)
        double mag_magnitude = mag_raw.norm();
        if (mag_magnitude < 2e-5 || mag_magnitude > 8e-5) {
            consecutive_good_readings_ = 0;
            return;
        }
        consecutive_good_readings_++;
        
        Eigen::Vector3d mag_corrected = soft_iron_matrix_ * (mag_raw - hard_iron_offset_);
        
        // IMU 쿼터니언을 회전행렬로 변환
        tf2::Quaternion q(latest_imu_->orientation.x, latest_imu_->orientation.y,
                         latest_imu_->orientation.z, latest_imu_->orientation.w);
        tf2::Matrix3x3 R(q);
        
        // Tilt 보정: 자력계를 수평면으로 투영
        tf2::Vector3 mag_tf(mag_corrected.x(), mag_corrected.y(), mag_corrected.z());
        tf2::Vector3 mag_horizontal = R.transpose() * mag_tf;
        
        // 자력계 기반 heading 계산 (드리프트 보정용)
        mag_heading_ = atan2(-mag_horizontal.y(), mag_horizontal.x()) + magnetic_declination_;
        while (mag_heading_ > M_PI) mag_heading_ -= 2.0 * M_PI;
        while (mag_heading_ <= -M_PI) mag_heading_ += 2.0 * M_PI;
        
        // 자력계 안정성 체크 (settling time 고려)
        double mag_change = abs(mag_heading_ - prev_mag_heading_);
        if (mag_change > M_PI) mag_change = 2*M_PI - mag_change;
        
        if (mag_change < M_PI/180.0) {  // 1도 미만 변화
            mag_stable_count_ = std::min(mag_stable_count_ + 1, MAG_STABLE_THRESHOLD);
        } else {
            mag_stable_count_ = 0;  // 큰 변화 감지시 리셋
        }
        prev_mag_heading_ = mag_heading_;
        
        // 적응형 가중치 계산
        double mag_stability_ratio = (double)mag_stable_count_ / MAG_STABLE_THRESHOLD;
        double adaptive_mag_weight = MIN_MAG_WEIGHT + 
            (MAX_MAG_WEIGHT - MIN_MAG_WEIGHT) * mag_stability_ratio;
        double adaptive_gyro_weight = 1.0 - adaptive_mag_weight;
        
        // 드리프트 보정만 수행 (실제 움직임은 IMU callback에서 처리됨)
        if (!filter_initialized_) {
            // 초기화: 자력계 값으로 시작
            gyro_heading_ = mag_heading_;
            fused_heading_ = mag_heading_;
            filter_initialized_ = true;
        } else {
            // 드리프트 보정: 움직이고 있을 때만 적용
            bool is_stationary = (stationary_count_ >= MIN_STATIONARY_COUNT);
            
            if (!is_stationary) {
                // 움직이고 있을 때만 드리프트 보정 적용
                double heading_error = mag_heading_ - fused_heading_;
                while (heading_error > M_PI) heading_error -= 2.0 * M_PI;
                while (heading_error <= -M_PI) heading_error += 2.0 * M_PI;
                
                // 매우 작은 드리프트 보정만 적용 (실제 움직임은 억제하지 않음)
                double drift_correction = adaptive_mag_weight * 0.05 * heading_error;  // 5%로 더 감소
                fused_heading_ += drift_correction;
                gyro_heading_ += drift_correction;
            }
            // 정지 상태에서는 드리프트 보정 안함 (현재 값 유지)
        }
        
        // 정규화
        while (fused_heading_ > M_PI) fused_heading_ -= 2.0 * M_PI;
        while (fused_heading_ <= -M_PI) fused_heading_ += 2.0 * M_PI;
        
        // 최소한의 안정성 체크 (품질 좋은 데이터만)
        if (consecutive_good_readings_ >= MIN_GOOD_READINGS) {
            // 원본 IMU의 orientation을 절대 방향으로 변환
            auto imu_absolute_msg = std::make_shared<sensor_msgs::msg::Imu>(*latest_imu_);
            
            // 원본 IMU의 쿼터니언 가져오기
            tf2::Quaternion original_quat(latest_imu_->orientation.x, 
                                         latest_imu_->orientation.y,
                                         latest_imu_->orientation.z, 
                                         latest_imu_->orientation.w);
            original_quat.normalize();
            
            // 더 안정적인 방법: 쿼터니언 직접 조작으로 yaw 교체
            // Z축 회전만 추출 (Roll, Pitch 보존)
            tf2::Matrix3x3 original_rot(original_quat);
            double roll, pitch, relative_yaw;
            original_rot.getRPY(roll, pitch, relative_yaw);
            
            // 절대 yaw로 새 쿼터니언 생성 (연속성 보장)
            tf2::Quaternion absolute_quat;
            absolute_quat.setRPY(roll, pitch, fused_heading_);
            absolute_quat.normalize();
            
            // 절대 orientation 설정
            imu_absolute_msg->orientation.x = absolute_quat.x();
            imu_absolute_msg->orientation.y = absolute_quat.y();
            imu_absolute_msg->orientation.z = absolute_quat.z();
            imu_absolute_msg->orientation.w = absolute_quat.w();
            
            // 타임스탬프 업데이트
            imu_absolute_msg->header.stamp = msg->header.stamp;
            imu_absolute_msg->header.frame_id = "imu_absolute_link";
            
            imu_absolute_pub_->publish(*imu_absolute_msg);
            
            // rviz2 시각화를 위한 TF 발행
            publishTF(imu_absolute_msg, absolute_quat);
            
            // 절대 IMU 정보는 아래 통합 로그에서 출력
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                "자력계 품질 불량 (%d/%d) - 각속도만 사용중", 
                consecutive_good_readings_, MIN_GOOD_READINGS);
        }
        
        // 통합 디버그 출력 (0.5초마다, 모든 정보 한 줄)
        static auto last_debug = this->now();
        if ((this->now() - last_debug).seconds() > 0.5) {  // 3초 → 0.5초로 변경
            double mag_raw_magnitude = sqrt(msg->magnetic_field.x * msg->magnetic_field.x + 
                                          msg->magnetic_field.y * msg->magnetic_field.y + 
                                          msg->magnetic_field.z * msg->magnetic_field.z);
            
            // 원시 IMU 방향 (상대 좌표계)
            tf2::Quaternion raw_quat(latest_imu_->orientation.x, latest_imu_->orientation.y,
                                    latest_imu_->orientation.z, latest_imu_->orientation.w);
            tf2::Matrix3x3 raw_rot(raw_quat);
            double raw_roll, raw_pitch, raw_yaw;
            raw_rot.getRPY(raw_roll, raw_pitch, raw_yaw);
            
            // 절대 IMU 방향 (발행되는 값)
            tf2::Quaternion abs_quat;
            abs_quat.setRPY(raw_roll, raw_pitch, fused_heading_);
            tf2::Matrix3x3 abs_rot(abs_quat);
            double abs_roll, abs_pitch, abs_yaw;
            abs_rot.getRPY(abs_roll, abs_pitch, abs_yaw);
            
            // 원시 자력계 각도 (tilt 보정 전)
            double raw_mag_heading = atan2(-msg->magnetic_field.y, msg->magnetic_field.x) + magnetic_declination_;
            while (raw_mag_heading > M_PI) raw_mag_heading -= 2.0 * M_PI;
            while (raw_mag_heading <= -M_PI) raw_mag_heading += 2.0 * M_PI;
            
            // 각도를 도(degree)로 변환
            double raw_roll_deg = raw_roll * 180.0 / M_PI;
            double raw_pitch_deg = raw_pitch * 180.0 / M_PI;
            double raw_yaw_deg = raw_yaw * 180.0 / M_PI;
            if (raw_yaw_deg < 0) raw_yaw_deg += 360.0;
            
            double abs_roll_deg = abs_roll * 180.0 / M_PI;
            double abs_pitch_deg = abs_pitch * 180.0 / M_PI;
            double abs_yaw_deg = abs_yaw * 180.0 / M_PI;
            if (abs_yaw_deg < 0) abs_yaw_deg += 360.0;
            
            double raw_mag_deg = raw_mag_heading * 180.0 / M_PI;
            if (raw_mag_deg < 0) raw_mag_deg += 360.0;
            
            // 출력용으로만 0~360도 변환
            double fused_deg = fused_heading_ * 180.0 / M_PI;
            if (fused_deg < 0) fused_deg += 360.0;
            double gyro_deg = gyro_heading_ * 180.0 / M_PI;
            if (gyro_deg < 0) gyro_deg += 360.0;
            double mag_deg = mag_heading_ * 180.0 / M_PI;
            if (mag_deg < 0) mag_deg += 360.0;
            
            RCLCPP_INFO(this->get_logger(), 
                "원시IMU: R=%.1f° P=%.1f° Y=%.1f° | 📤절대IMU: R=%.1f° P=%.1f° Y=%.1f° | 원시자력계=%.1f° 융합=%.1f° 각속도=%.1f° 틸트보정자력계=%.1f° | 자력계크기=%.1fe-5 안정성=%d/10 가중치=%.1f%% | 정지=%d/%d", 
                raw_roll_deg, raw_pitch_deg, raw_yaw_deg,
                abs_roll_deg, abs_pitch_deg, abs_yaw_deg,
                raw_mag_deg, fused_deg, gyro_deg, mag_deg, 
                mag_raw_magnitude*1e5, mag_stable_count_, adaptive_mag_weight*100,
                stationary_count_, MIN_STATIONARY_COUNT);
            last_debug = this->now();
        }
    }
    
    bool loadCalibrationParameters()
    {
        try {
            YAML::Node config = YAML::LoadFile(calib_file_path_);
            
            auto offset = config["hard_iron_offset"];
            hard_iron_offset_ = Eigen::Vector3d(
                offset[0].as<double>(),
                offset[1].as<double>(),
                offset[2].as<double>()
            );
            
            auto matrix = config["soft_iron_matrix"];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    soft_iron_matrix_(i, j) = matrix[i][j].as<double>();
                }
            }
            
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    double calculateCoverageQuality()
    {
        if (calib_data_.size() < 50) return 0.0;
        
        // 각 축의 최대/최소값 범위 계산
        Eigen::Vector3d min_vals = calib_data_[0];
        Eigen::Vector3d max_vals = calib_data_[0];
        
        for (const auto& data : calib_data_) {
            for (int i = 0; i < 3; ++i) {
                min_vals[i] = std::min(min_vals[i], data[i]);
                max_vals[i] = std::max(max_vals[i], data[i]);
            }
        }
        
        // 각 축의 커버리지 계산
        Eigen::Vector3d ranges = max_vals - min_vals;
        double avg_range = ranges.mean();
        double coverage = 0.0;
        
        // 범위가 고르게 분포되었는지 확인
        for (int i = 0; i < 3; ++i) {
            if (avg_range > 1e-8) {
                coverage += std::min(1.0, ranges[i] / avg_range);
            }
        }
        
        return coverage / 3.0;  // 0~1 범위로 정규화
    }
    
    void saveParametersToFile()
    {
        try {
            // config 디렉토리 생성 (패키지 내)
            size_t last_slash = calib_file_path_.find_last_of("/");
            if (last_slash != std::string::npos) {
                std::string config_dir = calib_file_path_.substr(0, last_slash);
                std::system(("mkdir -p " + config_dir).c_str());
            }
            
            YAML::Node config;
            
            config["hard_iron_offset"] = std::vector<double>{
                hard_iron_offset_.x(),
                hard_iron_offset_.y(),
                hard_iron_offset_.z()
            };
            
            std::vector<std::vector<double>> matrix_data;
            for (int i = 0; i < 3; ++i) {
                std::vector<double> row;
                for (int j = 0; j < 3; ++j) {
                    row.push_back(soft_iron_matrix_(i, j));
                }
                matrix_data.push_back(row);
            }
            config["soft_iron_matrix"] = matrix_data;
            
            // 캘리브레이션 품질 정보 추가
            double final_coverage = calculateCoverageQuality();
            config["calibration_quality"] = final_coverage;
            config["num_samples"] = calib_data_.size();
            
            std::ofstream fout(calib_file_path_);
            fout << config;
            fout.close();
            
            RCLCPP_INFO(this->get_logger(), 
                "캘리브레이션 저장: %s (품질: %.1f%%)", 
                calib_file_path_.c_str(), final_coverage * 100.0);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "파라미터 저장 실패: %s", e.what());
        }
    }
    
    void publishTF(const sensor_msgs::msg::Imu::SharedPtr& imu_msg, const tf2::Quaternion& absolute_quat)
    {
        // 원본 IMU 쿼터니언 정규화 확인
        tf2::Quaternion original_quat(latest_imu_->orientation.x, 
                                     latest_imu_->orientation.y,
                                     latest_imu_->orientation.z, 
                                     latest_imu_->orientation.w);
        original_quat.normalize();
        
        // 절대 쿼터니언도 정규화 확인
        tf2::Quaternion normalized_absolute_quat = absolute_quat;
        normalized_absolute_quat.normalize();
        
        // 원본 IMU TF (상대 방향)
        geometry_msgs::msg::TransformStamped original_transform;
        original_transform.header.stamp = imu_msg->header.stamp;
        original_transform.header.frame_id = "world";
        original_transform.child_frame_id = "imu_relative";
        original_transform.transform.translation.x = 0.0;
        original_transform.transform.translation.y = 0.0;
        original_transform.transform.translation.z = 0.0;
        original_transform.transform.rotation.x = original_quat.x();
        original_transform.transform.rotation.y = original_quat.y();
        original_transform.transform.rotation.z = original_quat.z();
        original_transform.transform.rotation.w = original_quat.w();
        
        // 절대 방향 IMU TF (진북 기준)
        geometry_msgs::msg::TransformStamped absolute_transform;
        absolute_transform.header.stamp = imu_msg->header.stamp;
        absolute_transform.header.frame_id = "world";
        absolute_transform.child_frame_id = "imu_absolute";
        absolute_transform.transform.translation.x = 0.5;  // 오른쪽으로 0.5m 떨어뜨려서 구분
        absolute_transform.transform.translation.y = 0.0;
        absolute_transform.transform.translation.z = 0.0;
        absolute_transform.transform.rotation.x = normalized_absolute_quat.x();
        absolute_transform.transform.rotation.y = normalized_absolute_quat.y();
        absolute_transform.transform.rotation.z = normalized_absolute_quat.z();
        absolute_transform.transform.rotation.w = normalized_absolute_quat.w();
        
        // 자력계 방향 표시 (진북 표시용)
        geometry_msgs::msg::TransformStamped north_transform;
        north_transform.header.stamp = imu_msg->header.stamp;
        north_transform.header.frame_id = "world";
        north_transform.child_frame_id = "true_north";
        north_transform.transform.translation.x = -0.5;  // 왼쪽으로 0.5m
        north_transform.transform.translation.y = 0.0;
        north_transform.transform.translation.z = 0.0;
        tf2::Quaternion north_quat;
        north_quat.setRPY(0, 0, 0);  // 진북(0도) 방향
        north_transform.transform.rotation.x = north_quat.x();
        north_transform.transform.rotation.y = north_quat.y();
        north_transform.transform.rotation.z = north_quat.z();
        north_transform.transform.rotation.w = north_quat.w();
        
        // TF 발행
        tf_broadcaster_->sendTransform(original_transform);
        tf_broadcaster_->sendTransform(absolute_transform);
        tf_broadcaster_->sendTransform(north_transform);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MagHeadingNode>());
    rclcpp::shutdown();
    return 0;
} 