/*
 * Copyright 2017 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/pose_extrapolator.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

PoseExtrapolator::PoseExtrapolator(const common::Duration pose_queue_duration,
                                   double imu_gravity_time_constant)
    : pose_queue_duration_(pose_queue_duration),
      gravity_time_constant_(imu_gravity_time_constant),
      cached_extrapolated_pose_{common::Time::min(),
                                transform::Rigid3d::Identity()} {}

std::unique_ptr<PoseExtrapolator> PoseExtrapolator::InitializeWithImu(
    const common::Duration pose_queue_duration,
    const double imu_gravity_time_constant, const sensor::ImuData& imu_data) {
  auto extrapolator = absl::make_unique<PoseExtrapolator>(
      pose_queue_duration, imu_gravity_time_constant);
  extrapolator->AddImuData(imu_data);
  extrapolator->imu_tracker_ =
      absl::make_unique<ImuTracker>(imu_gravity_time_constant, imu_data.time);
  extrapolator->imu_tracker_->AddImuLinearAccelerationObservation(
      imu_data.linear_acceleration);
  extrapolator->imu_tracker_->AddImuAngularVelocityObservation(
      imu_data.angular_velocity);
  extrapolator->imu_tracker_->Advance(imu_data.time);
  extrapolator->AddPose(
      imu_data.time,
      transform::Rigid3d::Rotation(extrapolator->imu_tracker_->orientation()));
  return extrapolator;
}

sensor::OdometryData GetTimedOdomety(
    std::deque<sensor::OdometryData> odometry_data, const common::Time time) {
  for (auto& odom : odometry_data) {
    if (odom.time > time) {
      return odom;
    }
  }
  return odometry_data.back();
}

common::Time PoseExtrapolator::GetLastPoseTime() const {
  if (timed_pose_queue_.empty()) {
    return common::Time::min();
  }
  return timed_pose_queue_.back().time;
}

common::Time PoseExtrapolator::GetLastExtrapolatedTime() const {
  if (!extrapolation_imu_tracker_) {
    return common::Time::min();
  }
  return extrapolation_imu_tracker_->time();
}

void PoseExtrapolator::AddPose(const common::Time time,
                               const transform::Rigid3d& pose) {
  if (imu_tracker_ == nullptr) {
    common::Time tracker_start = time;
    if (!imu_data_.empty()) {
      tracker_start = std::min(tracker_start, imu_data_.front().time);
    }
    imu_tracker_ =
        absl::make_unique<ImuTracker>(gravity_time_constant_, tracker_start);
  }
  timed_pose_queue_.push_back(TimedPose{time, pose});
  while (timed_pose_queue_.size() > 2 &&
         timed_pose_queue_[1].time <= time - pose_queue_duration_) {
    timed_pose_queue_.pop_front();
  }
  UpdateVelocitiesFromPoses();
  AdvanceImuTracker(time, imu_tracker_.get());
  TrimImuData();
  TrimOdometryData();
  // add by dh, 记录与定位时间最近的里程计数据
  if (!odometry_data_.empty()) {
    reference_odometry_ = GetTimedOdomety(odometry_data_, time);
  } else {
    reference_odometry_ = {common::Time::min(), transform::Rigid3d::Identity()};
  }
  odometry_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
  extrapolation_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
}

void PoseExtrapolator::AddImuData(const sensor::ImuData& imu_data) {
  CHECK(timed_pose_queue_.empty() ||
        imu_data.time >= timed_pose_queue_.back().time);
  imu_data_.push_back(imu_data);
  TrimImuData();
}

void PoseExtrapolator::AddOdometryData(
    const sensor::OdometryData& odometry_data) {
  if (odometry_data.time < timed_pose_queue_.back().time) {
    return;
  }
  CHECK(timed_pose_queue_.empty() ||
        odometry_data.time >= timed_pose_queue_.back().time);
  if (reference_odometry_.time == common::Time::min()) {
    reference_odometry_ = odometry_data;
  }
  odometry_data_.push_back(odometry_data);
  TrimOdometryData();
  if (odometry_data_.size() < 2) {
    return;
  }
  // TODO(whess): Improve by using more than just the last two odometry poses.
  // Compute extrapolation in the tracking frame.
  auto oldest = odometry_data_.front();
  if(odometry_data_.size() > 5){
    oldest = odometry_data_[odometry_data_.size() - 6];
  }
  const sensor::OdometryData& odometry_data_oldest = oldest;
  const sensor::OdometryData& odometry_data_newest = odometry_data_.back();
  const double odometry_time_delta =
      common::ToSeconds(odometry_data_oldest.time - odometry_data_newest.time);
  const transform::Rigid3d odometry_pose_delta =
      odometry_data_newest.pose.inverse() * odometry_data_oldest.pose;
  angular_velocity_from_odometry_ =
      transform::RotationQuaternionToAngleAxisVector(
          odometry_pose_delta.rotation()) /
      odometry_time_delta;
  if(std::abs(angular_velocity_from_odometry_.z()) > common::DegToRad(200)){
    LOG(WARNING)<<"angular_velocity_from_odometry_: "<<angular_velocity_from_odometry_.z()<<", "<<common::RadToDeg(angular_velocity_from_odometry_.z())<<"°";
    LOG(WARNING)<<"oldest: ["<<odometry_data_oldest.pose.translation().x()<<", "<<odometry_data_oldest.pose.translation().y()<<"], "<<transform::RotationQuaternionToAngleAxisVector(odometry_data_oldest.pose.rotation()).z();
    LOG(WARNING)<<"newest: [ "<<odometry_data_newest.pose.translation().x()<<", "<<odometry_data_newest.pose.translation().y()<<"], "<<transform::RotationQuaternionToAngleAxisVector(odometry_data_newest.pose.rotation()).z();
    LOG(WARNING)<<"odometry_time_delta: "<<odometry_time_delta<<", "<<odometry_data_oldest.time<<", "<<odometry_data_newest.time;
  }
  if(std::abs(odometry_time_delta) < 0.0012)
  {
    return;
  }
  if (timed_pose_queue_.empty()) {
    return;
  }
  const Eigen::Vector3d
      linear_velocity_in_tracking_frame_at_newest_odometry_time =
          odometry_pose_delta.translation() / odometry_time_delta;
  const Eigen::Quaterniond orientation_at_newest_odometry_time =
      timed_pose_queue_.back().pose.rotation() *
      ExtrapolateRotation(odometry_data_newest.time,
                          odometry_imu_tracker_.get());
  linear_velocity_from_odometry_ =
      orientation_at_newest_odometry_time *
      linear_velocity_in_tracking_frame_at_newest_odometry_time;
}

transform::Rigid3d PoseExtrapolator::ExtrapolatePose(const common::Time time) {
  /*
    by dh
    cartographer 强依赖激光数据,位姿外推器仅使用 odom 和 imu 计算线速度和角速度
    通过线速度,角速度,时间差和上一帧激光定位使用匀速运动模型来推出当前位姿,
    这对于激光数据会暂停的情况极度不友好,此时并不满足匀速运动模型
    所以需要修改为通过里程数据累积和上一帧激光定位来推出当前位姿
  */
#if 0
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  CHECK_GE(time, newest_timed_pose.time);
  if (cached_extrapolated_pose_.time != time) {
    const Eigen::Vector3d translation =
        ExtrapolateTranslation(time) + newest_timed_pose.pose.translation();
    const Eigen::Quaterniond rotation =
        newest_timed_pose.pose.rotation() *
        ExtrapolateRotation(time, extrapolation_imu_tracker_.get());
    cached_extrapolated_pose_ =
        TimedPose{time, transform::Rigid3d{translation, rotation}};
  }
  return cached_extrapolated_pose_.pose;
#else
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  CHECK_GE(time, newest_timed_pose.time);
  auto duration = common::ToSeconds(time - newest_timed_pose.time);
  if (duration > 0.5) {
    if (cached_extrapolated_pose_.time != time) {
      sensor::OdometryData newest_odomety_ =
          GetTimedOdomety(odometry_data_, time);
      
      transform::Rigid3d odom_diff =
          reference_odometry_.pose.inverse() * newest_odomety_.pose;
      cached_extrapolated_pose_ =
          TimedPose{time, newest_timed_pose.pose * odom_diff};
    }
  } else {
    if (cached_extrapolated_pose_.time != time) {
      auto delta_translation = ExtrapolateTranslation(time);
      const Eigen::Vector3d translation =
          delta_translation + newest_timed_pose.pose.translation();
      auto delta_rotation =
          ExtrapolateRotation(time, extrapolation_imu_tracker_.get());
      const Eigen::Quaterniond rotation =
          newest_timed_pose.pose.rotation() * delta_rotation;
      cached_extrapolated_pose_ =
          TimedPose{time, transform::Rigid3d{translation, rotation}};
    }
  }
  return cached_extrapolated_pose_.pose;
#endif
}

Eigen::Vector3d PoseExtrapolator::AngularVelocityFromOdometry()
{
  return angular_velocity_from_odometry_;
}
transform::Rigid3d PoseExtrapolator::ExtrapolatePoseLog(
    const common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  CHECK_GE(time, newest_timed_pose.time);
  LOG(INFO)<< "time: " << time << ", newest_timed_pose.time: " << newest_timed_pose.time;
  auto duration = common::ToSeconds(time - newest_timed_pose.time);
  if (duration > 0.5) {
    if (cached_extrapolated_pose_.time != time) {
      sensor::OdometryData newest_odomety_ =
          GetTimedOdomety(odometry_data_, time);
      LOG(INFO)<< "newest_timed_pose: " << newest_timed_pose.pose.translation().x() << ","
              << newest_timed_pose.pose.translation().y() << ","
              << newest_timed_pose.pose.translation().z() << ","
              << common::RadToDeg(transform::QuaternionToEulerAngles(newest_timed_pose.pose.rotation()).z()) << "°,"
              << newest_timed_pose.time;
      LOG(INFO)<< "newest_odomety_: " << newest_odomety_.pose.translation().x() << ","
              << newest_odomety_.pose.translation().y() << ","
              << newest_odomety_.pose.translation().z() << "," 
              << common::RadToDeg(transform::QuaternionToEulerAngles(newest_odomety_.pose.rotation()).z()) << "°,"
              << newest_odomety_.time;
      LOG(INFO)<< "reference_odometry_: " << reference_odometry_.pose.translation().x() << ","
              << reference_odometry_.pose.translation().y() << ","
              << reference_odometry_.pose.translation().z() << "," 
              << common::RadToDeg(transform::QuaternionToEulerAngles(reference_odometry_.pose.rotation()).z()) << "°,"
              << reference_odometry_.time;

      transform::Rigid3d odom_diff =
          reference_odometry_.pose.inverse() * newest_odomety_.pose;
      cached_extrapolated_pose_ =
          TimedPose{time, newest_timed_pose.pose * odom_diff};
    }
    else{
      LOG(INFO)<< "cached_extrapolated_pose_: " << cached_extrapolated_pose_.pose.translation().x() << ","
              << cached_extrapolated_pose_.pose.translation().y() << ","
              << cached_extrapolated_pose_.pose.translation().z() << ","
              << common::RadToDeg(transform::QuaternionToEulerAngles(cached_extrapolated_pose_.pose.rotation()).z()) << "°,"
              << cached_extrapolated_pose_.time;
      sensor::OdometryData newest_odomety_ =
          GetTimedOdomety(odometry_data_, time);
      LOG(INFO)<< "newest_timed_pose: " << newest_timed_pose.pose.translation().x() << ","
              << newest_timed_pose.pose.translation().y() << ","
              << newest_timed_pose.pose.translation().z() << ","
              << common::RadToDeg(transform::QuaternionToEulerAngles(newest_timed_pose.pose.rotation()).z()) << "°,"
              << newest_timed_pose.time;
      LOG(INFO)<< "newest_odomety_: " << newest_odomety_.pose.translation().x() << ","
              << newest_odomety_.pose.translation().y() << ","
              << newest_odomety_.pose.translation().z() << "," 
              << common::RadToDeg(transform::QuaternionToEulerAngles(newest_odomety_.pose.rotation()).z()) << "°,"
              << newest_odomety_.time;
      LOG(INFO)<< "reference_odometry_: " << reference_odometry_.pose.translation().x() << ","
              << reference_odometry_.pose.translation().y() << ","
              << reference_odometry_.pose.translation().z() << "," 
              << common::RadToDeg(transform::QuaternionToEulerAngles(reference_odometry_.pose.rotation()).z()) << "°,"
              << reference_odometry_.time;
    }
  } else {
    LOG(INFO) << "newest_timed_pose: " << newest_timed_pose.pose;
    // LOG(INFO) << "cached_extrapolated_pose_.time: "
    //           << cached_extrapolated_pose_.time << ", " << time;
    LOG(INFO) << "angular_velocity_from_poses_: "
              << common::RadToDeg(angular_velocity_from_poses_.x()) << "°, "
              << common::RadToDeg(angular_velocity_from_poses_.y()) << "°, "
              << common::RadToDeg(angular_velocity_from_poses_.z()) << "°";
    LOG(INFO) << "angular_velocity_from_odometry_: "
              << common::RadToDeg(angular_velocity_from_odometry_.x()) << "°, "
              << common::RadToDeg(angular_velocity_from_odometry_.y()) << "°, "
              << common::RadToDeg(angular_velocity_from_odometry_.z()) << "°";
    LOG(INFO) << "linear_velocity_from_odometry_: "
              << linear_velocity_from_odometry_.x() << ","
              << linear_velocity_from_odometry_.y() << ","
              << linear_velocity_from_odometry_.z();
    // LOG(INFO) << "odometry_data_.size(): " << odometry_data_.size();
    // if (cached_extrapolated_pose_.time != time) {
    auto delta_translation = ExtrapolateTranslation(time);
    auto delta_rotation =
        ExtrapolateRotation(time, extrapolation_imu_tracker_.get());
    LOG(INFO) << "translation: " << delta_translation.x() << ","
              << delta_translation.y() << "," << delta_translation.z();
    Eigen::Vector3d euler = transform::QuaternionToEulerAngles(delta_rotation);
    LOG(INFO) << "rotation: " << common::RadToDeg(euler.x()) << "°,"
              << common::RadToDeg(euler.y()) << "°,"
              << common::RadToDeg(euler.z()) << "°";
    const Eigen::Vector3d translation =
        delta_translation + newest_timed_pose.pose.translation();
    const Eigen::Quaterniond rotation =
        newest_timed_pose.pose.rotation() * delta_rotation;
    cached_extrapolated_pose_ =
        TimedPose{time, transform::Rigid3d{translation, rotation}};
  }
  return cached_extrapolated_pose_.pose;
}

Eigen::Quaterniond PoseExtrapolator::EstimateGravityOrientation(
    const common::Time time) {
#if 0
  ImuTracker imu_tracker = *imu_tracker_;
  AdvanceImuTracker(time, &imu_tracker);
  return imu_tracker.orientation();
#else
  if (odometry_data_.empty()) {
    return Eigen::Quaterniond::Identity();
  }
  return Eigen::Quaterniond::Identity();

  // auto matrix = odometry_data_.back().pose.rotation().toRotationMatrix();
  Eigen::Vector3d v = odometry_data_.back().pose.rotation().toRotationMatrix().eulerAngles(0, 1, 2);
  // LOG(INFO)<< "v: " <<  common::RadToDeg(v.x()) << "," << common::RadToDeg(v.y()) << "," << common::RadToDeg(v.z());
  const Eigen::AngleAxisd yaw_angle(-v.z(), Eigen::Vector3d::UnitZ());
  auto q = odometry_data_.back().pose.rotation() * yaw_angle;
  // Eigen::Vector3d v2 = q.toRotationMatrix().eulerAngles(0, 1, 2);
  // LOG(INFO)<< "v2: " <<  common::RadToDeg(v2.x()) << "," << common::RadToDeg(v2.y()) << "," << common::RadToDeg(v2.z());
  // LOG(INFO)<< "q: " << q.w() << "," << q.x() << "," << q.y() << "," << q.z();
  // auto m = q.toRotationMatrix();
  // LOG(INFO)<< "m: " << m(0, 0) << "," << m(0, 1) << "," << m(0, 2);
  // LOG(INFO)<< "m: " << m(1, 0) << "," << m(1, 1) << "," << m(1, 2);
  // LOG(INFO)<< "m: " << m(2, 0) << "," << m(2, 1) << "," << m(2, 2);
  //return q;
#endif
}

void PoseExtrapolator::UpdateVelocitiesFromPoses() {
  if (timed_pose_queue_.size() < 2) {
    // We need two poses to estimate velocities.
    return;
  }
  CHECK(!timed_pose_queue_.empty());
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const auto newest_time = newest_timed_pose.time;
  const TimedPose& oldest_timed_pose = timed_pose_queue_.front();
  const auto oldest_time = oldest_timed_pose.time;
  const double queue_delta = common::ToSeconds(newest_time - oldest_time);
  if (queue_delta < common::ToSeconds(pose_queue_duration_)) {
    LOG(WARNING) << "Queue too short for velocity estimation. Queue duration: "
                 << queue_delta << " s";
    return;
  }
  const transform::Rigid3d& newest_pose = newest_timed_pose.pose;
  const transform::Rigid3d& oldest_pose = oldest_timed_pose.pose;
  linear_velocity_from_poses_ =
      (newest_pose.translation() - oldest_pose.translation()) / queue_delta;
  angular_velocity_from_poses_ =
      transform::RotationQuaternionToAngleAxisVector(
          oldest_pose.rotation().inverse() * newest_pose.rotation()) /
      queue_delta;
}

void PoseExtrapolator::TrimImuData() {
  while (imu_data_.size() > 1 && !timed_pose_queue_.empty() &&
         imu_data_[1].time <= timed_pose_queue_.back().time) {
    imu_data_.pop_front();
  }
}

void PoseExtrapolator::TrimOdometryData() {
  while (odometry_data_.size() > 2 && common::ToSeconds(odometry_data_.back().time -
                          odometry_data_.front().time) > 10 && !timed_pose_queue_.empty() &&
         odometry_data_[1].time <= timed_pose_queue_.back().time) {
    odometry_data_.pop_front();
  }
}

void PoseExtrapolator::AdvanceImuTracker(const common::Time time,
                                         ImuTracker* const imu_tracker) const {
  CHECK_GE(time, imu_tracker->time());
  if (imu_data_.empty() || time < imu_data_.front().time) {
    // There is no IMU data until 'time', so we advance the ImuTracker and use
    // the angular velocities from poses and fake gravity to help 2D stability.
    imu_tracker->Advance(time);
    imu_tracker->AddImuLinearAccelerationObservation(Eigen::Vector3d::UnitZ());
    imu_tracker->AddImuAngularVelocityObservation(
        odometry_data_.size() < 2 ? angular_velocity_from_poses_
                                  : angular_velocity_from_odometry_);
    return;
  }
  if (imu_tracker->time() < imu_data_.front().time) {
    // Advance to the beginning of 'imu_data_'.
    imu_tracker->Advance(imu_data_.front().time);
  }
  auto it = std::lower_bound(
      imu_data_.begin(), imu_data_.end(), imu_tracker->time(),
      [](const sensor::ImuData& imu_data, const common::Time& time) {
        return imu_data.time < time;
      });
  while (it != imu_data_.end() && it->time < time) {
    imu_tracker->Advance(it->time);
    imu_tracker->AddImuLinearAccelerationObservation(it->linear_acceleration);
    imu_tracker->AddImuAngularVelocityObservation(it->angular_velocity);
    ++it;
  }
  imu_tracker->Advance(time);
}

Eigen::Quaterniond PoseExtrapolator::ExtrapolateRotation(
    const common::Time time, ImuTracker* const imu_tracker) const {
  if (time < imu_tracker->time()) {
    AdvanceImuTracker(imu_tracker->time(), imu_tracker);
    const Eigen::Quaterniond last_orientation = imu_tracker_->orientation();
    return last_orientation.inverse() * imu_tracker->orientation();
  }
  CHECK_GE(time, imu_tracker->time());
  AdvanceImuTracker(time, imu_tracker);
  const Eigen::Quaterniond last_orientation = imu_tracker_->orientation();
  return last_orientation.inverse() * imu_tracker->orientation();
}

Eigen::Vector3d PoseExtrapolator::ExtrapolateTranslation(common::Time time) {
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const double extrapolation_delta =
      common::ToSeconds(time - newest_timed_pose.time);
  if (extrapolation_delta > 0.5) {
    LOG(WARNING) << "extrapolation_delta: " << extrapolation_delta;
  }
  if (std::abs(linear_velocity_from_odometry_.x()) > 0.3) {
    LOG(WARNING) << "linear_velocity_from_odometry_x: "
                 << linear_velocity_from_odometry_.x();
    linear_velocity_from_odometry_.x() = linear_velocity_from_odometry_.x() > 0 ? 0.3 : -0.3;
  }
  if (std::abs(linear_velocity_from_odometry_.y()) > 0.3) {
    LOG(WARNING) << "linear_velocity_from_odometry_y: "
                 << linear_velocity_from_odometry_.y();
    linear_velocity_from_odometry_.y() = linear_velocity_from_odometry_.y() > 0 ? 0.3 : -0.3;
  }
  if (odometry_data_.size() < 2) {
    return extrapolation_delta * linear_velocity_from_poses_;
  }
  return extrapolation_delta * linear_velocity_from_odometry_;
}

PoseExtrapolator::ExtrapolationResult
PoseExtrapolator::ExtrapolatePosesWithGravity(
    const std::vector<common::Time>& times) {
  std::vector<transform::Rigid3f> poses;
  for (auto it = times.begin(); it != std::prev(times.end()); ++it) {
    poses.push_back(ExtrapolatePose(*it).cast<float>());
  }

  const Eigen::Vector3d current_velocity = odometry_data_.size() < 2
                                               ? linear_velocity_from_poses_
                                               : linear_velocity_from_odometry_;
  return ExtrapolationResult{poses, ExtrapolatePose(times.back()),
                             current_velocity,
                             EstimateGravityOrientation(times.back())};
}

}  // namespace mapping
}  // namespace cartographer
