/*
 * Copyright 2016 The Cartographer Authors
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

#include "cartographer/mapping/internal/2d/local_trajectory_builder_2d.h"

#include <limits>
#include <memory>

#include "absl/memory/memory.h"
#include "cartographer/metrics/family_factory.h"
#include "cartographer/sensor/range_data.h"

namespace cartographer {
namespace mapping {

static auto* kLocalSlamLatencyMetric = metrics::Gauge::Null();
static auto* kLocalSlamRealTimeRatio = metrics::Gauge::Null();
static auto* kLocalSlamCpuRealTimeRatio = metrics::Gauge::Null();
static auto* kRealTimeCorrelativeScanMatcherScoreMetric =
    metrics::Histogram::Null();
static auto* kCeresScanMatcherCostMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualDistanceMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualAngleMetric = metrics::Histogram::Null();

LocalTrajectoryBuilder2D::LocalTrajectoryBuilder2D(
    const proto::LocalTrajectoryBuilderOptions2D& options,
    const std::vector<std::string>& expected_range_sensor_ids)
    : options_(options),
      active_submaps_(options.submaps_options()),
      motion_filter_(options_.motion_filter_options()),
      real_time_correlative_scan_matcher_(
          options_.real_time_correlative_scan_matcher_options()),
      ceres_scan_matcher_(options_.ceres_scan_matcher_options()),
      in_location_inserter_(options_.in_location_inserter()),
      range_data_collator_(expected_range_sensor_ids) {
        auto opt = options_.real_time_correlative_scan_matcher_options();
        double old_angular_search_window = opt.angular_search_window();
        opt.set_angular_search_window(old_angular_search_window * 2);
        opt.set_linear_search_window(opt.linear_search_window() * 2);
        real_time_rotation_rescan_matcher_ = std::make_shared<scan_matching::RealTimeCorrelativeScanMatcher2D>(opt);

        opt.set_linear_search_window(opt.linear_search_window() * 4);
        opt.set_angular_search_window(old_angular_search_window);
        real_time_translation_rescan_matcher_ = std::make_shared<scan_matching::RealTimeCorrelativeScanMatcher2D>(opt);
      }

LocalTrajectoryBuilder2D::~LocalTrajectoryBuilder2D() {}

sensor::RangeData
LocalTrajectoryBuilder2D::TransformToGravityAlignedFrameAndFilter(
    const transform::Rigid3f& transform_to_gravity_aligned_frame,
    const sensor::RangeData& range_data) const {
  const sensor::RangeData cropped =
      sensor::CropRangeData(sensor::TransformRangeData(
                                range_data, transform_to_gravity_aligned_frame),
                            options_.min_z(), options_.max_z());
  return sensor::RangeData{
      cropped.origin,
      sensor::VoxelFilter(cropped.returns, options_.voxel_filter_size()),
      sensor::VoxelFilter(cropped.misses, options_.voxel_filter_size())};
}

std::unique_ptr<transform::Rigid2d> LocalTrajectoryBuilder2D::ScanMatch(
    const common::Time time, const transform::Rigid2d& pose_prediction,
    const sensor::PointCloud& filtered_gravity_aligned_point_cloud,
    double& match_score) {
  if (active_submaps_.submaps().empty()) {
    return absl::make_unique<transform::Rigid2d>(pose_prediction);
  }
  std::shared_ptr<const Submap2D> matching_submap =
      active_submaps_.submaps().front();
  // The online correlative scan matcher will refine the initial estimate for
  // the Ceres scan matcher.
  transform::Rigid2d initial_ceres_pose = pose_prediction;
  transform::Rigid2d pose_prediction2 = pose_prediction;

  if (options_.use_online_correlative_scan_matching()) {
    const double score = real_time_correlative_scan_matcher_.Match(
        pose_prediction, filtered_gravity_aligned_point_cloud,
        *matching_submap->grid(), &initial_ceres_pose);
    kRealTimeCorrelativeScanMatcherScoreMetric->Observe(score);
    LOG(INFO) << "online_correlative_scan_matching score :" << score;
    match_score = score;
    if(score < 0.5 && matching_submap->num_range_data() > 10 && filtered_gravity_aligned_point_cloud.size() > 100)
    {
      LOG(WARNING) << "pose_prediction: " << pose_prediction << " initial_ceres_pose: " << initial_ceres_pose;
      transform::Rigid2d rotate_remach_pose = initial_ceres_pose;
      double rotation_rematch_score = real_time_rotation_rescan_matcher_->Match(
        initial_ceres_pose, filtered_gravity_aligned_point_cloud,
        *matching_submap->grid(), &rotate_remach_pose);

      // transform::Rigid2d translation_rematch_pose = initial_ceres_pose;
      // double translation_rematch_score = real_time_translation_rescan_matcher_->Match(
      //   initial_ceres_pose, filtered_gravity_aligned_point_cloud,
      //   *matching_submap->grid(), &translation_rematch_pose);
      if(rotation_rematch_score > 0.7)
      {
          match_score = rotation_rematch_score;
          initial_ceres_pose = rotate_remach_pose;
          LOG(WARNING) << "pose_prediction: " << pose_prediction << " rotate_remach_pose: " << rotate_remach_pose;
          pose_prediction2 = rotate_remach_pose;
      }
      LOG(WARNING) << "online_correlative_scan_matching score low :" << score << " rematch score:" << rotation_rematch_score;
    }
  }

  auto pose_observation = absl::make_unique<transform::Rigid2d>();
  ceres::Solver::Summary summary;
  ceres_scan_matcher_.Match(pose_prediction2.translation(), initial_ceres_pose,
                            filtered_gravity_aligned_point_cloud,
                            *matching_submap->grid(), pose_observation.get(),
                            &summary);
  if (pose_observation) {
    kCeresScanMatcherCostMetric->Observe(summary.final_cost);
    const double residual_distance =
        (pose_observation->translation() - pose_prediction.translation())
            .norm();
    kScanMatcherResidualDistanceMetric->Observe(residual_distance);
    const double residual_angle =
        std::abs(pose_observation->rotation().angle() -
                 pose_prediction.rotation().angle());
    kScanMatcherResidualAngleMetric->Observe(residual_angle);

    LOG(INFO) << "initial_ceres_pose: " << initial_ceres_pose << " pose_observation: " << *pose_observation;
  }
  return pose_observation;
}

std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddRangeData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& unsynchronized_data) {
  auto synchronized_data =
      range_data_collator_.AddRangeData(sensor_id, unsynchronized_data);
  if (synchronized_data.ranges.empty()) {
    LOG(INFO) << "Range data collator filling buffer.";
    return nullptr;
  }

  const common::Time& time = synchronized_data.time;
  // Initialize extrapolator now if we do not ever use an IMU.
  if (!options_.use_imu_data()) {
    InitializeExtrapolator(time);
  }

  if (extrapolator_ == nullptr) {
    // Until we've initialized the extrapolator with our first IMU message, we
    // cannot compute the orientation of the rangefinder.
    LOG(INFO) << "Extrapolator not yet initialized.";
    return nullptr;
  }

  CHECK(!synchronized_data.ranges.empty());
  // TODO(gaschler): Check if this can strictly be 0.
  CHECK_LE(synchronized_data.ranges.back().point_time.time, 0.f);
  const common::Time time_first_point =
      time +
      common::FromSeconds(synchronized_data.ranges.front().point_time.time);
  if (time_first_point < extrapolator_->GetLastPoseTime()) {
    LOG(INFO) << "Extrapolator is still initializing." << time_first_point << " < " << extrapolator_->GetLastPoseTime();
    return nullptr;
  }

  std::vector<transform::Rigid3f> range_data_poses;
  range_data_poses.reserve(synchronized_data.ranges.size());
  bool warned = false;
  // 对雷达点做时间矫正
  for (const auto& range : synchronized_data.ranges) {
    common::Time time_point = time + common::FromSeconds(range.point_time.time);
    if (time_point < extrapolator_->GetLastExtrapolatedTime()) {
      if (!warned) {
        LOG(ERROR)
            << "Timestamp of individual range data point jumps backwards from "
            << extrapolator_->GetLastExtrapolatedTime() << " to " << time_point;
        warned = true;
      }
      time_point = extrapolator_->GetLastExtrapolatedTime();
    }
    range_data_poses.push_back(
        extrapolator_->ExtrapolatePose(time_point).cast<float>());
  }

  if (num_accumulated_ == 0) {
    // 'accumulated_range_data_.origin' is uninitialized until the last
    // accumulation.
    accumulated_range_data_ = sensor::RangeData{{}, {}, {}};
  }

  // Drop any returns below the minimum range and convert returns beyond the
  // maximum range into misses.
  for (size_t i = 0; i < synchronized_data.ranges.size(); ++i) {
    const sensor::TimedRangefinderPoint& hit =
        synchronized_data.ranges[i].point_time;
    const Eigen::Vector3f origin_in_local =
        range_data_poses[i] *
        synchronized_data.origins.at(synchronized_data.ranges[i].origin_index);
    sensor::RangefinderPoint hit_in_local =
        range_data_poses[i] * sensor::ToRangefinderPoint(hit);
    const Eigen::Vector3f delta = hit_in_local.position - origin_in_local;
    const float range = delta.norm();
    if (range >= options_.min_range()) {
      if (range <= options_.max_range()) {
        accumulated_range_data_.returns.push_back(hit_in_local);
      } else {
        hit_in_local.position =
            origin_in_local +
            options_.missing_data_ray_length() / range * delta;
        accumulated_range_data_.misses.push_back(hit_in_local);
      }
    }
  }
  ++num_accumulated_;
  // 积攒了 num_accumulated_ 数据才建图
  if (num_accumulated_ >= options_.num_accumulated_range_data()) {
    const common::Time current_sensor_time = synchronized_data.time;
    absl::optional<common::Duration> sensor_duration;
    if (last_sensor_time_.has_value()) {
      sensor_duration = current_sensor_time - last_sensor_time_.value();
    }
    last_sensor_time_ = current_sensor_time;
    num_accumulated_ = 0;
    const transform::Rigid3d gravity_alignment = transform::Rigid3d::Rotation(
        extrapolator_->EstimateGravityOrientation(time));
    // TODO(gaschler): This assumes that 'range_data_poses.back()' is at time
    // 'time'.
    accumulated_range_data_.origin = range_data_poses.back().translation();
    LOG(INFO)<<"accumulated_range_data_.returns.size():"<<accumulated_range_data_.returns.size();
    return AddAccumulatedRangeData(
        time,
        unsynchronized_data.fit_angle,
        TransformToGravityAlignedFrameAndFilter(
            gravity_alignment.cast<float>() * range_data_poses.back().inverse(),
            accumulated_range_data_),
        gravity_alignment, sensor_duration);
  }
  return nullptr;
}

std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddAccumulatedRangeData(
    const common::Time time,
    const Eigen::Vector3f & fit_vector,
    const sensor::RangeData& gravity_aligned_range_data,
    const transform::Rigid3d& gravity_alignment,
    const absl::optional<common::Duration>& sensor_duration) {
  if (gravity_aligned_range_data.returns.empty()) {
    LOG(WARNING) << "Dropped empty horizontal range data.";
    return nullptr;
  }

  LOG(INFO)<< "gravity_aligned_range_data.returns.size():" << gravity_aligned_range_data.returns.size();
  if(gravity_aligned_range_data.returns.size() < 25)
  {
    LOG(WARNING) << "gravity_aligned_range_data.returns.size():" << gravity_aligned_range_data.returns.size();
    return nullptr;
  }

  auto time_start = std::chrono::steady_clock::now();
  // Computes a gravity aligned pose prediction.
  transform::Rigid3d non_gravity_aligned_pose_prediction =
      extrapolator_->ExtrapolatePoseLog(time);

  LOG(INFO) << "ExtrapolatePoseLog: " << non_gravity_aligned_pose_prediction;
  
  transform::Rigid2d pose_prediction = transform::Project2D(
      non_gravity_aligned_pose_prediction * gravity_alignment.inverse());

  const sensor::PointCloud& filtered_gravity_aligned_point_cloud =
      sensor::AdaptiveVoxelFilter(gravity_aligned_range_data.returns,
                                  options_.adaptive_voxel_filter_options());
  if (filtered_gravity_aligned_point_cloud.empty()) {
    return nullptr;
  }

  // LOG(INFO) << "ScanMatch Pose in: " << pose_prediction << "gravity_alignment: " << gravity_alignment; 
  // local map frame <- gravity-aligned frame

  if(fit_vector.z() != 0 && std::abs(extrapolator_->AngularVelocityFromOdometry().z()) < common::DegToRad(30))
  {
    const double errDeg = 5.0;
    double a0 = common::NormalizeAngleDifference(-fit_vector.z());
    double a1 = common::NormalizeAngleDifference(a0 + M_PI_2);
    double a2 = common::NormalizeAngleDifference(a0 + M_PI);
    double a3 = common::NormalizeAngleDifference(a0 + M_PI + M_PI_2);
    LOG(INFO) << "robot yaw maybe: " << common::RadToDeg(a0) << "°, " << common::RadToDeg(a1) << "°, " << common::RadToDeg(a2) << "°, " << common::RadToDeg(a3) << "°";
    Eigen::Vector3d euler = non_gravity_aligned_pose_prediction.rotation().toRotationMatrix().eulerAngles(0, 1, 2);
    LOG(INFO) << "euler: " << common::RadToDeg(euler.x()) << "°," << common::RadToDeg(euler.y()) << "°," << common::RadToDeg(euler.z()) << "°";
    LOG(INFO) << "robot match yaw: " << common::RadToDeg(pose_prediction.rotation().angle()) << "°, " << common::RadToDeg(euler.z()) << "°";
    if(std::abs(common::NormalizeAngleDifference(a0-euler.z())) < common::DegToRad(errDeg))
    {
      LOG(INFO)<< "robot a0: " << common::RadToDeg(a0) << "°";
      const Eigen::AngleAxisd yaw_angle(a0-euler.z(), Eigen::Vector3d::UnitZ());
      non_gravity_aligned_pose_prediction = transform::Rigid3d(non_gravity_aligned_pose_prediction.translation(), non_gravity_aligned_pose_prediction.rotation() * yaw_angle);
    }
    else if(std::abs(common::NormalizeAngleDifference(a1-euler.z())) < common::DegToRad(errDeg))
    {
      LOG(INFO)<< "robot a1: " << common::RadToDeg(a1) << "°";
      const Eigen::AngleAxisd yaw_angle(a1-euler.z(), Eigen::Vector3d::UnitZ());
      non_gravity_aligned_pose_prediction = transform::Rigid3d(non_gravity_aligned_pose_prediction.translation(), non_gravity_aligned_pose_prediction.rotation() * yaw_angle);
    }
    else if(std::abs(common::NormalizeAngleDifference(a2-euler.z())) < common::DegToRad(errDeg))
    {
      LOG(INFO)<< "robot a2: " << common::RadToDeg(a2) << "°";
      const Eigen::AngleAxisd yaw_angle(a2-euler.z(), Eigen::Vector3d::UnitZ());
      non_gravity_aligned_pose_prediction = transform::Rigid3d(non_gravity_aligned_pose_prediction.translation(), non_gravity_aligned_pose_prediction.rotation() * yaw_angle);
    }
    else if(std::abs(common::NormalizeAngleDifference(a3-euler.z())) < common::DegToRad(errDeg))
    {
      LOG(INFO)<< "robot a3: " << common::RadToDeg(a3) << "°";
      const Eigen::AngleAxisd yaw_angle(a3-euler.z(), Eigen::Vector3d::UnitZ());
      non_gravity_aligned_pose_prediction = transform::Rigid3d(non_gravity_aligned_pose_prediction.translation(), non_gravity_aligned_pose_prediction.rotation() * yaw_angle);
      // non_gravity_aligned_pose_prediction = transform::Rigid3d(non_gravity_aligned_pose_prediction.translation(), transform::RollPitchYaw(euler.x(), euler.y(), a3));
    }
    euler = non_gravity_aligned_pose_prediction.rotation().toRotationMatrix().eulerAngles(0, 1, 2);
    pose_prediction = transform::Project2D(
      non_gravity_aligned_pose_prediction * gravity_alignment.inverse());  
    LOG(INFO) << "new euler: " << common::RadToDeg(euler.x()) << "°," << common::RadToDeg(euler.y()) << "°," << common::RadToDeg(euler.z()) << "°";
  }  
  
  double score = 0;
  std::unique_ptr<transform::Rigid2d> pose_estimate_2d = ScanMatch(
      time, pose_prediction, filtered_gravity_aligned_point_cloud, score);
  if (pose_estimate_2d == nullptr) {
    LOG(WARNING) << "Scan matching failed.";
    return nullptr;
  }

  transform::Rigid3d pose_estimate =
      transform::Embed3D(*pose_estimate_2d) * gravity_alignment;
  
  if(fit_vector.z() != 0 && std::abs(extrapolator_->AngularVelocityFromOdometry().z()) < common::DegToRad(30))
  {
    const double errDeg = 5.0;
    double a0 = common::NormalizeAngleDifference(-fit_vector.z());
    double a1 = common::NormalizeAngleDifference(a0 + M_PI_2);
    double a2 = common::NormalizeAngleDifference(a0 + M_PI);
    double a3 = common::NormalizeAngleDifference(a0 + M_PI + M_PI_2);
    Eigen::Vector3d euler = pose_estimate.rotation().toRotationMatrix().eulerAngles(0, 1, 2);
    if(std::abs(common::NormalizeAngleDifference(a0-euler.z())) < common::DegToRad(errDeg))
    {
      pose_estimate = transform::Rigid3d(pose_estimate.translation(), transform::RollPitchYaw(euler.x(), euler.y(), a0));
    }
    else if(std::abs(common::NormalizeAngleDifference(a1-euler.z())) < common::DegToRad(errDeg))
    {
      pose_estimate = transform::Rigid3d(pose_estimate.translation(), transform::RollPitchYaw(euler.x(), euler.y(), a1));
    }
    else if(std::abs(common::NormalizeAngleDifference(a2-euler.z())) < common::DegToRad(errDeg))
    {
      pose_estimate = transform::Rigid3d(pose_estimate.translation(), transform::RollPitchYaw(euler.x(), euler.y(), a2));
    }
    else if(std::abs(common::NormalizeAngleDifference(a3-euler.z())) < common::DegToRad(errDeg))
    {
      pose_estimate = transform::Rigid3d(pose_estimate.translation(), transform::RollPitchYaw(euler.x(), euler.y(), a3));
    }
    euler = pose_estimate.rotation().toRotationMatrix().eulerAngles(0, 1, 2);
    pose_estimate_2d->Rotation(euler.z());  // todo
    LOG(INFO) << "pose_estimate_2d: " << common::RadToDeg(pose_estimate_2d->rotation().angle()) << "°";
  }
  else{
    LOG(INFO) << "No fit line --------------------------";
  }

  if (score > 0) {
     //LOG(INFO) << "ScanMatch Pose in: " << pose_prediction  << "("<<pose_prediction.normalized_angle() * 180 / 3.1415 <<")"<<"Score: " << score;
  }

  sensor::RangeData range_data_in_local =
      TransformRangeData(gravity_aligned_range_data,
                         transform::Embed3D(pose_estimate_2d->cast<float>()));

  std::unique_ptr<InsertionResult> insertion_result = nullptr;
  if (score < 0.5)
  {
    if(!active_submaps_.submaps().empty() && active_submaps_.submaps().front()->num_range_data() > 10)
    {
      LOG(WARNING) << "ScanMatch Score Low: " << score << " use prediction pose";
       extrapolator_->AddPose(time, transform::Embed3D(pose_prediction) * gravity_alignment);
       return nullptr;
    }
    else
    {
      if(active_submaps_.IsWrongFrame(range_data_in_local))
      {
        LOG(ERROR)<< "It is wrong frame, donnot insert!!! Score: " << score;
      }
      else if(std::abs(extrapolator_->AngularVelocityFromOdometry().z()) > common::DegToRad(45))
      {
        LOG(WARNING)<< "Donnot insert, angular velocity is high: "<<common::RadToDeg(extrapolator_->AngularVelocityFromOdometry().z());
      }
      else
      {
        insertion_result = InsertIntoSubmap(
          time, range_data_in_local, filtered_gravity_aligned_point_cloud,
          pose_estimate, gravity_alignment.rotation());
      }
    }
  }
  else if (score < in_location_inserter_.insert_point_threshold()) 
  {
      // LOG(INFO) << "ScanMatch Score: " << score << " insert threshold: " << in_location_inserter_.insert_point_threshold();
      if(active_submaps_.IsWrongFrame(range_data_in_local))
      {
        LOG(WARNING)<< "It is wrong frame, donnot insert! Score: " << score;
      }
      //else
      if(std::abs(extrapolator_->AngularVelocityFromOdometry().z()) > common::DegToRad(45))
      {
        LOG(WARNING)<< "Donnot insert, angular velocity is high: "<<common::RadToDeg(extrapolator_->AngularVelocityFromOdometry().z());
      }
      else{
        insertion_result = InsertIntoSubmap(
          time, range_data_in_local, filtered_gravity_aligned_point_cloud,
          pose_estimate, gravity_alignment.rotation());
      }
  }
  else if(score < in_location_inserter_.donnot_insert_threshold())
  {
      // LOG(INFO) << "ScanMatch Score: " << score << " donnot_insert_threshold: " << in_location_inserter_.donnot_insert_threshold();
      if(extrapolator_->AngularVelocityFromOdometry().z() > common::DegToRad(45))
      {
        LOG(WARNING)<< "Donnot insert, angular velocity is high: "<<common::RadToDeg(extrapolator_->AngularVelocityFromOdometry().z());
      }
      else{
        insertion_result = InsertIntoSubmap(
          time, range_data_in_local, filtered_gravity_aligned_point_cloud,
          pose_estimate, gravity_alignment.rotation(), in_location_inserter_);
      }
  }

  extrapolator_->AddPose(time, pose_estimate);
  
  const auto wall_time = std::chrono::steady_clock::now();
  if (last_wall_time_.has_value()) {
    const auto wall_time_duration = wall_time - last_wall_time_.value();
    kLocalSlamLatencyMetric->Set(common::ToSeconds(wall_time_duration));
    if (sensor_duration.has_value()) {
      kLocalSlamRealTimeRatio->Set(common::ToSeconds(sensor_duration.value()) /
                                   common::ToSeconds(wall_time_duration));
    }
  }
  const double thread_cpu_time_seconds = common::GetThreadCpuTimeSeconds();
  if (last_thread_cpu_time_seconds_.has_value()) {
    const double thread_cpu_duration_seconds =
        thread_cpu_time_seconds - last_thread_cpu_time_seconds_.value();
    if (sensor_duration.has_value()) {
      kLocalSlamCpuRealTimeRatio->Set(
          common::ToSeconds(sensor_duration.value()) /
          thread_cpu_duration_seconds);
    }
  }
  last_wall_time_ = wall_time;
  last_thread_cpu_time_seconds_ = thread_cpu_time_seconds;

  if(common::ToSeconds(std::chrono::steady_clock::now() - time_start) > 0.1)
  {
    LOG(WARNING) << "ScanMatch wall_time_duration: " << common::ToSeconds(std::chrono::steady_clock::now() - time_start);
  }

  return absl::make_unique<MatchingResult>(
      MatchingResult{time, pose_estimate, std::move(range_data_in_local),
                     std::move(insertion_result)});
}

std::unique_ptr<LocalTrajectoryBuilder2D::InsertionResult>
LocalTrajectoryBuilder2D::InsertIntoSubmap(
    const common::Time time, const sensor::RangeData& range_data_in_local,
    const sensor::PointCloud& filtered_gravity_aligned_point_cloud,
    const transform::Rigid3d& pose_estimate,
    const Eigen::Quaterniond& gravity_alignment) {
  std::vector<std::shared_ptr<const Submap2D>> insertion_submaps =
      active_submaps_.InsertRangeData(range_data_in_local);
  return absl::make_unique<InsertionResult>(InsertionResult{
      std::make_shared<const TrajectoryNode::Data>(TrajectoryNode::Data{
          time,
          gravity_alignment,
          filtered_gravity_aligned_point_cloud,
          {},  // 'high_resolution_point_cloud' is only used in 3D.
          {},  // 'low_resolution_point_cloud' is only used in 3D.
          {},  // 'rotational_scan_matcher_histogram' is only used in 3D.
          pose_estimate}),
      std::move(insertion_submaps)});
}

std::unique_ptr<LocalTrajectoryBuilder2D::InsertionResult>
LocalTrajectoryBuilder2D::InsertIntoSubmap(
    const common::Time time, const sensor::RangeData& range_data_in_local,
    const sensor::PointCloud& filtered_gravity_aligned_point_cloud,
    const transform::Rigid3d& pose_estimate,
    const Eigen::Quaterniond& gravity_alignment,
    const proto::InLocationInserterOptions& in_location_inserter) {
  std::vector<std::shared_ptr<const Submap2D>> insertion_submaps =
      active_submaps_.InsertRangeData(range_data_in_local, in_location_inserter);
  return absl::make_unique<InsertionResult>(InsertionResult{
      std::make_shared<const TrajectoryNode::Data>(TrajectoryNode::Data{
          time,
          gravity_alignment,
          filtered_gravity_aligned_point_cloud,
          {},  // 'high_resolution_point_cloud' is only used in 3D.
          {},  // 'low_resolution_point_cloud' is only used in 3D.
          {},  // 'rotational_scan_matcher_histogram' is only used in 3D.
          pose_estimate}),
      std::move(insertion_submaps)});
}

void LocalTrajectoryBuilder2D::AddImuData(const sensor::ImuData& imu_data) {
  CHECK(options_.use_imu_data()) << "An unexpected IMU packet was added.";
  InitializeExtrapolator(imu_data.time);
  extrapolator_->AddImuData(imu_data);
}

void LocalTrajectoryBuilder2D::AddOdometryData(
    const sensor::OdometryData& odometry_data) {
  if (extrapolator_ == nullptr) {
    // Until we've initialized the extrapolator we cannot add odometry data.
    LOG(INFO) << "Extrapolator not yet initialized.";
    return;
  }
  extrapolator_->AddOdometryData(odometry_data);
}

void LocalTrajectoryBuilder2D::SetGlobalInitialPose(
    transform::Rigid3d& initial_pose) {
  this->initial_pose = initial_pose;
}

void LocalTrajectoryBuilder2D::ResetExtrapolator(
    common::Time time, transform::Rigid3d& pose_estimate) {
  LOG(INFO) << "1-----------------------------";
  if (extrapolator_ != nullptr) {
    extrapolator_.reset();
  }
  LOG(INFO) << "2-----------------------------";

  extrapolator_ = absl::make_unique<PoseExtrapolator>(
      ::cartographer::common::FromSeconds(options_.pose_extrapolator_options()
                                              .constant_velocity()
                                              .pose_queue_duration()),
      options_.pose_extrapolator_options()
          .constant_velocity()
          .imu_gravity_time_constant());
  LOG(INFO) << "3-----------------------------";

  extrapolator_->AddPose(time, pose_estimate);
  LOG(INFO) << "4-----------------------------";

}

void LocalTrajectoryBuilder2D::RebuildActiveSubmap(std::shared_ptr<const Submap> submap)
{
  active_submaps_.RebuildSubmap(std::dynamic_pointer_cast<const Submap2D>(submap));
}

bool LocalTrajectoryBuilder2D::IsWrongFrame(transform::Rigid2d pose_estimate_2d, const sensor::PointCloud & frame)
{
  sensor::RangeData range_data;
  for (const auto & p : frame)
  {
    auto range = p.position.norm();
    if(range >= options_.min_range() && range <= options_.max_range())
    {
      range_data.returns.push_back(p);
    }
  }
  range_data.origin = Eigen::Vector3f(0, 0, 0);
  sensor::RangeData range_data_in_local =
      TransformRangeData(range_data,
                         transform::Embed3D(pose_estimate_2d.cast<float>()));
  // LOG(INFO)<<"1 range_data_in_local.origin: "<<range_data_in_local.origin;   
  return active_submaps_.IsWrongFrame(range_data_in_local);
}

void LocalTrajectoryBuilder2D::InitializeExtrapolator(const common::Time time) {
  if (extrapolator_ != nullptr) {
    return;
  }
  CHECK(!options_.pose_extrapolator_options().use_imu_based());
  // TODO(gaschler): Consider using InitializeWithImu as 3D does.
  extrapolator_ = absl::make_unique<PoseExtrapolator>(
      ::cartographer::common::FromSeconds(options_.pose_extrapolator_options()
                                              .constant_velocity()
                                              .pose_queue_duration()),
      options_.pose_extrapolator_options()
          .constant_velocity()
          .imu_gravity_time_constant());
  // extrapolator_->AddPose(time, transform::Rigid3d::Identity());
  extrapolator_->AddPose(time, initial_pose);
}

void LocalTrajectoryBuilder2D::RegisterMetrics(
    metrics::FamilyFactory* family_factory) {
  auto* latency = family_factory->NewGaugeFamily(
      "mapping_2d_local_trajectory_builder_latency",
      "Duration from first incoming point cloud in accumulation to local slam "
      "result");
  kLocalSlamLatencyMetric = latency->Add({});
  auto* real_time_ratio = family_factory->NewGaugeFamily(
      "mapping_2d_local_trajectory_builder_real_time_ratio",
      "sensor duration / wall clock duration.");
  kLocalSlamRealTimeRatio = real_time_ratio->Add({});

  auto* cpu_real_time_ratio = family_factory->NewGaugeFamily(
      "mapping_2d_local_trajectory_builder_cpu_real_time_ratio",
      "sensor duration / cpu duration.");
  kLocalSlamCpuRealTimeRatio = cpu_real_time_ratio->Add({});
  auto score_boundaries = metrics::Histogram::FixedWidth(0.05, 20);
  auto* scores = family_factory->NewHistogramFamily(
      "mapping_2d_local_trajectory_builder_scores", "Local scan matcher scores",
      score_boundaries);
  kRealTimeCorrelativeScanMatcherScoreMetric =
      scores->Add({{"scan_matcher", "real_time_correlative"}});
  auto cost_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 100);
  auto* costs = family_factory->NewHistogramFamily(
      "mapping_2d_local_trajectory_builder_costs", "Local scan matcher costs",
      cost_boundaries);
  kCeresScanMatcherCostMetric = costs->Add({{"scan_matcher", "ceres"}});
  auto distance_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 10);
  auto* residuals = family_factory->NewHistogramFamily(
      "mapping_2d_local_trajectory_builder_residuals",
      "Local scan matcher residuals", distance_boundaries);
  kScanMatcherResidualDistanceMetric =
      residuals->Add({{"component", "distance"}});
  kScanMatcherResidualAngleMetric = residuals->Add({{"component", "angle"}});
}

}  // namespace mapping
}  // namespace cartographer
