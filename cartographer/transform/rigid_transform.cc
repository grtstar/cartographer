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

#include "cartographer/transform/rigid_transform.h"

#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "glog/logging.h"

namespace cartographer {
namespace transform {

namespace {

Eigen::Vector3d TranslationFromDictionary(
    common::LuaParameterDictionary* dictionary) {
  const std::vector<double> translation = dictionary->GetArrayValuesAsDoubles();
  CHECK_EQ(3, translation.size()) << "Need (x, y, z) for translation.";
  return Eigen::Vector3d(translation[0], translation[1], translation[2]);
}

}  // namespace

Eigen::Vector3d QuaternionToEulerAngles(const Eigen::Quaterniond& q) {
  double r, p, y;

  double tol = static_cast<double>(1e-15);
  double squ;
  double sqx;
  double sqy;
  double sqz;
  squ = q.w() * q.w();
  sqx = q.x() * q.x();
  sqy = q.y() * q.y();
  sqz = q.z() * q.z();

  // Pitch
  double sarg = -2 * (q.x() * q.z() - q.w() * q.y());
  if (sarg <= double(-1.0)) {
    p = (double(-0.5 * M_PI));
  } else if (sarg >= double(1.0)) {
    p = (double(0.5 * M_PI));
  } else {
    p = (double(asin(sarg)));
  }

  // If the pitch angle is PI/2 or -PI/2, we can only compute
  // the sum roll + yaw.  However, any combination that gives
  // the right sum will produce the correct orientation, so we
  // set yaw = 0 and compute roll.
  // pitch angle is PI/2
  if (std::abs(sarg - 1) < tol) {
    y = (0);
    r = (double(
        atan2(2 * (q.x() * q.y() - q.z() * q.w()), squ - sqx + sqy - sqz)));
  }
  // pitch angle is -PI/2
  else if (std::abs(sarg + 1) < tol) {
    y = (0);
    r = (double(
        atan2(-2 * (q.x() * q.y() - q.z() * q.w()), squ - sqx + sqy - sqz)));
  } else {
    // Roll
    r = (double(
        atan2(2 * (q.y() * q.z() + q.w() * q.x()), squ - sqx - sqy + sqz)));

    // Yaw
    y = (double(
        atan2(2 * (q.x() * q.y() + q.w() * q.z()), squ + sqx - sqy - sqz)));
  }

  return Eigen::Vector3d(r, p, y);
}

Eigen::Quaterniond RollPitchYaw(const double roll, const double pitch,
                                const double yaw) {
  const Eigen::AngleAxisd roll_angle(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch_angle(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw_angle(yaw, Eigen::Vector3d::UnitZ());
  return yaw_angle * pitch_angle * roll_angle;
}

transform::Rigid3d FromDictionary(common::LuaParameterDictionary* dictionary) {
  const Eigen::Vector3d translation =
      TranslationFromDictionary(dictionary->GetDictionary("translation").get());

  auto rotation_dictionary = dictionary->GetDictionary("rotation");
  if (rotation_dictionary->HasKey("w")) {
    const Eigen::Quaterniond rotation(rotation_dictionary->GetDouble("w"),
                                      rotation_dictionary->GetDouble("x"),
                                      rotation_dictionary->GetDouble("y"),
                                      rotation_dictionary->GetDouble("z"));
    CHECK_NEAR(rotation.norm(), 1., 1e-9);
    return transform::Rigid3d(translation, rotation);
  } else {
    const std::vector<double> rotation =
        rotation_dictionary->GetArrayValuesAsDoubles();
    CHECK_EQ(3, rotation.size()) << "Need (roll, pitch, yaw) for rotation.";
    return transform::Rigid3d(
        translation, RollPitchYaw(rotation[0], rotation[1], rotation[2]));
  }
}

}  // namespace transform
}  // namespace cartographer
