

#ifndef CARTOGRAPHER_MAPPING_INTERNAL_IN_LOCATION_INSERTER_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_IN_LOCATION_INSERTER_H_

#include <limits>

#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/proto/in_location_inserter.pb.h"
#include "cartographer/transform/rigid_transform.h"

namespace cartographer {
namespace mapping {

proto::InLocationInserterOptions CreateInLocaltionInsertOptions(
    common::LuaParameterDictionary* parameter_dictionary)
    {
        proto::InLocationInserterOptions options;
        options.set_donnot_insert_threshold(
            parameter_dictionary->GetDouble("donnot_insert_threshold"));
        options.set_insert_point_threshold(
            parameter_dictionary->GetDouble("insert_point_threshold"));
        options.set_max_hit_length(
            parameter_dictionary->GetDouble("max_hit_length"));
        options.set_max_miss_length(
            parameter_dictionary->GetDouble("max_miss_length"));
        options.set_hit_tolerance_grid(
            parameter_dictionary->GetInt("hit_tolerance_grid"));
        options.set_hit_probability(
            parameter_dictionary->GetDouble("hit_probability"));
        return options;
    }
}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_MOTION_FILTER_H_
