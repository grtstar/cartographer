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

#include "cartographer/sensor/internal/ordered_multi_queue.h"

#include <algorithm>
#include <sstream>
#include <vector>

#include "absl/memory/memory.h"
#include "glog/logging.h"

namespace cartographer {
namespace sensor {

namespace {

// Number of items that can be queued up before we log which queues are waiting
// for data.
const int kMaxQueueSize = 1000;

}  // namespace

inline std::ostream& operator<<(std::ostream& out, const QueueKey& key) {
  return out << '(' << key.trajectory_id << ", " << key.sensor_id << ')';
}

OrderedMultiQueue::OrderedMultiQueue() {}

OrderedMultiQueue::~OrderedMultiQueue() {
  for (auto& entry : queues_) {
    CHECK(entry.second.finished);
  }
}

void OrderedMultiQueue::AddQueue(const QueueKey& queue_key, Callback callback) {
  CHECK_EQ(queues_.count(queue_key), 0);
  queues_[queue_key].callback = std::move(callback);
}

void OrderedMultiQueue::MarkQueueAsFinished(const QueueKey& queue_key) {
  auto it = queues_.find(queue_key);
  CHECK(it != queues_.end()) << "Did not find '" << queue_key << "'.";
  auto& queue = it->second;
  CHECK(!queue.finished);
  queue.finished = true;
  Dispatch();
}

void OrderedMultiQueue::Add(const QueueKey& queue_key,
                            std::unique_ptr<Data> data) {
  auto it = queues_.find(queue_key);
  if (it == queues_.end()) {
    LOG_EVERY_N(WARNING, 1000)
        << "Ignored data for queue: '" << queue_key << "'";
    return;
  }
  const auto* last_data = it->second.queue.Peek<Data>();
  if(last_data == nullptr || data->GetTime() > last_data->GetTime())
  {
    // std::string sensor_id = data->GetSensorId();
    // if(sensor_id == "range0")
    // {
    //   LOG(INFO) << "Dispatch at " << data->GetTime() << " for " << queue_key;
    // }
    it->second.queue.Push(std::move(data));
    Dispatch();
    // if(sensor_id == "range0")
    // {
    //   LOG(INFO) << "Dispatch end for " << queue_key;
    // }
  }
}

void OrderedMultiQueue::Flush() {
  std::vector<QueueKey> unfinished_queues;
  for (auto& entry : queues_) {
    if (!entry.second.finished) {
      unfinished_queues.push_back(entry.first);
    }
  }
  for (auto& unfinished_queue : unfinished_queues) {
    MarkQueueAsFinished(unfinished_queue);
  }
}

QueueKey OrderedMultiQueue::GetBlocker() const {
  CHECK(!queues_.empty());
  return blocker_;
}

void OrderedMultiQueue::Dispatch() {
  while (true) {
    const Data* next_data = nullptr;
    Queue* next_queue = nullptr;
    QueueKey next_queue_key;
    for (auto it = queues_.begin(); it != queues_.end();) {
      const auto* data = it->second.queue.Peek<Data>();
      if (data == nullptr) {
        if (it->second.finished) {
          queues_.erase(it++);
          continue;
        }
        CannotMakeProgress(it->first);
        // add by dh, 在某一个传感器无数据时,允许其他传感器数据继续输入
        it++;
        continue; 
      }
      if (next_data == nullptr || data->GetTime() < next_data->GetTime()) {
        next_data = data;
        next_queue = &it->second;
        next_queue_key = it->first;
      }
      if(last_dispatched_time_ > next_data->GetTime())
      {
        LOG(ERROR) << "Non-sorted data added to queue: '" << it->first << "'";
      }
      ++it;
    }
    if (next_data == nullptr) {
      //// block by dh, 在某一个传感器无数据时,允许其他传感器数据继续输入
      if(queues_.empty())
      {
        LOG(ERROR) << "queues_ is empty";
      }
      return;
    }

    // If we haven't dispatched any data for this trajectory yet, fast forward
    // all queues of this trajectory until a common start time has been reached.
    const common::Time common_start_time =
        GetCommonStartTime(next_queue_key.trajectory_id);
    // add by dh, 在某一个传感器无数据时,避免计算错误
    if(common_start_time == common::Time::min())
    {
      // LOG(ERROR) << "common_start_time is min";

      return;
    }
    if (next_data->GetTime() >= common_start_time) {
      // Happy case, we are beyond the 'common_start_time' already.
     
      if(last_dispatched_time_ < next_data->GetTime())
      {
        // if(next_data->GetSensorId() == "range0"){
        //   LOG(INFO) << "last_dispatched_time_ is " << last_dispatched_time_;
        //   LOG(INFO) << "next_data->GetTime() is " << next_data->GetTime();
        // }
        last_dispatched_time_ = next_data->GetTime();
        next_queue->callback(next_queue->queue.Pop());
      }
      else
      {
        next_queue->queue.Pop();
      }
    } else if (next_queue->queue.Size() < 2) {
      if (!next_queue->finished) {
        // We cannot decide whether to drop or dispatch this yet.
        CannotMakeProgress(next_queue_key);

        return;
      }
      last_dispatched_time_ = next_data->GetTime();
      next_queue->callback(next_queue->queue.Pop());
    } else {
      // We take a peek at the time after next data. If it also is not beyond
      // 'common_start_time' we drop 'next_data', otherwise we just found the
      // first packet to dispatch from this queue.

      std::unique_ptr<Data> next_data_owner = next_queue->queue.Pop();
      if (next_queue->queue.Peek<Data>()->GetTime() > common_start_time) {
        if(last_dispatched_time_ < next_data->GetTime())
        {
          last_dispatched_time_ = next_data->GetTime();
          next_queue->callback(std::move(next_data_owner));
        }
      }
    }
  }
}

void OrderedMultiQueue::CannotMakeProgress(const QueueKey& queue_key) {
  blocker_ = queue_key;
  for (auto& entry : queues_) {
    if (entry.second.queue.Size() > kMaxQueueSize) {
      LOG_EVERY_N(WARNING, 60) << "Queue waiting for data: " << queue_key;
      entry.second.queue.Pop();
      return;
    }
  }
}

common::Time OrderedMultiQueue::GetCommonStartTime(const int trajectory_id) {
  // add by dh, 在某一个传感器无数据时,避免计算错误
  if(common_start_time_per_trajectory_.find(trajectory_id) == common_start_time_per_trajectory_.end())
  {
    for (auto& entry : queues_) {
      if (entry.first.trajectory_id == trajectory_id) {
        if(entry.second.queue.Peek<Data>() == nullptr)
        {
          // LOG(INFO) << entry.first << " is empty";
          return common::Time::min();
        }
      }
    }
  }
  
  auto emplace_result = common_start_time_per_trajectory_.emplace(
      trajectory_id, common::Time::min());
  common::Time& common_start_time = emplace_result.first->second;
  if (emplace_result.second) {
    for (auto& entry : queues_) {
      if (entry.first.trajectory_id == trajectory_id) {
        common_start_time = std::max(
            common_start_time, entry.second.queue.Peek<Data>()->GetTime());
      }
    }
    LOG(INFO) << "All sensor data for trajectory " << trajectory_id
              << " is available starting at '" << common_start_time << "'.";
  }
  return common_start_time;
}

}  // namespace sensor
}  // namespace cartographer
