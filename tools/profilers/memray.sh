# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o pipefail

(conda list | grep memray > /dev/null) || (conda install memray)
(conda list | grep memray > /dev/null) || (echo "memray is required" && exit -1)

rm -f /tmp/memray_output.bin /tmp/memray_flamegraph.html

echo Profiling...
memray run --aggregate --native -o /tmp/memray_output.bin ~/flood-forecasting/googlehydrology/run.py \
        train --config-file ~/flood-forecasting/config/multimet_mean_embedding_forecast_lstm.yml

echo Analysing...
memray flamegraph -o /tmp/memray_flamegraph.html /tmp/memray_output.bin

open /tmp/memray_flamegraph.html
