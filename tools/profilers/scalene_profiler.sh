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

rm -f /tmp/scalene_output.html
echo Be patient after done running.
scalene --profile-all --stacks --web --outfile /tmp/scalene_output.html $HOME/flood-forecasting/googlehydrology/run.py --- train --config-file $HOME/flood-forecasting/config/multimet_mean_embedding_forecast_lstm.yml
open /tmp/scalene_output.html
