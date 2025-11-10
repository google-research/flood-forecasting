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

(conda list | grep gprof2dot > /dev/null) || (conda install gprof2dot -c conda-forge --yes)
(conda list | grep graphviz > /dev/null) || (conda install graphviz -c conda-forge --yes)

rm -f /tmp/profile.pstats /tmp/profile.png
python3 -m cProfile -o /tmp/profile.pstats $HOME/flood-forecasting/googlehydrology/run.py train --config-file $HOME/flood-forecasting/config/multimet_mean_embedding_forecast_lstm.yml
gprof2dot -f pstats /tmp/profile.pstats | dot -Tpng -o /tmp/profile.png
open /tmp/profile.png
