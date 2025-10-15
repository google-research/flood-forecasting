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

class NoTrainDataError(Exception):
    """Raised, when basin contains no valid samples in training period"""

class NoEvaluationDataError(Exception):
    """Raised, when basin contains no valid samples in validation or test period""" 

class AllNaNError(Exception):
    """Raised by `calculate_(all_)metrics` if all observations or all simulations are NaN. """
