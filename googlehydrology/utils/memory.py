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

"""Memory utils."""

import contextlib
import ctypes
import gc
from ctypes.util import find_library


def release() -> None:
    """Collect freed memory, and trim on Linux-like systems.

    Return freed C memory to the OS. This has two purposes:
    1. OS allocates us new defragmented allocs when needed.
    2. Prevent races where memory is fragmented, and we ask
       for new allocs yet the underlying allocator didn't
       return the fragmented memory yet. In this case,
       we both allocate even more memory while holding onto
       memory that isn't used yet because it's fragmented
       and it was too fast to get reused.

    Linux supports trim. MacOS and Windows release freed
    memory more aggressively automatically.
    """
    gc.collect()

    with contextlib.suppress(OSError, AttributeError):  # Ignore non Unix-like
        ctypes.CDLL(find_library('c') or 'libc.so.6').malloc_trim(0)
