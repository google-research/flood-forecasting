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

"""Unit tests for googlehydrology.utils.logging_utils."""

import logging

import pytest

from googlehydrology.utils.logging_utils import (
    WarningOnceFilter,
    get_git_hash,
    setup_logging,
)


@pytest.mark.unit
def test_warning_once_filter():
    filt = WarningOnceFilter()
    record1 = logging.LogRecord(
        'test', logging.WARNING, 'test.py', 10, 'duplicate msg', (), None
    )
    record2 = logging.LogRecord(
        'test', logging.WARNING, 'test.py', 10, 'duplicate msg', (), None
    )
    record_info = logging.LogRecord(
        'test', logging.INFO, 'test.py', 10, 'info msg', (), None
    )

    # First warning passes
    assert filt.filter(record1) is True
    # Duplicate warning filtered out
    assert filt.filter(record2) is False
    # Info log passes regardless
    assert filt.filter(record_info) is True


@pytest.mark.unit
def test_get_git_hash():
    # Calling in git repo should return string hash or None
    git_hash = get_git_hash()
    assert git_hash is None or isinstance(git_hash, str)


@pytest.mark.unit
def test_setup_logging(tmp_path):
    log_file = tmp_path / 'test_run.log'
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    try:
        for h in original_handlers:
            root_logger.removeHandler(h)

        setup_logging(
            log_file=str(log_file),
            level=logging.INFO,
            print_warnings_once=True,
        )
        logging.info('Test log line')
        for h in root_logger.handlers:
            h.flush()

        assert log_file.exists()
        content = log_file.read_text()
        assert 'Test log line' in content or 'initialized' in content
    finally:
        for h in list(root_logger.handlers):
            h.close()
            root_logger.removeHandler(h)
        for h in original_handlers:
            root_logger.addHandler(h)
