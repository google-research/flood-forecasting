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

from pathlib import Path

from setuptools import setup

# read the description from the README.md
readme_file = Path(__file__).absolute().parent / "README.md"
with readme_file.open("r") as fp:
    long_description = fp.read()

about = {}
with open("googlehydrology/__about__.py", "r") as fp:
    exec(fp.read(), about)

setup(
    name='googlehydrology',
    version=about['__version__'],
    packages=[
        'googlehydrology',
        'googlehydrology.datasetzoo',
        'googlehydrology.datautils',
        'googlehydrology.utils',
        'googlehydrology.modelzoo',
        'googlehydrology.training',
        'googlehydrology.evaluation',
    ],
    url='https://googlehydrology.readthedocs.io',
    project_urls={
        'Documentation': 'https://googlehydrology.readthedocs.io',
        'Source': 'https://github.com/googlehydrology/googlehydrology',
        'Research Blog': 'https://googlehydrology.github.io/',
    },
    author='Amit Markel, Frederik Kratzert, Grey Nearing, Martin Gauch, Omri Shefi',
    author_email='googlehydrology@googlegroups.com',
    description='Library for training deep learning models with environmental focus',
    long_description=long_description,
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'schedule-runs=googlehydrology.run_scheduler:_main',
            'run=googlehydrology.run:_main',
            'results-ensemble=googlehydrology.utils.results_ensemble:_main',
        ]
    },
    python_requires='>=3.12',
    install_requires=[
        'matplotlib',
        'numba',
        'numpy',
        'pandas',
        'ruamel.yaml',
        'torch',
        'scipy',
        'tensorboard',
        'tqdm',
        'xarray',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Hydrology',
        'License :: OSI Approved :: BSD License',
    ],
    keywords='deep learning hydrology lstm neural network streamflow discharge rainfall-runoff',
)
