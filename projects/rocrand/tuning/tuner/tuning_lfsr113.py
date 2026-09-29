#!/usr/bin/env python3

# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import Optional, OrderedDict, Callable
import sys
import os

sys.path.append(f"{os.path.dirname(__file__)}/../")

from utils import TYPE_CONFIGS
from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024, 2048]
GRID_SIZES = [32, 64, 128, 256, 512, 1024]

class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name='lfsr113')

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)

    def _get_tune_params(self) -> OrderedDict:
        params = OrderedDict()
        params['block_size_x'] = BLOCK_SIZES
        params['grid_size'] = GRID_SIZES
        return params

    def _get_restrictions(self) -> Callable[[dict], bool]:
        min_total_threads = 32768
        def validate(params):
            threads = params['block_size_x']
            blocks = params['grid_size']

            return threads * blocks >= min_total_threads

        return validate

if __name__ == "__main__":
    Tuner.cli()