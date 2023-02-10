"""
    Copyright 2023 Reza NasiriGerdeh. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import os
import numpy as np

import logging
logger = logging.getLogger("utils")


class ResultFile:
    def __init__(self, result_file_name):
        # create root directory for results
        result_root = './results'
        if not os.path.exists(result_root):
            os.mkdir(result_root)

        # open result file
        self.result_file = open(file=f'{result_root}/{result_file_name}', mode='w')

    def write_header(self, header):
        self.result_file.write(f'{header}\n')
        self.result_file.flush()

    def write_result(self, epoch, result_list):
        digits_precision = 8

        result_str = f'{epoch},'
        for result in result_list:
            if result != '-':
                result = np.round(result, digits_precision)
            result_str += f'{result},'

        # remove final comma
        result_str = result_str[0:-1]

        self.result_file.write(f'{result_str}\n')
        self.result_file.flush()

    def close(self):
        self.result_file.close()
