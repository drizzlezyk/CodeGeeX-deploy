# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
PanGu predict run
"""
import json

from src.pangu_alpha_config import set_parse
from src.utils import get_args

from generation_values_1p import load_model, run_predict_single

from flask import Flask, request
from multiprocessing import cpu_count, Process

from gevent import monkey
from gevent.pywsgi import WSGIServer
monkey.patch_all(thread=False)


APP = Flask(__name__)


@APP.route('/health', methods=['GET'])
def health_func():
    return json.dumps({'health': 'true'}, indent=4)


@APP.route('/', methods=['POST'])
def inference_text():
    input = request.json
    samples = input['samples']

    opt = get_args(True)
    set_parse(opt)
    model_predict, config, rank = load_model(opt)

    result = run_predict_single(samples, model_predict, config, opt, rank)

    res_data = {
        "result": result
    }
    print(result)
    return json.dumps(res_data, indent=4)




def run(MULTI_PROCESS):
    if MULTI_PROCESS == False:
        WSGIServer(('0.0.0.0', 8080), APP).serve_forever()
    else:
        mulserver = WSGIServer(('0.0.0.0', 8080), APP)
        mulserver.start()

        def server_forever():
            mulserver.start_accepting()
            mulserver._stop_event.wait()

        # for i in range(cpu_count()):
        for i in range(1):
            p = Process(target=server_forever)
            p.start()


if __name__ == "__main__":
    APP.run(host='0.0.0.0', port='8080')
