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

import time
import requests
import tqdm

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import moxing as mox
import numpy as np
from mindspore import context, Tensor
from mindspore import export
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel.nn.transformer import TransformerOpParallelConfig
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.code_tokenizer import CodeTokenizer
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.pangu_alpha_fp16_predict import EvalNet, PanguAlphaModel
from src.utils import get_args
import re

# from generation_values_1p import load_model, run_predict_single

from flask import Flask, request
from threading import Thread
import json

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


APP = Flask(__name__)

class MyThread(Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result   # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

#gloabl varibale
model_predict = None
config = None
rank = 0

@APP.route('/health', methods=['GET'])
def health_func():
    return json.dumps({'health': 'true'}, indent=4)


@APP.route('/codegeex', methods=['POST'])
def inference_text():
    input = request.json
    samples = input['samples']
    result = run_predict_single(samples, model_predict, config, opt, rank)
    res_data = {
        "result": result
    }
    print(result)
    return json.dumps(res_data, indent=4)

def download_file(url, name):
    print("download_file")
    # 查看本地下载了多少
    if os.path.exists(name):
        # 本地已经下载的文件大小
        temp_size = os.path.getsize(name)
    else:
        temp_size = 0
    # 向头加入Range信息
    headers = {}
    headers['Range'] = 'bytes={}-'.format(temp_size)
    resp = requests.get(url=url, headers=headers, stream=True)
    total_size = int(resp.headers['Content-Length'])
    content_size = total_size

    print("已下载：", temp_size)
    print("总共需要下载：", total_size)
    with open(name, "wb") as f:
        print("Pkg total size is:", content_size, 'k,start...')
        for data in tqdm.tqdm(iterable=resp.iter_content(1024), total=content_size, unit='k', desc=name):
            f.write(data)
        print(name + "download finished!")
    print("total size: ", os.path.getsize(name))

def load_model(args_opt):
    r"""
     The main function for load model
    """
    global model_predict, config, rank
    # Set execution mode
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target=args_opt.device_target)
    context.set_context(variable_memory_max_size="30GB")
    print("init tokenizer")
    tokenizer = CodeTokenizer(mode="6b")
    print(tokenizer.encode_code("#quick start"))
    print("end tokenizer")
    # Set parallel context
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("rank_id is {}, device_num is {}".format(rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path)
    context.set_context(
        save_graphs=False,
        save_graphs_path="/cache/graphs_of_device_id_" + str(rank),
    )
    use_past = (args_opt.use_past == "true")
    print('local_rank:{}, start to run...'.format(rank), flush=True)
    if args_opt.export:
        use_past = True
    # Set model property
    print("===args_opt: ", args_opt, flush=True)
    print("===device_num is: ", device_num, flush=True)
    args_opt.op_level_model_parallel_num = 1
    model_parallel_num = args_opt.op_level_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    print("===data_parallel_num is: ", data_parallel_num, flush=True)

    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num,
                                                  model_parallel=model_parallel_num,
                                                  pipeline_stage=args_opt.stage_num,
                                                  micro_batch_num=args_opt.micro_size,
                                                  optimizer_shard=False,
                                                  vocab_emb_dp=bool(args_opt.word_emb_dp),
                                                  recompute=True)

    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num

    if args_opt.run_type == "predict":
        batch_size = 1
    config = PanguAlphaConfig(
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        hidden_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        ffn_hidden_size=args_opt.embedding_size * 4,
        use_past=use_past,
        eod_token=args_opt.eod_id,
        eod_reset=False,
        parallel_config=parallel_config,
        load_ckpt_path=args_opt.load_ckpt_path,
        param_init_type=mstype.float32
        if args_opt.param_init_type == 'fp32'
        else mstype.float16,
    )
    print("=====args_opt is: ", args_opt, flush=True)
    ckpt_name = args_opt.load_ckpt_name
    print("start download=============================")
    download_file(args_opt.load_ckpt_path, ckpt_name)
    # param_dict = load_checkpoint(os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name))
    print("download success==========================")
    # Define network
    print("Define network")
    pangu_alpha = PanguAlphaModel(config)
    eval_net = EvalNet(pangu_alpha, pad_token=50256)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    # Compile network and obtain tensor layout for loading ckpt
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    if args_opt.distribute == "false":
        predict_layout = None
    elif config.use_past:
        batch_valid_length = Tensor(np.array([0]), mstype.int32)
        init_true = Tensor([True], mstype.bool_)
        print("Input shape:", inputs_np.shape, flush=True)
        inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
        print("is_first_iteration=True", flush=True)
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index, init_true, batch_valid_length)
        model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
        print("is_first_iteration=False", flush=True)
        init_false = Tensor([False], mstype.bool_)
        _ = model_predict.infer_predict_layout(inputs_np_1, current_index, init_false, batch_valid_length)
    else:
        predict_layout = model_predict.infer_predict_layout(inputs_np, current_index)

    if context.get_context("save_graphs"):
        print("==============save_graph", flush=True)
        jobid = os.environ["BATCH_JOB_ID"]
        rank_id = rank
        mox.file.make_dirs("s3://wudao-1/yyf/graphs_" + jobid)
        mox.file.copy_parallel(src_url="/cache/graphs_of_device_id_" + str(rank_id),
                               dst_url="s3://wudao-1/yyf/graphs_" + jobid + "/" + str(rank_id))
    print("======start load_distributed checkpoint", flush=True)
    print("====epoch", args_opt.load_ckpt_epoch)
    if args_opt.load_ckpt_epoch > 0:
        # time.sleep(rank * 0.5)
        # if not mox.file.exists(os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}")):
        #     os.mkdir(os.path.join(args_opt.save_checkpoint_path, f"rank_{rank}"))
        # ckpt_name = f"code-13B0-{args_opt.load_ckpt_epoch}.ckpt"
        # if not mox.file.exists(os.path.join(args_opt.load_ckpt_path, f"rank_{rank}", ckpt_name)):
        #    print(f"Checkpoint from rank {rank} doesn't exist!")
        # mox.file.copy(args_opt.load_ckpt_path, ckpt_name)
        print("start load================================")
        param_dict = load_checkpoint(ckpt_name)
        print("load end==================================")
        # TODO: add them back if not for the 1st run!
        # if param_dict.get("epoch_num") and param_dict.get("step_num"):
        #     args_opt.has_trained_epoches = int(param_dict["epoch_num"].data.asnumpy())
        #     args_opt.has_trained_steps = int(param_dict["step_num"].data.asnumpy())
        # os.mkdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/1/rank_{rank}')
        # while True:
        #     num = len(os.listdir(f'/home/work/sfs/cache/{os.environ["BATCH_JOB_ID"]}/1'))
        #     if num == device_num:
        #         break
        #     if rank % 8 == 0:
        #         print("Loaded ckpt in step 1: ", num)
        #     time.sleep(1)
        net_not_load = load_param_into_net(pangu_alpha, param_dict)
        print("====== load_distributed checkpoint done, net_not_load: ", net_not_load, flush=True)
        print("start warmup")
        run_predict_single("#quick start", model_predict, config, args_opt, rank)
        print("warmup end")
    return model_predict, config, rank

# 加载模型
opt = get_args(True)
set_parse(opt)
# tokenizer = CodeTokenizer(mode='6b')
t = MyThread(load_model, args=(opt, ))
t.start()

def export_mindir(model_predict, config):
    """Export mindir model"""
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    current_index = Tensor(np.array([0]), mstype.int32)

    batch_valid_length = Tensor(np.array([0]), mstype.int32)
    init_true = Tensor([True], mstype.bool_)
    inputs_np_1 = Tensor(np.ones(shape=(config.batch_size, 1)), mstype.int32)

    model_predict.predict_network.add_flags_recursive(is_first_iteration=True)
    export(model_predict.predict_network, inputs_np, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1024', file_format='MINDIR')
    model_predict.predict_network.add_flags_recursive(is_first_iteration=False)
    export(model_predict.predict_network, inputs_np_1, current_index,
           init_true, batch_valid_length, file_name='pangu_alpha_1', file_format='MINDIR')
    print("Export finished and now exit.")


def run_predict(model_predict, config, args_opt, rank):
    """run predict"""
    from src.generate import generate, generate_increment
    # Define tokenizer
    tokenizer = CodeTokenizer(mode='6b')

    # Tokenize input sentence to ids
    samples = [
        "# language: Python\ndef add(a, b):\n    '''\n    Find the sum of a and b.\n    '''\n",
        "def add(a, b):\n    '''\n    Find the sum of a and b.\n    '''\n",
        "# language: Python\ndef optimization():\n    '''\n    Find the maximum of P=E**2*R/(R + r)**2 if E and r are fixed but R varies. Import sympy. Use sympy. Find where the derivative is equal to zero. Substitute the value of R into P.\n    '''\n",
        "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
        "// language: C++\nint add(int a, int b) {\n    /* Find the sum of a and b. */\n",
        "int add(int a, int b) {\n    /* Find the sum of a and b. */\n",
        "bool prime(int n) {\n    // Find whether n is a prime number\n",
        "// language: JavaScript\nfunction add(a, b) {\n    // Find the sum of a and b.\n",
        "# language: R\nadd<-function(a, b) {\n    # Find the sum of a and b.\n",
    ]
    # verbose = (rank == 0)
    verbose = False
    for i, sample in enumerate(samples):
        for _ in range(1):
            tokenized_token = tokenizer.encode_code(sample)
            input_ids = np.array(tokenized_token).reshape(1, -1)
            # Call inference
            generate_func = generate_increment if config.use_past else generate
            t0 = time.perf_counter()
            output_ids = generate_func(model_predict, input_ids, args_opt, verbose)
            # Decode output ids to sentence
            t1 = time.perf_counter()
            output_samples = tokenizer.decode_code(output_ids.tolist())
            output_samples_str = "".join(output_samples)
            if rank % 8 == 0:
                print(f"=================== prompt {i} ====================")
                print(sample, flush=True)
                print(f"=================== generation {i} ====================")
                print(output_samples_str, flush=True)
                print(
                    f"=== Total time (s): {t1 - t0}, {output_ids.shape[-1] - input_ids.shape[-1]} tokens, {(output_ids.shape[-1] - input_ids.shape[-1]) / (t1 - t0)} token/s")
        break


def string_processing(sample):
    if not sample:
        return '', 0
    if sample.endswith('\n'):
        return sample, 0
    if len(sample) > 3 and re.search('def', sample[-4:]):
        return sample, 0
    if re.search('def', sample):
        return sample + '\n ', 1
    return sample + '\n', 1


def run_predict_single(sample, model_predict, config, args_opt, rank):
    """run predict"""
    from src.generate import generate, generate_increment
    t0 = time.time()
    # Define tokenizer
    tokenizer = CodeTokenizer(mode='6b')

    # Tokenize input sentence to ids

    # verbose = (rank == 0)
    # preprocessing string
    add_status = 0

    # if not sample:
    #     return '# empty input'
    #
    # if sample[-1] != '\n' and sample[-1] != '\t':
    #     sample = sample + '\n'
    #     if re.search('def', sample) and not re.search('    ', sample):
    #         sample = sample + '    '
    #     add_status = 1
    # elif len(sample) > 3 and sample[-4:] != '    ' and sample[-1] != '\n':
    #     sample = sample + '\n'
    #     add_status = 1

    sample, add_status = string_processing(sample)

    verbose = False
    tokenized_token = tokenizer.encode_code(sample)
    input_ids = np.array(tokenized_token).reshape(1, -1)

    # Call inference
    generate_func = generate_increment if config.use_past else generate
    output_ids = generate_func(model_predict, input_ids, args_opt, verbose)

    # Decode output ids to sentence
    t1 = time.time()
    output_samples = tokenizer.decode_code(output_ids.tolist())
    output_samples_str = "".join(output_samples)
    output_samples_single = output_samples_str.split('\n')[-1]
    if output_samples_single != '' and add_status == 1:
        output_samples_single = '\n' + output_samples_single

    print("=================================Input=========================================")
    print(sample, flush=True)
    print("================================Output All=====================================")
    print(output_samples_str, flush=True)
    print("=============================Output Single=====================================")
    print(output_samples_single, flush=True)
    print(
        f"=== Total time (s)============: {t1 - t0}")
    return output_samples_single


def main():
    """Main process for predict or export model"""
    opt = get_args(True)
    set_parse(opt)
    model_predict, config, rank = load_model(opt)
    if opt.export:
        export_mindir(model_predict, config)
    else:
        run_predict(model_predict, config, opt, rank)


if __name__ == "__main__":
    APP.run(host='0.0.0.0', port='8080')
