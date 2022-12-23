FROM swr.cn-north-4.myhuaweicloud.com/modelarts-job-dev-image/mindspore-ascend910-cp37-euleros2.8-aarch64-training:1.3.0-3.3.0-roma

# 安装自定义版本Ascend Tookkit，这里使用的是5.0.4版本
USER root
COPY Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run /home/work/Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run
RUN /usr/local/Ascend/nnae/latest/script/uninstall.sh
RUN chmod a+x /home/work/Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run
RUN /home/work/Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run --full
RUN rm -rf Ascend-cann-toolkit_5.1.RC1_linux-aarch64.run

# 设置环境变量
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:$LD_LIBRARY_PATH \
    PYTHONPAiTH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH \
    PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:$PATH \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit \
    ASCEND_AUTOML_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools

# 安装Mindspore
USER work
RUN pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-0.4.0-py3-none-any.whl \
                /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-0.4.0-py3-none-any.whl \
                /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-0.1.0-py3-none-any.whl \
                https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.7.0/MindSpore/ascend/aarch64/mindspore_ascend-1.7.0-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple



# 安装Flask
RUN pip install Flask -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖库
RUN pip install loguru sentencepiece jieba tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install multiprocess \
		gevent \
		gunicorn -i https://pypi.tuna.tsinghua.edu.cn/simple


# 拷贝应用服务代码进镜像里面
COPY --chown=work:work model /home/model

WORKDIR /home/model/CodeGeex/codegeex/mindspore
# 制定启i动命令
# CMD python3 /home/model/CodeGeex/codegeex/mindspore/service.py --end_token 198 --use_past true --distribute false --vocab_size 51200 --mode 13B --micro_size 32 --param_init_type fp16 --per_batch_size 1 --top_p 0.95 --top_k_num 100 --temperature 0.8 --op_level_model_parallel_num 1 --frequency_penalty 0.0 --presence_penalty 0.0 --load_ckpt_name code-13B0-213000-FP16.ckpt --load_ckpt_path https://codegeex.obs.cn-central-221.ovaijisuan.com/checkpoint/code-13B0-213000-FP16.ckpt --load_ckpt_epoch 1 --tokenizer_path /home/model/CodeGeex/codegeex/tokenizer/
CMD gunicorn -w 2 -k gevent -b 0.0.0.0:8080 -t 1200 CodeGeex.codegeex.mindspore.service_multi:APP
