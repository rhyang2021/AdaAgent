#!/usr/bin/env bash

#pip install keras
#pip install tf-keras
#pip install urllib3==1.26.6
#
echo '——————————kill gpu process——————————'
pid=$(ps -ef | grep 'gpu' | grep 'python' | grep -v grep | awk '{print $2}')
echo $pid
kill -9 $pid

llamafactory-cli train llamafactory_ds.yaml
#NNODES=3 NODE_RANK=1 MASTER_ADDR=$CHIEF_IP MASTER_PORT=6000 llamafactory-cli train deepseek_distll_qwen_sft.yaml

echo '____启动gpu进程____'
cd /apdcephfs_cq10/share_1567347/share_info/ruihanyang
bash occupy_gpus.sh
