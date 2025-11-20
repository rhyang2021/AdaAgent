export OPENAI_API_KEY=8beCPow2KcmVGufSecmUZrTQhVN2OnPb
export AZURE_OPENAI_ENDPOINT=https://text-embedding-3-small-ailab.openai.azure.com/
export AZURE_OPENAI_API_KEY=b229340d8931472392a45e66eeef05a6
export DEEPSEEK_API_KEY=sk-cbb3a9d1898946339878d3974f99b6e9

# export ACTOR_MODEL="deepseek-reasoner"
# export ACTOR_MODEL="gpt-4o"
export ACTOR_MODEL="deepseek-reasoner"
# export ACTOR_MODEL="openai-o3"
# export ACTOR_MODEL="gemini-pro-2.5"
# export ACTOR_MODEL="claude-sonne-4"
# export ACTOR_MODEL="qwen2.5-7b"
# export ACTOR_MODEL="qwen2.5-7b_cog_cold_start_grpo_0830_ckp50"
# export ACTOR_MODEL="QwQ-32b"
# export ACTOR_MODEL=qwen2.5-7b_cog-sft_balance_sci_verl
# export ACTOR_MODEL=llama3.1-8b_gigpo_alf
# export ACTOR_MODEL=llama3.1-8b_gigpo_sci
# export ACTOR_MODEL=llama3.1-8b_grpo_sci_ckp130
# export ACTOR_MODEL=qwen2.5-7b_sft_sci
# export ACTOR_MODEL=r1-distill-qwen2.5-7b
# export ACTOR_MODEL=r1-distill-llama3.1-8b
# export ACTOR_MODEL=llama3.1-8b_cog-sft_sci
# export ACTOR_MODEL=llama3.1-8b_cog-sft_alf
# export ACTOR_MODEL=qwen2.5-7b_sft_sci
# export ACTOR_MODEL=qwen2.5-7b_sft_alf
# export ACTOR_MODEL=llama3.1-8b
# export ACTOR_MODEL=qwen2-72b

# export ACTOR_PORT=8032
# export ACTOR_PORT=8033
export ACTOR_PORT=8034
# export ACTOR_PORT=8035

export MODE=0
# export TASK=sciworld
export TASK=alfworld
# export TASK=babyai
# export TASK=webshop
# export STEP=20
export STEP=15
# export SERVER_BASE=http://29.127.81.26:8403
export SERVER_BASE=http://29.127.81.26:8401
export SET=test

cd /apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/src
echo "Current directory $(pwd)"

# make command
CMD="python eval.py \
    --env $TASK \
    --env_step_limit $STEP \
    --thinking_mode $MODE \
    --env_server_base $SERVER_BASE \
    --output_path /apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/${SET}/${TASK}/${ACTOR_MODEL}_mode${MODE} \
    --actor_model $ACTOR_MODEL \
    --actor_port $ACTOR_PORT \
    --set $SET"

if [ -n "$CRITIC_MODEL" ]; then
    CMD="$CMD --critic_model $CRITIC_MODEL"
    CMD="$CMD --critic_port $CRITIC_PORT"
    CMD="$CMD --output_path /apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/${SET}/${TASK}/${ACTOR_MODEL}-${CRITIC_MODEL}"
fi

# run command
echo "Running command: $CMD"
$CMD