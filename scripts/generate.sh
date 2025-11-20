export OPENAI_API_KEY=8beCPow2KcmVGufSecmUZrTQhVN2OnPb
export AZURE_OPENAI_ENDPOINT=https://text-embedding-3-small-ailab.openai.azure.com/
export AZURE_OPENAI_API_KEY=b229340d8931472392a45e66eeef05a6
export DEEPSEEK_API_KEY=sk-b134cd42fd8a4ae68ffb272a9e27f558
# export ACTOR_MODEL="gpt-4.5-preview"
export ACTOR_MODEL="gpt-4o"
# export ACTOR_MODEL=qwen2-7b
# export ACTOR_MODEL=qwen2-32b
# export ACTOR_MODEL=qwen2-72b
# export ACTOR_PORT=8031
export ACTOR_PORT=8036

export MODE=0
export N_REPEAT=5
# export TASK=sciworld
export TASK=alfworld
# export TASK=babyai
# export TASK=webarena
export STEP=50
# export STEP=15
# export STEP=10
# export STEP=20
# export SERVER_BASE=http://9.135.121.49:8403
# export SERVER_BASE=http://11.216.73.147:8401
export SERVER_BASE=http://11.216.73.147:8033
# export SERVER_BASE=http://11.216.73.147:8033
# export SERVER_BASE=http://29.81.242.111:8035
# export SERVER_BASE=http://29.81.242.111:8000
export SET=train_mini

cd /apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/src
echo "Current directory $(pwd)"

# make command
CMD="python generate_gold.py \
    --env $TASK \
    --env_step_limit $STEP \
    --thinking_mode $MODE \
    --n_repeat $N_REPEAT \
    --env_server_base $SERVER_BASE \
    --output_path /apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/${SET}/${TASK}/${ACTOR_MODEL}_mode${MODE} \
    --actor_model $ACTOR_MODEL \
    --actor_port $ACTOR_PORT \
    --set $SET"

if [ -n "$CRITIC_MODEL" ]; then
    CMD="$CMD --critic_model $CRITIC_MODEL"
    CMD="$CMD --critic_port $CRITIC_PORT"
    CMD="$CMD --output_path /apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/${SET}/${TASK}/${ACTOR_MODEL}-${CRITIC_MODEL}"
fi

# run command
echo "Running command: $CMD"
$CMD