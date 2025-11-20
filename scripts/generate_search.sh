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

export MODE=5
export TASK=search
export STEP=5
export SET=train_mini

cd /apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/src
echo "Current directory $(pwd)"

# make command
python generate_search.py \
    --env $TASK \
    --env_step_limit $STEP \
    --output_path /apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/${SET}/${TASK}/${ACTOR_MODEL}_mode${MODE} \
    --actor_model $ACTOR_MODEL \
    --actor_port $ACTOR_PORT \
    --set $SET
