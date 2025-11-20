#!/usr/bin/env python3
"""
RLVCR Trajectory测试：从实际trajectory数据中构造thinking process并计算advantage
"""
import pdb
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import Dict, List, Tuple

"""
RLVCR Thinking Level Prompts
Defines different thinking level templates for generating alternative thinking patterns
"""

ALFWORLD_TEMPLATE_ADA = """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. 
If you choose 'ACTION', you should directly output the action in this turn. Your output must strictly follow this format: 'Action: your next action'. After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment outputs 'Nothing happened', that means the previous action is invalid and you should try more options. 
Reminder: the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. Think when necessary, try to act directly more in the process.

There are four thinking levels:
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

At each step, you must first choose an appropriate level of thinking (one of the four levels) to respond based on the given scenario. The chosen level MUST be enclosed within <level> </level> tags.
Next, reason through the problem step-by-step using the chosen thinking level. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

[Output Format] 
Your output must adhere to the following format:
EXAMPLE 1:
<level>1</level>
<action>your_next_action</action>

EXAMPLE 2:
<level>2</level>
<think>
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Response: [Give a preliminary response]
</think>
<action>your_next_action</action>

EXAMPLE 3:
<level>3</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Response: [Give a preliminary response]
</think>
<action>your_next_action</action>

EXAMPLE 4:
<level>4</level>
<think>
Goal: [What needs to be accomplished]
Current state: [Current state and inventory]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Response: [Give a preliminary response]
</think>
<action>your_next_action</action>

{task_description}

Here is the task history
{action_history}

Now it's your turn to generate next step response.
""".strip()


THINK_MODE_2 = """
You are given a specific next step action that has already been selected. Your task is to reconstruct the logical thinking process that would lead to choosing **this exact action**, based solely on the information provided.

Context:
{history}

Next Step Action:
{action}

Your output should follow the structure below:

<think>
Goal: [Clearly state the main objective based on the task or situation]
Current State: [Describe the current situation, observations, or context]
Available Actions: [List the reasonable actions or options available at this point]
Response: [Summarize the reasoning process that logically leads to the chosen action]
</think>

Important Guidelines:
- You are **not** deciding what to do next — you are **explaining the thought process** that justifies the already chosen action.
- You **must not** mention or imply that you already know what the next step is.
- The reasoning should appear as if it led naturally to the selected action without revealing that it was pre-determined.
""".strip()


THINK_MODE_3 = """
You are given a specific next step action that has already been selected. Your task is to reconstruct the logical thinking process that would lead to choosing **this exact action**, based solely on the information provided.

Context:
{history}

Next Step Action:
{action}

Your output should follow the structure below:

<think>
Goal: [Clearly state the main objective based on the task or situation]
Current State: [Describe the current situation, observations, or context]
Available Actions: [List the reasonable actions or options available at this point]
Reflection: [Reflect on the history—what actions were taken, and what was learned from them?]
Response: [Summarize the reasoning process that logically leads to the chosen action]
</think>

Important Guidelines:
- You are **not** deciding what to do next — you are **explaining the thought process** that justifies the already chosen action.
- You **must not** mention or imply that you already know what the next step is.
- The reasoning should appear as if it led naturally to the selected action without revealing that it was pre-determined.
""".strip()

THINK_MODE_4 = """
You are given a specific next step action that has already been selected. Your task is to reconstruct the logical thinking process that would lead to choosing **this exact action**, based solely on the information provided.

Context:
{history}

Next Step Action:
{action}

Your output should follow the structure below:

<think>
Goal: [Clearly state the main objective based on the task or situation]
Current State: [Describe the current situation, observations, or context]
Available Actions: [List the reasonable actions or options available at this point]
Reflection: [Reflect on the history—what actions were taken, and what was learned from them?]
Evaluation: [Critically evaluate why the chosen action is the most effective or appropriate choice]
Response: [Summarize the reasoning process that logically leads to the chosen action]
</think>

Important Guidelines:
- You are **not** deciding what to do next — you are **explaining the thought process** that justifies the already chosen action.
- You **must not** mention or imply that you already know what the next step is.
- The reasoning should appear as if it led naturally to the selected action without revealing that it was pre-determined.
""".strip()

# 加载模型
model_path = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/models/qwen2.5-7b_cog-sft_alf_lr2e6_bs16_epoch5_full_0810"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

def extract_thinking_and_action(step_text: str) -> Tuple[str, str, int]:
    """从step中提取thinking, action和level"""
    # 提取level
    level_match = re.search(r'<level>(\d+)</level>', step_text)
    level = int(level_match.group(1)) if level_match else 1
    
    # 提取thinking
    think_match = re.search(r'<think>(.*?)</think>', step_text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    
    # 提取action
    action_match = re.search(r'<action>(.*?)</action>', step_text)
    action = action_match.group(1).strip() if action_match else ""
    
    return thinking, action, level

def build_history_context(trajectory_data: List[Dict], step_idx: int) -> str:
    """构建到指定step为止的完整history"""
    context_parts = []
    
    # 添加任务描述
    if trajectory_data:
        first_step = trajectory_data[0]
        context_parts.append(first_step['task_description'])
        context_parts.append(first_step['next_obs'])
    
    # 添加之前所有步骤的action和observation
    for i in range(step_idx):
        step = trajectory_data[i]
        # 只添加action部分，不包括thinking
        _, action, _ = extract_thinking_and_action(step['next_step'])
        if action:
            context_parts.append(f"Action: {action}")
        context_parts.append(f"Observation: {trajectory_data[i]['next_obs']}")
    pdb.set_trace()
    return "\n\n".join(context_parts)


def generate_alternative_thinking(history: str, action: str, original_thinking: str, original_level: int, target_level: int) -> str:
    """使用RLVCR prompt让模型生成指定level的thinking"""
    if target_level == 1:
        return ""
    elif target_level == original_level:
        return original_thinking
    
    # 清理history格式

    # 选择对应的prompt template
    if target_level == 2:
        prompt = THINK_MODE_2.format(history=history, action=action)
    elif target_level == 3:
        prompt = THINK_MODE_3.format(history=history, action=action)
    elif target_level == 4:
        prompt = THINK_MODE_4.format(history=history, action=action)
    else:
        return ""
    
    # 使用模型生成thinking
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    pdb.set_trace()
    # 提取thinking部分
    thinking_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
    if thinking_match:
        return thinking_match.group(1).strip()
    else:
        return generated_text 


def compute_confidence(history: str, thinking: str, action: str, level: int) -> float:
    """计算action confidence (geometric mean of token probabilities)"""
    # 构建完整的input
    if level == 1:
        response = f"<level>1</level><action>{action}</action>"
    else:
        response = f"<level>{level}</level><think>{thinking}</think><action>{action}</action>"
    
    response = response.split(f"<action>")[0] + "<action>"
    prefix = history + response
    
    # Tokenize
    pdb.set_trace()
    inputs = tokenizer(prefix + action, return_tensors="pt")
    action_tokens = tokenizer(action, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
    
    # 计算logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
    
    # 计算action tokens的probabilities
    prefix_len = len(tokenizer(prefix, return_tensors="pt")['input_ids'][0])
    log_probs = []
    
    for i, token_id in enumerate(action_tokens):
        pos = prefix_len + i - 1
        if pos >= 0 and pos < logits.shape[0]:
            token_logits = logits[pos]
            probs = torch.softmax(token_logits, dim=-1)
            prob = probs[token_id].item()
            log_probs.append(torch.log(torch.tensor(prob + 1e-10)).item())
    
    # 几何平均 (geometric mean)
    if log_probs:
        confidence = torch.exp(torch.tensor(np.min(log_probs))).item()
        return min(max(confidence, 0.0), 1.0)  # clamp到[0,1]
    else:
        return 0.0


def analyze_trajectory_step(trajectory_data: List[Dict], step_idx: int):
    """分析trajectory中的一个step"""
    if step_idx >= len(trajectory_data):
        print(f"Step {step_idx} out of range")
        return
    task_description = trajectory[0]['task_description']
    step = trajectory_data[step_idx]
    original_thinking, action, original_level = extract_thinking_and_action(step['next_step'])
    
    print(f"\n{'='*60}")
    print(f"=== Step {step_idx}: {action} ===")
    print(f"Original Level: {original_level}")
    print(f"Score: {step['score']}")
    print(f"Original thinking: {original_thinking[:100]}..." if len(original_thinking) > 100 else f"Original thinking: {original_thinking}")
    
    # 构建history context
    history = build_history_context(trajectory_data, step_idx)
    pdb.set_trace()
    print(f"History length: {len(history)} chars")
    
    # 测试4个thinking levels
    results = {}
    thinking_rewards = []
    
    for level in [1, 2, 3, 4]:
        # 生成对应level的thinking
        if level == original_level:
            thinking = original_thinking
        else:
            thinking = generate_alternative_thinking(history, action, original_thinking, original_level, level)
        
        # 计算thinking cost
        thinking_cost = len(tokenizer.encode(thinking)) if thinking else 0
        
        pdb.set_trace()
        # 计算confidence
        instruction = ALFWORLD_TEMPLATE_ADA.format(
            task_description=task_description,
            action_history=history
        )
        confidence = compute_confidence(instruction, thinking, action, level)
        
        # RLVCR reward计算
        R_entropy = confidence
        normalized_cost = min(1.0, thinking_cost / 250)  # cost_max = 250
        R_length = 0.5 - normalized_cost
        reward = R_entropy + 1.0 * R_length  # thinking_cost_alpha = 1.0
        
        thinking_rewards.append(reward)
        results[level] = {
            'thinking': thinking,
            'thinking_cost': thinking_cost,
            'confidence': confidence,
            'R_entropy': R_entropy,
            'R_length': R_length,
            'reward': reward
        }
        
        print(f"\nLevel {level}:")
        print(f"  Thinking: {thinking[:80]}..." if len(thinking) > 80 else f"  Thinking: {thinking}")
        print(f"  Cost: {thinking_cost} tokens")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Reward: R_entropy({R_entropy:.3f}) + R_length({R_length:.3f}) = {reward:.3f}")
    
    return results

def load_trajectory(file_path: str) -> List[Dict]:
    """加载trajectory数据"""
    trajectory = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                trajectory.append(json.loads(line.strip()))
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return []

# 主程序
if __name__ == "__main__":
    # 你的trajectory文件路径
    trajectory_path = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/qwen2.5-7b_cog_cold_start_grpo_0830_ckp50_mode5/1756447897/variation-537.jsonl"
    
    print("Loading trajectory...")
    trajectory = load_trajectory(trajectory_path)
    
    if not trajectory:
        print("Failed to load trajectory data")
        exit(1)
    
    print(f"Loaded trajectory with {len(trajectory)} steps")
    print(f"Task: {trajectory[0]['task_name']}")
    
    # 分析多个关键步骤
    key_steps = [6, 12, 14]  # 分析这几个有代表性的steps
    results = []
    
    for step_idx in key_steps:
        if step_idx < len(trajectory):
            result = analyze_trajectory_step(trajectory, step_idx)
            if result:
                results.append(result)
    
    # 总结分析
    print(f"\n{'='*60}")
    print("=== Overall Analysis ===")
    
    improvement_count = sum(1 for r in results if r['improvement_found'])
    improvement_ratio = improvement_count / len(results) if results else 0
    
    print(f"Steps analyzed: {len(results)}")
    print(f"Steps with better thinking found: {improvement_count}")
    print(f"Improvement ratio: {improvement_ratio:.2%}")
    
    if results:
        avg_advantage_range = np.mean([r['advantage_range'] for r in results])
        print(f"Average advantage range: {avg_advantage_range:.4f}")
        
        # 显示每个步骤的结果
        for r in results:
            status = "✓ Improved" if r['improvement_found'] else "✗ No improvement"
            print(f"Step {r['step_idx']}: L{r['original_level']}→L{r['best_level']}, "
                  f"adv {r['original_advantage']:+.3f}→{r['best_advantage']:+.3f}, {status}")