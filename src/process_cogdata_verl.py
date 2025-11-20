import json
import os
import re
from pathlib import Path
from collections import defaultdict
import time

# 导入标准模板
from prompt_base import (
    SCIWORLD_TEMPLATE_NO_HIS_ADA, 
    SCIWORLD_TEMPLATE_ADA, 
    SCIWORLD_TEMPLATE_NO_HIS_ADA_V2, 
    SCIWORLD_TEMPLATE_ADA_V2,
    ALFWORLD_TEMPLATE_NO_HIS_ADA,
    ALFWORLD_TEMPLATE_ADA
)

def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

def format_history(trajectory, current_step_idx):
    """
    格式化历史步骤: Observation+ Action
    """
    if current_step_idx == 0:
        return "No previous steps."
    
    action_history = ""
    
    # 显示所有历史步骤
    for i in range(0, current_step_idx):
        step = trajectory[i]
        # Observation 是上一步的 next_obs
        if i == 0:
            # 第一步的 observation 就是任务描述（初始观察）
            obs = trajectory[0].get('task_description', '')
        else:
            obs = trajectory[i-1].get('next_obs', '')
        
        # Action 是当前步的 action（从原始输出中提取）
        action = extract_action_from_output(step.get('next_step', ''))
        step_number = i + 1
        
        action_history += f"\n[Step {step_number}, Observation {step_number}: '{obs}', Action {step_number}: '{action}']"
    
    # 添加最近3步的reasoning（需要处理格式）
    if current_step_idx > 0:
        history_think_length = min(3, current_step_idx)
        think_start_index = current_step_idx - history_think_length
        action_history += "\n- recent reasoning process: \n"
        
        for i in range(think_start_index, current_step_idx):
            step = trajectory[i]
            # 获取原始输出并处理格式
            original_output = step.get('next_step', '')
            processed_output = process_output(original_output)
            step_number = i + 1
            action_history += f"[Step {step_number}, output {step_number}: '{processed_output}']\n"
    
    return action_history.strip()

def extract_action_from_output(output):
    """从输出中提取action"""
    # 尝试匹配 <action>...</action>
    match = re.search(r'<action>\s*(.+?)\s*</action>', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配 <|begin_of_answer|> ... <|end_of_answer|> 中的内容（兼容旧格式）
    match = re.search(r'<\|begin_of_answer\|>\s*Action:\s*(.+?)\s*<\|end_of_answer\|>', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 如果都没有，返回原始输出
    return output.strip()

def process_output(output):
    """
    处理输出格式：
    1. Level 1 如果没有 <think> 标签，补充固定文本
    2. 将所有 "Response:" 替换为 "Reasoning:"
    """
    # 1. 提取 level
    level_match = re.search(r'<level>(\d+)</level>', output)
    if not level_match:
        # 如果没有找到 <level> 标签，返回原始输出
        return output
    
    level = level_match.group(1)
    
    # 2. 检查是否有 <think> 标签
    has_think = re.search(r'<think>', output)
    
    # 3. 如果是 Level 1 且没有 <think> 标签，需要补充
    if level == "1" and not has_think:
        # 提取 action
        action_match = re.search(r'<action>(.+?)</action>', output, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            # 重新构建输出
            output = f"<level>1</level>\n<think>Okay, I think I have finished thinking.</think>\n<action>{action}</action>"
    
    # 4. 将 "Response:" 替换为 "Reasoning:"
    output = re.sub(r'\bResponse:', 'Reasoning:', output)
    
    return output

def build_human_prompt(task_description, trajectory, step_idx, env_name):
    """
    使用标准模板构建human prompt
    - 根据 env_name 选择对应的模板
    - 第一步：使用 NO_HIS 模板
    - 后续步骤：使用完整模板
    """
    # 根据环境选择模板
    if env_name == "alfworld":
        template_no_his = ALFWORLD_TEMPLATE_NO_HIS_ADA
        template_with_his = ALFWORLD_TEMPLATE_ADA
    else:  # sciworld
        template_no_his = SCIWORLD_TEMPLATE_NO_HIS_ADA_V2
        template_with_his = SCIWORLD_TEMPLATE_ADA_V2
    
    if step_idx == 0:
        # 第一步，使用 NO_HIS 模板
        prompt = template_no_his.format(
            task_description=task_description,
            current_observation=task_description
        )
    else:
        # 后续步骤，使用完整模板
        current_obs = trajectory[step_idx-1].get('next_obs', '')
        action_history = format_history(trajectory, step_idx)
        history_length = min(step_idx, 100)  # 实际包含的历史长度
        
        prompt = template_with_his.format(
            task_description=task_description,
            step_count=step_idx,
            history_length=history_length,
            action_history=action_history,
            current_step=step_idx + 1,
            current_observation=current_obs
        )
    
    return prompt

def main():
    # 配置
    env_name = "alfworld"  # 可以改为 "alfworld"
    mode = 0
    base_dir = f"/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/train_mini/{env_name}"
    models = [f"gpt-4o_mode{mode}"]
    output_dir = f"/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/processed_results/{env_name}_mode{mode}_verl_v2"

    # 测试配置
    TEST_LIMIT = 10000
    
    # Score阈值配置
    SCORE_THRESHOLDS = {
        "alfworld": 1.0,
        "sciworld": 1.0
    }
    
    success_threshold = SCORE_THRESHOLDS.get(env_name, 1.0)
    print(f"Success threshold for {env_name}: {success_threshold}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Single Turn Format Processor ===")
    print(f"Environment: {env_name}")
    print(f"Success threshold: {success_threshold}")
    
    # 根据环境显示使用的模板
    if env_name == "alfworld":
        print(f"Using templates: ALFWORLD_TEMPLATE_NO_HIS_ADA, ALFWORLD_TEMPLATE_ADA")
    else:
        print(f"Using templates: SCIWORLD_TEMPLATE_NO_HIS_ADA_V2, SCIWORLD_TEMPLATE_ADA_V2")
    
    print(f"Output processing:")
    print(f"  - Add <think> for Level 1 if missing")
    print(f"  - Replace 'Response:' with 'Reasoning:'")
    print(f"Models: {models}")
    print(f"Test limit: {TEST_LIMIT} files")
    
    # 检查路径
    if not Path(base_dir).exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return
    
    training_data = []
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0, "steps": 0}))
    
    processed = 0
    start_time = time.time()
    
    # 处理文件
    for model in models:
        model_path = Path(base_dir) / model
        if not model_path.exists():
            print(f"Skip {model} - path not found")
            continue
            
        print(f"\nProcessing {model}...")
        
        for exp_dir in sorted(model_path.iterdir()):
            if not exp_dir.is_dir():
                continue
                
            print(f"  Experiment: {exp_dir.name}")
            
            for var_dir in sorted(exp_dir.glob("variation-*")):
                print(f"    Variation: {var_dir.name}")
                
                for repeat_file in sorted(var_dir.glob("repeat-*.jsonl")):
                    if processed >= TEST_LIMIT:
                        print(f"    Reached test limit")
                        break
                        
                    processed += 1
                    print(f"      Processing {processed}/{TEST_LIMIT}: {repeat_file.name}")
                    
                    try:
                        # 读取轨迹
                        trajectory = []
                        with open(repeat_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    trajectory.append(json.loads(line.strip()))
                        
                        if not trajectory:
                            print(f"        Empty file, skipping")
                            continue
                        
                        # 获取基本信息
                        task_name = trajectory[0].get('task_name', 'unknown')
                        task_description = trajectory[0].get('task_description', 'No task description')
                        final_score = trajectory[-1].get('score', 0)
                        
                        print(f"        Task: {task_name}, Score: {final_score}, Steps: {len(trajectory)}")
                        
                        stats[model][task_name]["total"] += 1
                        
                        # 只处理成功的trajectory
                        if final_score >= success_threshold:
                            print(f"        ✓ Success! Generating {len(trajectory)} training samples")
                            
                            stats[model][task_name]["success"] += 1
                            stats[model][task_name]["steps"] += len(trajectory)
                            
                            # 为trajectory中的每一步生成一个训练样本
                            for step_idx, step in enumerate(trajectory):
                                # 构建human prompt（使用标准模板，传入env_name）
                                human_prompt = build_human_prompt(
                                    task_description, 
                                    trajectory, 
                                    step_idx,
                                    env_name  # 传入环境名称
                                )
                                
                                # 获取GPT输出并处理格式
                                original_output = step.get('next_step', '')
                                
                                if original_output:
                                    # 处理输出：补充Level 1的think，替换Response为Reasoning
                                    gpt_response = process_output(original_output)
                                    
                                    # 保存为单轮对话格式
                                    training_data.append({
                                        "conversations": [
                                            {
                                                "from": "human",
                                                "value": human_prompt  # 不clean，保留格式
                                            },
                                            {
                                                "from": "gpt",
                                                "value": gpt_response
                                            }
                                        ],
                                        "system": "",  # 空的system prompt
                                        "metadata": {
                                            "model": model,
                                            "experiment": exp_dir.name,
                                            "variation": var_dir.name,
                                            "repeat_file": repeat_file.name,
                                            "task_name": task_name,
                                            "final_score": final_score,
                                            "step_idx": step_idx,
                                            "total_steps": len(trajectory)
                                        }
                                    })
                            
                            print(f"        ✓ Added {len(trajectory)} samples (total: {len(training_data)})")
                        else:
                            print(f"        ✗ Score={final_score} < {success_threshold}, skipping")
                            
                    except Exception as e:
                        print(f"        ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                
                if processed >= TEST_LIMIT:
                    break
            
            if processed >= TEST_LIMIT:
                break
    
    # 保存结果
    print(f"\n=== Results ===")
    print(f"Total trajectories processed: {processed}")
    print(f"Training samples generated: {len(training_data)}")
    print(f"Processing time: {time.time()-start_time:.2f}s")
    
    # 输出统计
    for model, model_stats in stats.items():
        print(f"\n{model}:")
        total = sum(s["total"] for s in model_stats.values())
        success = sum(s["success"] for s in model_stats.values())
        total_steps = sum(s["steps"] for s in model_stats.values())
        print(f"  Total trajectories: {total}, Success: {success} ({success/total*100:.1f}%)" if total > 0 else "  No data")
        print(f"  Total training samples: {total_steps}")
        
        for task, s in sorted(model_stats.items()):
            rate = s["success"]/s["total"]*100 if s["total"] > 0 else 0
            print(f"  {task}: {s['success']}/{s['total']} ({rate:.1f}%), {s['steps']} samples")
    
    # 保存训练数据
    train_file = f"{output_dir}/{env_name}_training_data_standard.json"
    print(f"\nSaving to: {train_file}")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(training_data)} training samples")
    
    # 保存统计
    stats_file = f"{output_dir}/standard_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "env_name": env_name,
                "mode": mode,
                "test_limit": TEST_LIMIT,
                "success_threshold": success_threshold,
                "template": f"{env_name.upper()}_TEMPLATE_ADA"
            },
            "results": {
                "total_trajectories": processed,
                "training_samples": len(training_data),
                "processing_time": time.time() - start_time
            },
            "by_model": dict(stats)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved statistics to: {stats_file}")
    
    # 显示样本
    if training_data:
        print(f"\n=== Sample Training Data ===")
        sample = training_data[0]
        print(f"Task: {sample['metadata']['task_name']}")
        print(f"Step: {sample['metadata']['step_idx']}/{sample['metadata']['total_steps']}")
        print(f"\nHuman prompt (first 300 chars):")
        print(f"{sample['conversations'][0]['value'][:300]}...")
        print(f"\nGPT response (first 300 chars):")
        print(f"{sample['conversations'][1]['value'][:300]}...")

if __name__ == "__main__":
    main()