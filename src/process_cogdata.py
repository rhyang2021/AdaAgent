import json
import os
from pathlib import Path
from collections import defaultdict
import time

# 导入你的环境和提示
from prompt_base import INIT_SCIWORLD, INIT_ALFWORLD, THINK_MODE_5, THINK_MODE_0, THINK_MODE_4

env_prompts = {
    "sciworld": INIT_SCIWORLD,
    "alfworld": INIT_ALFWORLD
}

mode_prompts = {
    0: THINK_MODE_5,
    4: THINK_MODE_4,
    5: THINK_MODE_5
}

def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

def main():
    # 配置
    env_name = "sciworld"  # 改为 "sciworld" 如果处理科学世界数据
    mode = 0
    base_dir = f"/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/train_mini/{env_name}"
    models = [f"gpt-4o_mode{mode}"]
    output_dir = f"/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/processed_results/{env_name}_mode{mode}"

    # 测试配置
    TEST_LIMIT = 10000  # 增加一些测试数量
    
    # Score阈值配置 - 根据环境设置满分标准
    SCORE_THRESHOLDS = {
        "alfworld": 1.0,      # ALFWorld满分是1
        "sciworld": 1.0     # ScienceWorld满分是100
    }
    
    success_threshold = SCORE_THRESHOLDS.get(env_name, 1.0)
    print(f"Success threshold for {env_name}: {success_threshold}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Simple Score Filter ===")
    print(f"Environment: {env_name}")
    print(f"Success threshold: {success_threshold}")
    print(f"Models: {models}")
    print(f"Test limit: {TEST_LIMIT} files")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # 检查路径
    if not Path(base_dir).exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return
    
    training_data = []
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0}))
    
    processed = 0
    start_time = time.time()
    
    # 处理文件
    for model in models:
        model_path = Path(base_dir) / model
        if not model_path.exists():
            print(f"Skip {model} - path not found")
            continue
            
        print(f"\nProcessing {model}...")
        
        # 遍历所有文件
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
                        final_score = trajectory[-1].get('score', 0)
                        
                        print(f"        Task: {task_name}, Score: {final_score}")
                        
                        stats[model][task_name]["total"] += 1
                        
                        # 检查score是否达到成功阈值
                        if final_score >= success_threshold:
                            print(f"        ✓ Score={final_score} >= {success_threshold}, generating training data")
                            
                            stats[model][task_name]["success"] += 1
                            
                            # 获取任务描述
                            task_description = trajectory[0].get('task_description', 'No task description')
                            
                            # 构建对话
                            conversations = []
                            
                            # 第一轮：任务描述
                            conversations.append({
                                "from": "human",
                                "value": clean(task_description)
                            })
                            
                            # 后续轮次：观察 -> 动作
                            for i, step in enumerate(trajectory):
                                # 如果不是第一步，先添加观察
                                if i > 0:
                                    prev_obs = trajectory[i-1].get('next_obs', '')
                                    if prev_obs:
                                        conversations.append({
                                            "from": "human", 
                                            "value": clean(prev_obs)
                                        })
                                
                                # 添加动作
                                action = step.get('next_step', '')
                                if action:
                                    conversations.append({
                                        "from": "gpt",
                                        "value": action
                                    })
                            
                            # 构建系统提示
                            system_prompt = env_prompts.get(env_name, INIT_ALFWORLD)
                            if mode in mode_prompts:
                                system_prompt += '\n' + mode_prompts[mode]
                            
                            # 保存训练数据
                            training_data.append({
                                "conversations": conversations,
                                "system": system_prompt,
                                "metadata": {
                                    "model": model,
                                    "experiment": exp_dir.name,
                                    "variation": var_dir.name,
                                    "repeat_file": repeat_file.name,
                                    "task_name": task_name,
                                    "score": final_score,
                                    "trajectory_length": len(trajectory)
                                }
                            })
                            
                            print(f"        ✓ Added to training data (total: {len(training_data)})")
                        else:
                            print(f"        ✗ Score={final_score} < {success_threshold}, skipping")
                            
                    except Exception as e:
                        print(f"        ERROR: {e}")
                
                if processed >= TEST_LIMIT:
                    break
            
            if processed >= TEST_LIMIT:
                break
    
    # 保存结果
    print(f"\n=== Results ===")
    print(f"Total files processed: {processed}")
    print(f"Training examples generated: {len(training_data)}")
    print(f"Success rate: {len(training_data)/processed*100:.1f}%" if processed > 0 else "0%")
    print(f"Processing time: {time.time()-start_time:.2f}s")
    
    # 输出统计
    for model, model_stats in stats.items():
        print(f"\n{model}:")
        total = sum(s["total"] for s in model_stats.values())
        success = sum(s["success"] for s in model_stats.values())
        print(f"  Total: {total}, Success: {success} ({success/total*100:.1f}%)" if total > 0 else "  No data")
        
        for task, s in sorted(model_stats.items()):
            rate = s["success"]/s["total"]*100 if s["total"] > 0 else 0
            print(f"  {task}: {s['success']}/{s['total']} ({rate:.1f}%)")
    
    # 保存训练数据
    train_file = f"{output_dir}/{env_name}_training_data_simple.json"
    print(f"\nSaving to: {train_file}")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(training_data)} training examples")
    
    # 保存统计
    stats_file = f"{output_dir}/simple_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "env_name": env_name,
                "mode": mode,
                "test_limit": TEST_LIMIT,
                "success_threshold": success_threshold
            },
            "results": {
                "total_processed": processed,
                "training_examples": len(training_data),
                "success_rate": len(training_data)/processed*100 if processed > 0 else 0,
                "processing_time": time.time() - start_time
            },
            "by_model": dict(stats)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved statistics to: {stats_file}")
    
    # 显示一个样本
    if training_data:
        print(f"\n=== Sample Training Data ===")
        sample = training_data[0]
        print(f"Task: {sample['metadata']['task_name']}")
        print(f"Conversations: {len(sample['conversations'])} turns")
        print(f"First conversation:")
        print(f"  Human: {sample['conversations'][0]['value'][:100]}...")
        if len(sample['conversations']) > 1:
            print(f"  GPT: {sample['conversations'][1]['value']}")

if __name__ == "__main__":
    main()