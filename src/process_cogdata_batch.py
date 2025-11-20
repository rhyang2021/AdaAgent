import json
import os
from pathlib import Path
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 导入你的环境和提示
from prompt_base import INIT_SCIWORLD, INIT_ALFWORLD, THINK_MODE_5, THINK_MODE_0

env_prompts = {
    "sciworld": INIT_SCIWORLD,
    "alfworld": INIT_ALFWORLD
}

mode_prompts = {
    0: THINK_MODE_0,
    5: THINK_MODE_5
}

def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

def process_repeat_file(args):
    """处理单个repeat文件的函数"""
    repeat_file, env_name, success_threshold, mode = args
    
    try:
        # 读取轨迹
        trajectory = []
        with open(repeat_file, 'r') as f:
            for line in f:
                if line.strip():
                    trajectory.append(json.loads(line.strip()))
        
        if not trajectory:
            return None
        
        # 获取基本信息
        task_name = trajectory[0].get('task_name', 'unknown')
        final_score = trajectory[-1].get('score', 0)
        
        # 获取文件路径信息
        var_dir = repeat_file.parent
        exp_dir = var_dir.parent
        model_dir = exp_dir.parent
        
        result = {
            'model': model_dir.name,
            'experiment': exp_dir.name,
            'variation': var_dir.name,
            'repeat_file': repeat_file.name,
            'task_name': task_name,
            'score': final_score,
            'trajectory_length': len(trajectory),
            'success': final_score >= success_threshold,
            'file_path': str(repeat_file)
        }
        
        # 如果成功，生成训练数据
        if final_score >= success_threshold:
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
            
            # 添加训练数据到结果
            result['training_data'] = {
                "conversations": conversations,
                "system": system_prompt,
                "metadata": {
                    "model": model_dir.name,
                    "experiment": exp_dir.name,
                    "variation": var_dir.name,
                    "repeat_file": repeat_file.name,
                    "task_name": task_name,
                    "score": final_score,
                    "trajectory_length": len(trajectory)
                }
            }
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'file_path': str(repeat_file),
            'success': False
        }

def collect_repeat_files(base_dir, models, limit=None):
    """收集所有repeat文件"""
    repeat_files = []
    
    for model in models:
        model_path = Path(base_dir) / model
        if not model_path.exists():
            print(f"Skip {model} - path not found")
            continue
        
        for exp_dir in sorted(model_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            for var_dir in sorted(exp_dir.glob("variation-*")):
                for repeat_file in sorted(var_dir.glob("repeat-*.jsonl")):
                    repeat_files.append(repeat_file)
                    if limit and len(repeat_files) >= limit:
                        return repeat_files
    
    return repeat_files

def main():
    # 配置
    env_name = "alfworld"  # 改为 "sciworld" 如果处理科学世界数据
    mode = 0
    base_dir = f"/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/train_mini/{env_name}"
    models = [f"gpt-4o_mode{mode}"]
    output_dir = f"/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/processed_results_{env_name}_mode{mode}"

    # 测试配置
    TEST_LIMIT = 100  # 增加测试数量
    
    # Score阈值配置 - 根据环境设置满分标准
    SCORE_THRESHOLDS = {
        "alfworld": 1.0,      # ALFWorld满分是1
        "sciworld": 100.0     # ScienceWorld满分是100
    }
    
    success_threshold = SCORE_THRESHOLDS.get(env_name, 1.0)
    
    # 并行配置
    max_workers = min(mp.cpu_count()//2, 8)  # 限制最大进程数
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Parallel Simple Score Filter ===")
    print(f"Environment: {env_name}")
    print(f"Success threshold: {success_threshold}")
    print(f"Models: {models}")
    print(f"Test limit: {TEST_LIMIT} files")
    print(f"Max workers: {max_workers}")
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # 检查路径
    if not Path(base_dir).exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return
    
    start_time = time.time()
    
    # 收集所有repeat文件
    print("\nCollecting repeat files...")
    repeat_files = collect_repeat_files(base_dir, models, TEST_LIMIT)
    print(f"Found {len(repeat_files)} repeat files")
    
    if not repeat_files:
        print("No repeat files found!")
        return
    
    # 准备并行处理参数
    process_args = [(repeat_file, env_name, success_threshold, mode) for repeat_file in repeat_files]
    
    # 并行处理
    print(f"\nProcessing {len(repeat_files)} files with {max_workers} workers...")
    
    training_data = []
    stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0}))
    processed = 0
    errors = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_repeat_file, args): args[0] for args in process_args}
        
        # 处理结果
        for future in as_completed(future_to_file):
            repeat_file = future_to_file[future]
            processed += 1
            
            if processed % 20 == 0:
                print(f"  Processed {processed}/{len(repeat_files)} files, found {len(training_data)} successful")
            
            try:
                result = future.result()
                
                if result is None:
                    continue
                
                if 'error' in result:
                    errors += 1
                    print(f"  Error processing {result['file_path']}: {result['error']}")
                    continue
                
                # 更新统计
                model = result['model']
                task_name = result['task_name']
                stats[model][task_name]["total"] += 1
                
                if result['success']:
                    stats[model][task_name]["success"] += 1
                    
                    # 添加训练数据
                    if 'training_data' in result:
                        training_data.append(result['training_data'])
                
            except Exception as e:
                errors += 1
                print(f"  Error processing {repeat_file}: {e}")
    
    processing_time = time.time() - start_time
    
    # 输出结果
    print(f"\n=== Results ===")
    print(f"Total files processed: {processed}")
    print(f"Training examples generated: {len(training_data)}")
    print(f"Success rate: {len(training_data)/processed*100:.1f}%" if processed > 0 else "0%")
    print(f"Errors: {errors}")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Speed: {processed/processing_time:.1f} files/sec")
    
    # 按模型统计
    for model, model_stats in stats.items():
        print(f"\n{model}:")
        total = sum(s["total"] for s in model_stats.values())
        success = sum(s["success"] for s in model_stats.values())
        print(f"  Total: {total}, Success: {success} ({success/total*100:.1f}%)" if total > 0 else "  No data")
        
        for task, s in sorted(model_stats.items()):
            rate = s["success"]/s["total"]*100 if s["total"] > 0 else 0
            print(f"  {task}: {s['success']}/{s['total']} ({rate:.1f}%)")
    
    # 保存训练数据
    train_file = f"{output_dir}/{env_name}_training_data_parallel.json"
    print(f"\nSaving to: {train_file}")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(training_data)} training examples")
    
    # 保存统计
    stats_file = f"{output_dir}/parallel_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "env_name": env_name,
                "mode": mode,
                "test_limit": TEST_LIMIT,
                "success_threshold": success_threshold,
                "max_workers": max_workers
            },
            "results": {
                "total_processed": processed,
                "training_examples": len(training_data),
                "success_rate": len(training_data)/processed*100 if processed > 0 else 0,
                "errors": errors,
                "processing_time": processing_time,
                "files_per_second": processed/processing_time if processing_time > 0 else 0
            },
            "by_model": dict(stats)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved statistics to: {stats_file}")
    
    # 显示一个样本
    if training_data:
        print(f"\n=== Sample Training Data ===")
        sample = training_data[0]
        print(f"Task: {sample['metadata']['task_name']}")
        print(f"Score: {sample['metadata']['score']}")
        print(f"Conversations: {len(sample['conversations'])} turns")
        print(f"First conversation:")
        print(f"  Human: {sample['conversations'][0]['value'][:100]}...")
        if len(sample['conversations']) > 1:
            print(f"  GPT: {sample['conversations'][1]['value']}")
    
    print(f"\n=== Performance Summary ===")
    print(f"Files processed: {processed}")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Speed: {processed/processing_time:.1f} files/sec")
    print(f"Workers used: {max_workers}")
    print(f"Speedup estimate: ~{max_workers}x faster than sequential")

if __name__ == "__main__":
    main()