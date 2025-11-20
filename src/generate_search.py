#!/usr/bin/env python3
"""
Agent 与 Search 环境交互演示
基于 verl-agent 的逻辑，模拟完整的 search 交互流程
"""

import argparse
import json
import logging
import os
import re
import time
import pdb
import requests
from typing import Dict, List, Tuple
from logging import INFO
from llm_base import vllm, llm_azure, llm_hunyuan, llm_openai

# ==================== 配置 ====================
SEARCH_URL = "http://0.0.0.0:8000/retrieve"
PROXIES = {'http': None, 'https': None}  # 禁用代理
MAX_STEPS = 10
TOPK = 3

# ==================== Prompt 模板 ====================
SYSTEM_PROMPT = """You are an expert agent tasked with answering questions using a search engine.
You should first think about the problem, then choose one action:
(1) Search: <search>your query</search> - to get more information
(2) Answer: <answer>your answer</answer> - when you have enough information

You should enclose your thinking in <think></think> tags before taking action."""


SYSTEM_PROMPT_MODE_5 = """
You are an expert agent that answers questions using search when needed.

For each turn, select a thinking level, reason through the problem, then take ONE action:
- <search>query</search> - Get more information (search results will be provided)
- <answer>response</answer> - Provide final answer when you have sufficient information

## Thinking Levels
Level 1 - Instinctive Response: Immediate reaction based on intuition, no analysis.
Level 2 - Situational Awareness: Assess current state and available actions before acting.
Level 3 - Experience Integration: Reflect on past actions and outcomes to inform current decisions.
Level 4 - Strategic Planning: Assess the task goal, past lessons, and current state to analyze the future impact of each candidate action and optimize the decision.

## Output Format
<level>[1-4]</level>
<think>
[Your reasoning based on the chosen level - see templates below]
</think>
<search>query</search> OR <answer>response</answer>

## Thinking Templates
**Level 1:** Fixed Response (Skip Thinking)
Okay, I think I have finished thinking.

**Level 2:**
Goal: [What needs to be accomplished]
Current state: [What is already known]
Available actions: [What actions are valid right now]
Reasoning: [Choose the best action and explain why]

**Level 3:**
Goal: [What needs to be accomplished]
Current state: [What is already known]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Reasoning: [Choose the best action and explain why]

**Level 4:**
Goal: [What needs to be accomplished]
Current state: [What is already known]
Available actions: [What actions are valid right now]
Reflection: [How effective were recent actions, what was learned]
Evaluation: [Assess the potential effectiveness of each candidate action]
Reasoning: [Choose the best action and explain why]
""".strip()

INIT_TEMPLATE = """Question: {question}

Please solve this question step by step. You can search for information or give the final answer directly."""

OBSERVATION_TEMPLATE = """Search Results:
{search_results}"""


# ==================== 辅助函数 ====================

def parse_action(text: str) -> Tuple[str, str]:
    """
    解析模型输出，提取动作类型和内容
    
    Returns:
        (action_type, content): ('search', query) 或 ('answer', answer)
    """
    # 提取 search
    search_match = re.search(r"<search>(.*?)</search>", text, re.DOTALL | re.IGNORECASE)
    if search_match:
        return ('search', search_match.group(1).strip())
    
    # 提取 answer
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return ('answer', answer_match.group(1).strip())
    
    return ('invalid', '')


def call_search_api(query: str, logger) -> Dict:
    """调用检索服务"""
    payload = {
        "query": query,
        "topk": TOPK,
        "return_scores": True
    }
    
    logger.info(f"调用检索服务: {query}")
    
    try:
        resp = requests.post(
            SEARCH_URL,
            json=payload,
            proxies=PROXIES,
            timeout=30
        )
        
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("result", [[]])[0]
            logger.info(f"search success, return {len(results)} infos")
            return {
                "success": True,
                "results": results
            }
        else:
            logger.error(f"search failed: {resp.status_code}")
            return {"success": False, "error": f"HTTP {resp.status_code}"}
            
    except Exception as e:
        logger.error(f"error: {e}")
        return {"success": False, "error": str(e)}


def format_search_results(results: List[Dict]) -> str:
    """格式化搜索结果为 <information> 格式"""
    docs = []
    for i, item in enumerate(results, 1):
        doc = item.get("document", {})
        content = doc.get("contents", "")[:5000]  # 截断
        docs.append(f"Doc {i}: {content}")
    
    return "\n<information>\n" + "\n".join(docs) + "\n</information>\n"


def compute_reward(predicted_answer: str, ground_truth: List[str]) -> float:
    """计算奖励（简单的字符串匹配）"""
    predicted_lower = predicted_answer.lower().strip()
    
    for gt in ground_truth:
        if gt.lower().strip() in predicted_lower or predicted_lower in gt.lower().strip():
            return 1.0
    
    return 0.0


# ==================== 主交互循环 ====================

def run_episode(thinking_mode: int, question: str, ground_truth: List[str], variation: int, output_file: str, logger) -> Dict:
    """
    运行一个完整的 episode
    
    Args:
        question: 问题
        ground_truth: 标准答案列表
        variation: 问题索引
        output_file: 输出文件路径
        logger: 日志记录器
    
    Returns:
        episode_data: 包含完整轨迹的字典
    """
    logger.info("=" * 70)
    logger.info(f"开始 Episode")
    logger.info(f"问题: {question}")
    logger.info(f"标准答案: {ground_truth}")
    logger.info("=" * 70)
    
    # 初始化
    conversation_history = []
    done = False
    final_reward = 0.0
    
    # 第一步：发送问题
    if thinking_mode == 5:
        system_prompt = SYSTEM_PROMPT_MODE_5
    else:
        system_prompt = SYSTEM_PROMPT  # 其他模式暂时使用默认
    initial_prompt = system_prompt + "\n\n" + INIT_TEMPLATE.format(question=question)
    conversation_history.append({"role": "user", "content": initial_prompt})
    
    # 生成 task_name（简化版问题作为 ID）
    task_name = f"search-{variation}"
    task_description = question
    
    # 开始交互循环
    fail_counter = 0
    for step in range(MAX_STEPS):
        logger.info(f"\n--- Step {step}/{MAX_STEPS} ---")
        
        # 1. 构建 prompt
        prompt = conversation_history
        
        # 2. 调用 LLM 生成响应
        logger.info("生成动作...")
        agent_response = llm_hunyuan(conversation_history)
        logger.info(f"Agent 响应:\n{agent_response}")
        
        conversation_history.append({"role": "assistant", "content": agent_response})
        
        # 3. 解析动作
        action_type, action_content = parse_action(agent_response)
        logger.info(f"解析动作: {action_type} -> {action_content}")
        
        # 4. 执行动作并记录（alfworld 格式）
        next_obs = ""
        current_score = 0.0
        
        if action_type == 'search':
            # 执行搜索
            search_result = call_search_api(action_content, logger)
            
            if search_result["success"]:
                # 格式化结果
                formatted_results = format_search_results(search_result["results"])
                
                # 显示结果
                logger.info(f"搜索结果:\n{formatted_results[:500]}...")
                
                # 添加到对话历史
                observation = OBSERVATION_TEMPLATE.format(
                    search_results=formatted_results
                )
                conversation_history.append({"role": "user", "content": observation})
                
                next_obs = formatted_results
                current_score = 0.0
                fail_counter = 0
                
            else:
                logger.error(f"搜索失败: {search_result.get('error')}")
                next_obs = f"Error: {search_result.get('error')}"
                current_score = 0.0
                fail_counter += 1
                done = False
                
        elif action_type == 'answer':
            # 给出答案，计算奖励
            reward = compute_reward(action_content, ground_truth)
            logger.info(f"答案: {action_content}")
            logger.info(f"奖励: {reward}")
            
            next_obs = f"Final answer: {action_content}"
            current_score = reward
            final_reward = reward
            fail_counter = 0
            done = True
            
        else:
            # 无效动作
            logger.warning("无效动作，结束 episode")
            next_obs = "Invalid action"
            conversation_history.append({"role": "user", "content": observation})
            current_score = 0.0
            fail_counter += 1

        # 5. 保存为 alfworld 格式
        record = {
            "task_name": task_name,
            "task_description": task_description,
            "variation": variation,
            "gold_path": [],  # search 任务没有金标准路径
            "gold_path_string": "",
            "path_id": step,
            "next_step": agent_response,  # 包含 <think> 和 <search>/<answer>
            "next_obs": next_obs,
            "ground_truth": ground_truth,
            "score": current_score
        }
        
        # 追加写入文件
        with open(output_file, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"Step {step}: score={current_score}, done={done}")
        
        if fail_counter >= 5:
            done = True
            logger.info('Early stop due to consecutive invalid actions')
        
        # 检查是否结束
        if done:
            logger.info(f"Episode 结束于第 {step} 步")
            break
        
        time.sleep(1)  # 避免请求过快
    
    # 返回汇总数据
    episode_data = {
        "variation": variation,
        "question": question,
        "ground_truth": ground_truth,
        "total_steps": step + 1,
        "final_reward": final_reward,
        "success": final_reward > 0.5
    }
    
    logger.info("=" * 70)
    logger.info(f"Episode 完成")
    logger.info(f"总步数: {episode_data['total_steps']}")
    logger.info(f"最终奖励: {episode_data['final_reward']}")
    logger.info(f"成功: {episode_data['success']}")
    logger.info("=" * 70)
    
    return episode_data


# ==================== 主函数 ====================

def main():
    import random
    random.seed(42)

    parser = argparse.ArgumentParser(description="Search Agent Demo")
    parser.add_argument("--env", type=str, default="search", 
                       help="task")
    parser.add_argument("--env_step_limit", type=int, default=5, 
                       help="max steps")
    parser.add_argument("--set", default="test")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--actor_model", type=str, default="gpt-4")
    parser.add_argument("--actor_port", type=int, default=8401)
    parser.add_argument("--thinking_mode", type=int, default=5, help="Thinking mode level (0-5, 5=adaptive)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 初始化日志
    logging.basicConfig(
        level=INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_path}/agent.log")
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 示例数据（实际应该从 parquet 文件加载）
    with open("/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/data/nq_hotpotqa_train.json", 'r') as f:
        test_questions = json.load(f)

    test_questions = random.sample(test_questions, min(len(test_questions), 200))
    
    output_dir = f"{args.output_path}/{str(int(time.time()))}"
    if (not os.path.exists(output_dir)):
        try:
            os.makedirs(output_dir)
        except:
            pass
    
    # 运行多个 episodes
    results = []
    all_variations = []
    
    for i, data in enumerate(test_questions):
        variation = data.get("id", 0).split("_")[-1]
        all_variations.append(variation)
        
        logger.info(f"\n\n{'='*70}")
        logger.info(f"Variation {variation}")
        logger.info(f"{'='*70}")
        
        # 为每个 variation 创建单独的输出文件（JSONL 格式）
        output_file = f"{output_dir}/variation-{variation}.jsonl"
        
        try:
            episode_result = run_episode(
                thinking_mode=args.thinking_mode,
                question=data["question"],
                ground_truth=data["golden_answers"],
                variation=variation,
                output_file=output_file,
                logger=logger
            )
            results.append(episode_result)
                
        except Exception as e:
            logger.error(f"Episode {i} (variation {variation}) 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        time.sleep(2)
    
    # 计算统计信息
    success_rate = sum(r["success"] for r in results) / len(results) if results else 0
    avg_steps = sum(r["total_steps"] for r in results) / len(results) if results else 0
    avg_reward = sum(r["final_reward"] for r in results) / len(results) if results else 0
    
    # 保存 final_result.json（与你的格式一致）
    final_result = {
        "task_names": [f"search-{v}" for v in all_variations],
        "variations": all_variations,
        "scores": [r["final_reward"] for r in results],
        "args": vars(args)
    }
    
    with open(f"{output_dir}/final_result.json", 'w') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("所有 Episodes 完成")
    logger.info(f"成功率: {success_rate:.2%}")
    logger.info(f"平均步数: {avg_steps:.2f}")
    logger.info(f"平均奖励: {avg_reward:.2f}")
    logger.info(f"Variations: {all_variations}")
    logger.info(f"Scores: {[r['final_reward'] for r in results]}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()