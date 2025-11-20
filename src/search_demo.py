#!/usr/bin/env python3
"""简单的搜索演示 - 禁用代理访问本地服务"""
import re
import requests
import time
import json

SEARCH_URL = "http://0.0.0.0:8000/retrieve"

# ========== 关键：禁用代理 ==========
proxies = {
    'http': None,
    'https': None,
}

# 1. 问题
question = "Where is the capital of China?"
print("=" * 70)
print(f"问题: {question}\n")

# 2. 智能体输出（包含 search）
action = "<think>I think i need to search</think>\n<search>China Capital</search>"
print(f"智能体动作:\n{action}\n")

# 3. 提取查询
query = re.search(r"<search>(.*?)</search>", action, re.DOTALL).group(1).strip()
print(f"提取查询: '{query}'\n")

# 4. 准备请求数据
payload = {
    "query": query,
    "topk": 3,
    "return_scores": True
}

headers = {
    "Content-Type": "application/json"
}

print("调用检索服务...")
print(f"URL: {SEARCH_URL}")
print(f"禁用代理: {proxies}")
print(f"Payload: {json.dumps(payload, ensure_ascii=False)}\n")

# 5. 调用检索（禁用代理！）
try:
    resp = requests.post(
        SEARCH_URL,
        headers=headers,
        json=payload,
        proxies=proxies,  # ← 关键：禁用代理
        timeout=30
    )
    
    print(f"状态码: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        results = data.get("result", [[]])[0]
        
        print(f"✓ 成功! 返回 {len(results)} 个结果\n")
        
        # 6. 格式化并显示
        docs = []
        for i, item in enumerate(results, 1):
            doc = item.get("document", {})
            content = doc.get("contents", "")[:200]
            score = item.get("score", 0.0)
            docs.append(f"Doc {i}: {content}")
            print(f"[{i}] Score: {score:.3f}")
            print(f"    {content}...\n")
        
        # 7. 包装成 verl-agent 格式
        search_result = "\n<information>\n" + "\n".join(docs) + "\n</information>\n"
        
        print("verl-agent 格式的搜索结果:")
        print(search_result)
        
        
        # 8. 模拟下一步
        print("\n下一步智能体会基于这个结果生成答案:")
        print('<think>看到相关信息</think>')
        print('<answer>中国</answer>')
        
    else:
        print(f"❌ 失败: {resp.status_code}")
        print(f"响应: {resp.text[:300]}")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
