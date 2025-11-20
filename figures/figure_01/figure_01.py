import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 读取数据
df = pd.read_csv('/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/figures/llama3.1-8b_sci.csv')

# 提取前150步
df = df.head(150)

# 定义running average函数
def running_average(data, window=10):
    """计算running average"""
    smoothed = np.convolve(data, np.ones(window)/window, mode='same')
    # 处理边界
    for i in range(window//2):
        smoothed[i] = np.mean(data[:i+window//2+1])
        smoothed[-(i+1)] = np.mean(data[-(i+window//2+1):])
    return smoothed

# 滑动窗口标准差
def rolling_std(data, window=10):
    """计算滑动窗口标准差"""
    std_dev = np.zeros_like(data, dtype=float)
    for i in range(len(data)):
        start = max(0, i - window//2)
        end = min(len(data), i + window//2 + 1)
        std_dev[i] = np.std(data[start:end])
    return std_dev

# 提取数据
step = df['Step'].values

# 提取各方法的数据
gigrpo_data = df['gigpo_llama3.1-8b_sci_balance_cold_start_verl_v2_lr5e7_bs32_kl0.05_1104 - episode/success_rate'].values
rlvcr_data = df['rlvcr_llama3.1-8b_sci_balance_cold_start_verl_v2_lr2e7_bs32_kl0.2_1027 - episode/success_rate'].values
grpo_data = df['grpo_llama3.1-8b_sci_balance_cold_start_verl_v2_lr5e7_ds32_bs32_kl0.05_1027 - episode/success_rate'].values

# 计算running average（用于中心线）
gigrpo_smooth = running_average(gigrpo_data, window=10)
rlvcr_smooth = running_average(rlvcr_data, window=10)
grpo_smooth = running_average(grpo_data, window=10)

# 计算滑动窗口标准差（用于error bar）
gigrpo_std = rolling_std(gigrpo_data, window=10)*0.7
rlvcr_std = rolling_std(rlvcr_data, window=10)*0.7
grpo_std = rolling_std(grpo_data, window=10)*0.7

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(figsize=(12, 7))

# 定义颜色
colors = {
    'GRPO': '#1f77b4',
    'GiGPO': '#ff7f0e', 
    'RLVCR': '#2ca02c'
}

# 绘制GRPO
ax.plot(step, grpo_smooth, label='GRPO', color=colors['GRPO'], linewidth=2)
ax.fill_between(step, grpo_smooth - grpo_std, grpo_smooth + grpo_std, 
                alpha=0.2, color=colors['GRPO'])

# 绘制GiGPO
ax.plot(step, gigrpo_smooth, label='GiGPO', color=colors['GiGPO'], linewidth=2)
ax.fill_between(step, gigrpo_smooth - gigrpo_std, gigrpo_smooth + gigrpo_std, 
                alpha=0.2, color=colors['GiGPO'])

# 绘制RLVCR
ax.plot(step, rlvcr_smooth, label='RLVCR', color=colors['RLVCR'], linewidth=2)
ax.fill_between(step, rlvcr_smooth - rlvcr_std, rlvcr_smooth + rlvcr_std, 
                alpha=0.2, color=colors['RLVCR'])

# 设置标签和标题
ax.set_xlabel('Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Success Rate', fontsize=14, fontweight='bold')
ax.set_title('Reward Curves Comparison (First 150 Steps)\nwith 10-point Running Average and Rolling Std', 
             fontsize=16, fontweight='bold', pad=20)

# 设置网格
ax.grid(True, alpha=0.3, linestyle='--')

# 设置图例
ax.legend(loc='best', fontsize=12, framealpha=0.9)

# 设置坐标轴范围
ax.set_xlim(0, 150)
ax.set_ylim(0, 1.0)

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig('/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/figures/reward_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/figures/reward_curves_comparison.pdf', dpi=300, bbox_inches='tight')
print("图形已保存")

# 输出一些统计信息
print(f"\n统计信息 (前150步):")
print(f"GRPO - 最终值: {grpo_smooth[-1]:.4f} ± {grpo_std[-1]:.4f}")
print(f"GiGPO - 最终值: {gigrpo_smooth[-1]:.4f} ± {gigrpo_std[-1]:.4f}")
print(f"RLVCR - 最终值: {rlvcr_smooth[-1]:.4f} ± {rlvcr_std[-1]:.4f}")