import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.use('Agg')

data_path = "/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/figures/figure_01"
# 定义running average函数
def running_average(data, window=10):
    """计算running average"""
    # 处理NaN值
    data = np.array(data, dtype=float)
    mask = ~np.isnan(data)
    
    if not np.any(mask):
        return data
    
    smoothed = np.full_like(data, np.nan)
    valid_data = data[mask]
    
    if len(valid_data) < window:
        smoothed[mask] = valid_data
        return smoothed
    
    temp_smooth = np.convolve(valid_data, np.ones(window)/window, mode='same')
    
    # 处理边界
    for i in range(window//2):
        if i < len(valid_data):
            temp_smooth[i] = np.mean(valid_data[:i+window//2+1])
        if len(valid_data) - i - 1 >= 0:
            temp_smooth[-(i+1)] = np.mean(valid_data[-(i+window//2+1):])
    
    smoothed[mask] = temp_smooth
    return smoothed

# 滑动窗口标准差
def rolling_std(data, window=10):
    """计算滑动窗口标准差"""
    data = np.array(data, dtype=float)
    mask = ~np.isnan(data)
    
    if not np.any(mask):
        return data
    
    std_dev = np.full_like(data, np.nan)
    valid_indices = np.where(mask)[0]
    valid_data = data[mask]
    
    temp_std = np.zeros(len(valid_data), dtype=float)
    for i in range(len(valid_data)):
        start = max(0, i - window//2)
        end = min(len(valid_data), i + window//2 + 1)
        temp_std[i] = np.std(valid_data[start:end])
    
    std_dev[mask] = temp_std
    return std_dev

# 数据文件配置
datasets = [
    {
        'file': f'{data_path}/qwen2.5-7b_alf.csv',
        'title': 'ALFWorld',
        'model': 'Qwen2.5-7B',
        'methods': {
            'gigrpo': 'gigrpo_qwen2.5_7b_alf_balance_cold_start_lr5e7_bs32_1101 - episode/success_rate',
            'grpo': 'grpo_qwen2.5_7b_alf_balance_cold_start_lr5e7_bs64_0915 - episode/success_rate',
            'rlvcr': 'rlvcr_qwen2.5_7b_alf_balance_cold_start_lr5e7_bs64_tg_kl0.01_wofd_0923 - episode/success_rate'
        }
    },
    {
        'file': f'{data_path}/qwen2.5-7b_sci.csv',
        'title': 'ScienceWorld',
        'model': 'Qwen2.5-7B',
        'methods': {
            'gigpo': 'gigpo_qwen2.5_7b_sci_balance_cold_start_verl_v2_lr5e7_bs32_kl0.05_1103 - episode/success_rate',
            'grpo': 'grpo_qwen2.5_7b_sci_balance_cold_start_verl_v2_lr5e7_bs32_kl0.05_1023 - episode/success_rate',
            'rlvcr': 'rlvcr_qwen2.5_7b_sci_balance_cold_start_verl_v2_lr5e7_bs32_kl0.2_1019 - episode/success_rate'
        }
    },
        {
        'file': f'{data_path}/llama3.1-8b_alf.csv',
        'title': 'ALFWorld',
        'model': 'Llama3.1-8B',
        'methods': {
            'gigrpo': 'gigrpo_llama3.1-8b_alf_balance_cold_start_lr2e7_ds64_bs16_1106 - episode/success_rate',
            'grpo': 'grpo_llama3.1-8b_alf_balance_cold_start_old_template_lr5e7_bs32_kl0.01_1030 - episode/success_rate',
            'rlvcr': 'rlvcr_llama3.1-8b_alf_balance_cold_start_lr1e6_bs32_kl0.1_probs_1030 - episode/success_rate'
        }
    },
    {
        'file': f'{data_path}/llama3.1-8b_sci.csv',
        'title': 'ScienceWorld',
        'model': 'Llama3.1-8B',
        'methods': {
            'gigpo': 'gigpo_llama3.1-8b_sci_balance_cold_start_verl_v2_lr5e7_bs32_kl0.05_1104 - episode/success_rate',
            'grpo': 'grpo_llama3.1-8b_sci_balance_cold_start_verl_v2_lr5e7_ds32_bs32_kl0.05_1027 - episode/success_rate',
            'rlvcr': 'rlvcr_llama3.1-8b_sci_balance_cold_start_verl_v2_lr2e7_bs32_kl0.2_1027 - episode/success_rate'
        }
    }
]

# 定义颜色
colors = {
    'GRPO': '#569FE5',
    'GiGPO': '#F7D06B', 
    'CoPO': '#EC6E85'
}

# 创建1x4子图
fig, axes = plt.subplots(1, 4, figsize=(16.7, 3.36))

fs = 18
lw = 2.5

# 为每个数据集绘图
for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    
    # 读取数据
    df = pd.read_csv(dataset['file'])
    df = df.head(150)
    
    step = df['Step'].values
    
    # 处理每个方法
    for method_key in ['grpo', 'gigrpo', 'gigpo', 'rlvcr']:
        if method_key not in dataset['methods']:
            continue
            
        col_name = dataset['methods'][method_key]
        if col_name not in df.columns:
            continue
        
        # 提取数据并处理NaN
        data = df[col_name].values
        valid_mask = ~pd.isna(data)
        
        if not np.any(valid_mask):
            continue
        
        valid_steps = step[valid_mask]
        valid_data = data[valid_mask]
        
        # 计算平滑和标准差
        smooth = running_average(valid_data, window=10)
        std = rolling_std(valid_data, window=10) * 0.65
        
        # 确定标签和颜色
        if method_key in ['gigrpo', 'gigpo']:
            label = 'GiGPO'
            color = colors['GiGPO']
        elif method_key == 'grpo':
            label = 'GRPO'
            color = colors['GRPO']
        else:  # rlvcr
            label = 'CoPO'
            color = colors['CoPO']
        
        # 绘制曲线和阴影
        # 只在第一个子图添加label用于全局图例
        if idx == 0:
            ax.plot(valid_steps, smooth, label=label, color=color, linewidth=lw)
        else:
            ax.plot(valid_steps, smooth, color=color, linewidth=lw)
        ax.fill_between(valid_steps, smooth - std, smooth + std, 
                        alpha=0.2, color=color)
    
    # 设置子图属性
    ax.set_xlabel('Step', fontsize=fs)
    
    # 只有第一个子图有ylabel
    if idx == 0:
        ax.set_ylabel('Reward', fontsize=fs)
    
    # 标题格式：任务名 (模型名) - 增加pad参数让标题离图边框远一些
    ax.set_title(f"{dataset['title']} ({dataset['model']})", fontsize=fs, pad=11)
    
    # 设置网格样式：灰色虚线
    ax.grid(True, alpha=0.6, linestyle='--', linewidth=1, color='gray')
    ax.set_axisbelow(True)  # 让网格在数据下方
    
    # 设置边框：黑色，更粗
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    
    ax.set_xlim(0, 150)
    
    # 设置x轴刻度：只显示 0, 50, 100, 150
    ax.set_xticks([0, 50, 100, 150])
    
    # 让y轴显示5个刻度
    ax.yaxis.set_major_locator(MaxNLocator(5))
    
    # 增大刻度字体，去掉刻度线
    ax.tick_params(axis='both', which='major', labelsize=fs-1, 
                   length=0, width=0)  # length=0 去掉刻度线

# 从第一个子图获取图例句柄和标签，在顶部创建全局图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=fs, 
           frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, 1.13))

# 调整布局（不需要底部空间了）
plt.tight_layout(rect=[0, 0, 1, 0.94])

# 保存图形
plt.savefig(f'{data_path}/main_results_01.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{data_path}/main_results_01.pdf', dpi=300, bbox_inches='tight')
print("图形已保存")

# 输出统计信息
print("\n" + "="*60)
print("统计信息汇总 (前150步):")
print("="*60)

for dataset in datasets:
    df = pd.read_csv(dataset['file'])
    df = df.head(150)
    
    print(f"\n{dataset['model']} - {dataset['title']}")
    
    for method_key in ['grpo', 'gigrpo', 'gigpo', 'rlvcr']:
        if method_key not in dataset['methods']:
            continue
            
        col_name = dataset['methods'][method_key]
        if col_name not in df.columns:
            continue
        
        data = df[col_name].values
        valid_mask = ~pd.isna(data)
        
        if not np.any(valid_mask):
            continue
        
        valid_data = data[valid_mask]
        smooth = running_average(valid_data, window=10)
        std = rolling_std(valid_data, window=10) * 0.65
        
        if method_key in ['gigrpo', 'gigpo']:
            method_name = 'GiGPO'
        elif method_key == 'grpo':
            method_name = 'GRPO'
        else:
            method_name = 'CoPO'
        
        valid_smooth = smooth[~np.isnan(smooth)]
        valid_std = std[~np.isnan(std)]
        
        if len(valid_smooth) > 0:
            print(f"  {method_name:10s} - 最终值: {valid_smooth[-1]:.4f} ± {valid_std[-1]:.4f}")

print("\n" + "="*60)