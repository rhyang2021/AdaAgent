import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib
from collections import OrderedDict
matplotlib.use('Agg')

# 数据来自第一张图片
# Qwen2.5-7B - SciWorld (顺序: CogSFT, GRPO, GiGPO, CoPO)
qwen_sciworld = OrderedDict([
    ('CogSFT', [0.31, 44.13, 53.02, 2.52]),
    ('GRPO', [5.70, 9.39, 18.78, 66.14]),
    ('GiGPO', [0.23, 3.36, 6.42, 89.99]),
    ('CoPO', [72.58, 11.16, 8.62, 7.64])
])

# Qwen2.5-7B - AlfWorld
qwen_alfworld = OrderedDict([
    ('CogSFT', [0.84, 8.23, 90.93, 0.00]),
    ('GRPO', [0.00, 0.05, 23.15, 76.79]),
    ('GiGPO', [5.12, 5.98, 12.27, 76.63]),
    ('CoPO', [54.97, 18.74, 13.96, 12.33])
])

# Llama3.1-8B - SciWorld
llama_sciworld = OrderedDict([
    ('CogSFT', [0.38, 24.60, 71.61, 3.42]),
    ('GRPO', [9.74, 8.06, 11.17, 71.03]),
    ('GiGPO', [9.43, 11.83, 23.06, 55.68]),
    ('CoPO', [83.71, 7.33, 4.78, 4.18])
])

# Llama3.1-8B - AlfWorld
llama_alfworld = OrderedDict([
    ('CogSFT', [1.40, 20.23, 76.42, 1.95]),
    ('GRPO', [0.00, 20.68, 13.53, 65.41]),
    ('GiGPO', [10.87, 9.75, 8.54, 70.84]),
    ('CoPO', [87.25, 12.66, 0.07, 0.02])
])

# 定义颜色 - 匹配PDF图表的配色风格
colors = {
    'Level 1': '#F8D348',  # 黄色 (Critique)
    'Level 2': '#55B493',  # 绿色 (DGAP)
    'Level 3': '#7FA6DE',  # 蓝色 (Self-Critique)
    'Level 4': '#EB5E60'   # 橙红色 (GPT-4)
}

# 定义不同的纹理模式
hatches = {
    'Level 1': '///',      # 右斜线
    'Level 2': '\\\\\\',   # 左斜线
    'Level 3': '',      # 交叉
    'Level 4': '+++'       # 加号
}

# 创建图形 - 1行4列，使用与参考代码一致的尺寸
fig, axes = plt.subplots(1, 4, figsize=(16.7, 3.36))

# 字体大小设置
fs = 18

datasets = [
    (qwen_alfworld, 'AlfWorld (Qwen2.5-7B)', axes[0]),
    (qwen_sciworld, 'SciWorld (Qwen2.5-7B)', axes[1]),
    (llama_alfworld, 'AlfWorld (Llama3.1-8B)', axes[2]),
    (llama_sciworld, 'SciWorld (Llama3.1-8B)', axes[3])
]

for idx, (data_dict, title, ax) in enumerate(datasets):
    methods = list(data_dict.keys())
    y_pos = np.arange(len(methods))
    
    # 绘制堆叠条形图
    left = np.zeros(len(methods))
    
    # 找出每个方法的最大值位置
    max_indices = {}
    for i, method in enumerate(methods):
        values = data_dict[method]
        max_indices[i] = np.argmax(values)
    
    for level_idx, level_name in enumerate(['Level 1', 'Level 2', 'Level 3', 'Level 4']):
        values = [data_dict[method][level_idx] for method in methods]
        
        # 添加纹理阴影
        bars = ax.barh(y_pos, values, left=left, color=colors[level_name], 
                       edgecolor='white', linewidth=2, label=level_name,
                       hatch=hatches[level_name], 
                       alpha=0.85
                       )
        
        # 只标注每行中最大的那个值
        for i, (bar, val) in enumerate(zip(bars, values)):
            if max_indices[i] == level_idx and val > 0:  # 只在最大值处标注
                x_pos = left[i] + val / 2
                ax.text(x_pos, i, f'{val:.1f}', 
                       ha='center', va='center', fontsize=fs-4, fontweight='bold',
                       color='white' if val > 20 else 'black')
        
        left += values
    
    # 设置y轴
    ax.set_yticks(y_pos)
    if idx == 0:
        # 只在第一个子图显示y轴标签
        ax.set_yticklabels(methods, fontsize=fs-1)
    else:
        # 其他子图不显示y轴标签
        ax.set_yticklabels([])
    ax.invert_yaxis()  # 反转y轴，使顺序从上到下
    
    # 设置x轴
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([0, 20, 40, 60, 80, 100], fontsize=fs-1)
    ax.set_title(f"{title}", fontsize=fs, pad=11)
    # 所有子图都显示xlabel
    ax.set_xlabel('Percentage (%)', fontsize=fs)
    
    # 移除网格
    ax.grid(False)
    
    # 移除所有边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 增大刻度字体，去掉刻度线
    ax.tick_params(axis='both', which='major', labelsize=fs-1, 
                   length=0, width=0)  # length=0 去掉刻度线

# 创建图例 - 添加纹理
legend_elements = [mpatches.Patch(facecolor=colors[level], edgecolor='white', 
                                  linewidth=2, label=level, hatch=hatches[level], 
                                  alpha=0.85) 
                   for level in ['Level 1', 'Level 2', 'Level 3', 'Level 4']]

fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
          bbox_to_anchor=(0.5, 1.13), fontsize=fs, frameon=True, framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.94], w_pad=4.0)
plt.savefig('/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/figures/figure_02/figure_02.png', dpi=300, bbox_inches='tight')
plt.savefig('/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/figures/figure_02/figure_02.pdf', dpi=300, bbox_inches='tight')
print("图表已保存")