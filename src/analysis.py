import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 任务长度分类
TASK_CATEGORIES = {
    "boil": "L",
    "melt": "L",
    "freeze": "L",
    "change-the-state-of-matter-of": "L",
    "use-thermometer": "M",
    "measure-melting-point-known-substance": "M",
    # "measure-melting-point-unknown-substance": "L",
    "power-component": "S",
    "power-component-renewable-vs-nonrenewable-energy": "M",
    "test-conductivity": "M",
    "test-conductivity-of-unknown-substances": "M",
    "find-living-thing": "S",
    "find-non-living-thing": "S",
    "find-plant": "S",
    "find-animal": "S",
    "grow-plant": "L",
    "grow-fruit": "L",
    "chemistry-mix": "M",
    "chemistry-mix-paint-secondary-color": "S",
    "chemistry-mix-paint-tertiary-color": "M",
    "lifespan-longest-lived": "S",
    "lifespan-shortest-lived": "S",
    "lifespan-longest-lived-then-shortest-lived": "S",
    "identify-life-stages-1": "M",
    "identify-life-stages-2": "S",
    "inclined-plane-determine-angle": "L",
    "inclined-plane-friction-named-surfaces": "L",
    "inclined-plane-friction-unnamed-surfaces": "L",
    "mendelian-genetics-known-plant": "L",
    "mendelian-genetics-unknown-plant": "L"
}

# 长度标签映射
LENGTH_LABELS = {
    'S': 'Short',
    'M': 'Medium', 
    'L': 'Long'
}

def load_data(file_paths):
    """加载多个模式的数据文件，并确保长度一致（以最短为准）
    
    Args:
        file_paths: dict, 格式为 {'mode_name': 'file_path'}
    """
    data = {}
    min_length = float('inf')
    
    # 第一步：加载所有数据并找到最短长度
    print("=== 正在加载数据并检查长度 ===")
    for mode_name, file_path in file_paths.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            mode_data = json.load(f)
        data[mode_name] = mode_data
        
        # 检查数据长度
        if 'scores' in mode_data:
            data_length = len(mode_data['scores'])
            print(f"{mode_name}: {data_length} 条数据")
            min_length = min(min_length, data_length)
    
    print(f"最短数据长度: {min_length}")
    
    # 第二步：截取所有数据到最短长度
    if min_length < float('inf'):
        print(f"=== 将所有数据截取到 {min_length} 条 ===")
        for mode_name in data:
            original_length = len(data[mode_name]['scores'])
            if original_length > min_length:
                print(f"{mode_name}: 从 {original_length} 截取到 {min_length}")
                data[mode_name]['scores'] = data[mode_name]['scores'][:min_length]
                data[mode_name]['task_names'] = data[mode_name]['task_names'][:min_length]
                data[mode_name]['variations'] = data[mode_name]['variations'][:min_length]
            else:
                print(f"{mode_name}: 保持原长度 {original_length}")
    
    return data

def create_comparison_dataframe(data_dict):
    """创建多模式对比DataFrame
    
    Args:
        data_dict: dict, 包含各模式数据的字典
    """
    # 获取第一个模式的数据作为基础
    first_mode = list(data_dict.keys())[0]
    base_data = data_dict[first_mode]
    
    # 验证所有模式数据长度是否一致
    base_length = len(base_data['scores'])
    print(f"\n=== 验证数据长度一致性 ===")
    print(f"基准长度: {base_length}")
    
    for mode_name, mode_data in data_dict.items():
        current_length = len(mode_data['scores'])
        print(f"{mode_name}: {current_length}")
        if current_length != base_length:
            raise ValueError(f"数据长度不一致: {mode_name} 有 {current_length} 条数据，期望 {base_length} 条")
    
    print("✓ 所有模式数据长度一致")
    
    # 创建基础DataFrame
    df_data = {
        'task_name': base_data['task_names'],
        'variation': base_data['variations']
    }
    
    # 添加每个模式的分数
    for mode_name, mode_data in data_dict.items():
        df_data[f'{mode_name}_score'] = mode_data['scores']
    
    df = pd.DataFrame(df_data)
    
    # 添加任务长度信息
    df['task_length'] = df['task_name'].map(TASK_CATEGORIES)
    df['task_length_label'] = df['task_length'].map(LENGTH_LABELS)
    
    # 修正胜率计算逻辑：只有严格大于所有其他模式才算获胜
    mode_names = list(data_dict.keys())
    score_columns = [f'{mode}_score' for mode in mode_names]
    
    def find_winner(row):
        """找到获胜者，必须严格大于所有其他模式"""
        scores = {mode: row[f'{mode}_score'] for mode in mode_names}
        max_score = max(scores.values())
        
        # 找到所有达到最高分的模式
        max_modes = [mode for mode, score in scores.items() if score == max_score]
        
        # 只有当只有一个模式达到最高分时，该模式才获胜
        if len(max_modes) == 1:
            return max_modes[0]
        else:
            return 'tie'  # 平局
    
    df['winner'] = df.apply(find_winner, axis=1)
    df['best_score'] = df[score_columns].max(axis=1)  # 改名避免混淆
    
    # 计算模式间的差异
    for i in range(len(mode_names)):
        for j in range(i+1, len(mode_names)):
            mode1, mode2 = mode_names[i], mode_names[j]
            diff_col = f'{mode2}_vs_{mode1}_diff'
            df[diff_col] = df[f'{mode2}_score'] - df[f'{mode1}_score']
    
    return df

def analyze_by_task_type(df):
    """按任务类型分组分析性能"""
    # 严格定义score列
    mode_names = []
    for col in df.columns:
        if col.endswith('_score') and not col in ['max_score', 'best_score']:
            mode_name = col.replace('_score', '')
            if mode_name not in ['max', 'best', 'min', 'avg', 'mean']:
                mode_names.append(mode_name)
    
    score_columns = [f'{mode}_score' for mode in mode_names]
    
    # 基础统计
    agg_dict = {}
    for col in score_columns:
        agg_dict[col] = ['mean', 'std', 'count']
    
    task_summary = df.groupby('task_name').agg(agg_dict).round(2)
    
    # 扁平化列名
    task_summary.columns = ['_'.join(col).strip() for col in task_summary.columns]
    
    # 添加获胜者统计（包括平局）
    winner_stats = df.groupby('task_name')['winner'].value_counts().unstack(fill_value=0)
    task_summary = pd.concat([task_summary, winner_stats], axis=1)
    
    # 添加任务长度信息
    task_length_info = df.groupby('task_name')['task_length_label'].first()
    task_summary['task_length'] = task_length_info
    
    return task_summary

def analyze_by_task_length_improved(df):
    """改进版：按任务长度分组分析，先计算任务级平均值
    
    这个函数先对每个任务类型计算平均分，然后再按任务长度分组
    """
    # 严格定义score列：必须是模式名_score格式，排除max_score, best_score等
    # 从data_dict的键中获取模式名称更可靠
    mode_names = []
    for col in df.columns:
        if col.endswith('_score') and not col in ['max_score', 'best_score']:
            mode_name = col.replace('_score', '')
            # 额外检查：mode_name不应该是'max'或'best'
            if mode_name not in ['max', 'best', 'min', 'avg', 'mean']:
                mode_names.append(mode_name)
    
    score_columns = [f'{mode}_score' for mode in mode_names]
    
    print(f"检测到的模式: {mode_names}")  # 调试信息
    print(f"分数列: {score_columns}")  # 调试信息
    
    # 第一步：计算每个任务类型的平均分
    task_avg_scores = df.groupby(['task_name', 'task_length_label'])[score_columns].mean().reset_index()
    
    # 第二步：基于任务类型平均分，按任务长度分组统计
    # 注意：这里每个任务类型的权重相等，而不是每个任务实例权重相等
    length_analysis = task_avg_scores.groupby('task_length_label')[score_columns].agg(['mean', 'std']).round(2)
    
    # 添加任务类型计数
    task_counts = task_avg_scores.groupby('task_length_label')['task_name'].nunique()
    for col in score_columns:
        length_analysis[(col, 'task_count')] = task_counts
    
    # 计算每个任务类型的winner（基于任务平均分）
    def find_task_winner(row):
        """基于行数据找到获胜的模式"""
        max_score = -1
        winner = None
        tie = False
        
        for mode in mode_names:
            score = row[f'{mode}_score']
            if score > max_score:
                max_score = score
                winner = mode
                tie = False
            elif score == max_score:
                tie = True
        
        return 'tie' if tie else winner
    
    task_avg_scores['task_winner'] = task_avg_scores.apply(find_task_winner, axis=1)
    
    # 调试：查看任务获胜者分布
    print("\n任务获胜者分布:")
    print(task_avg_scores['task_winner'].value_counts())
    
    # 胜利统计（基于任务类型，而不是任务实例）
    length_win_counts = task_avg_scores.groupby('task_length_label')['task_winner'].value_counts().unstack(fill_value=0)
    
    # 确保所有模式都在列中（即使某个模式从未获胜）
    for mode in mode_names:
        if mode not in length_win_counts.columns:
            length_win_counts[mode] = 0
    
    # 如果有tie列，保留它
    if 'tie' not in length_win_counts.columns:
        length_win_counts['tie'] = 0
    
    # 重新排序列，把tie放在最后
    cols = [mode for mode in mode_names if mode in length_win_counts.columns] + ['tie']
    length_win_counts = length_win_counts[cols]
    
    # 计算胜率：胜利任务类型数 / 总任务类型数
    length_win_rates = length_win_counts.div(length_win_counts.sum(axis=1), axis=0) * 100
    length_win_rates = length_win_rates.round(1)
    
    # 计算平均分数（基于任务类型平均）
    length_avg_scores = task_avg_scores.groupby('task_length_label')[score_columns].mean()
    
    # 创建详细报告
    print("\n=== 改进版任务长度分析（基于任务类型平均） ===")
    for length in ['Short', 'Medium', 'Long']:
        if length not in task_avg_scores['task_length_label'].values:
            continue
        
        length_tasks = task_avg_scores[task_avg_scores['task_length_label'] == length]
        print(f"\n{length} 任务：")
        print(f"  包含 {len(length_tasks)} 个任务类型")
        print(f"  任务类型: {', '.join(length_tasks['task_name'].unique())}")
        
        # 打印每个模式的平均分
        print(f"  各模式平均分（基于任务类型）:")
        for col in score_columns:
            mode_name = col.replace('_score', '')
            avg_score = length_avg_scores.loc[length, col]
            print(f"    {mode_name}: {avg_score:.2f}")
        
        # 打印胜率
        print(f"  各模式胜率（基于任务类型）:")
        for mode in mode_names:
            if mode in length_win_rates.columns:
                win_rate = length_win_rates.loc[length, mode]
                print(f"    {mode}: {win_rate:.1f}%")
        if 'tie' in length_win_rates.columns:
            tie_rate = length_win_rates.loc[length, 'tie']
            print(f"    平局: {tie_rate:.1f}%")
    
    return length_analysis, length_win_rates, length_avg_scores, length_win_counts, task_avg_scores

def overall_statistics_improved(df):
    """改进版总体统计：考虑任务类型权重"""
    # 严格定义score列
    mode_names = []
    for col in df.columns:
        if col.endswith('_score') and not col in ['max_score', 'best_score']:
            mode_name = col.replace('_score', '')
            if mode_name not in ['max', 'best', 'min', 'avg', 'mean']:
                mode_names.append(mode_name)
    
    score_columns = [f'{mode}_score' for mode in mode_names]
    
    # 先计算每个任务类型的平均分
    task_avg_scores = df.groupby('task_name')[score_columns].mean()
    
    stats = {
        'total_instances': len(df),
        'total_task_types': df['task_name'].nunique(),
        'task_length_distribution': df.groupby('task_length_label')['task_name'].nunique().to_dict()
    }
    
    # 各模式平均分（基于任务类型平均）
    for col in score_columns:
        mode_name = col.replace('_score', '')
        # 基于任务实例的平均分
        stats[f'{mode_name}_avg_score_instance'] = df[col].mean()
        # 基于任务类型的平均分
        stats[f'{mode_name}_avg_score_task'] = task_avg_scores[col].mean()
    
    # 计算基于任务类型的胜率
    def find_task_type_winner(row):
        max_score = -1
        winner = None
        tie = False
        
        for mode in mode_names:
            score = row[f'{mode}_score']
            if score > max_score:
                max_score = score
                winner = mode
                tie = False
            elif score == max_score:
                tie = True
        
        return 'tie' if tie else winner
    
    task_winners = task_avg_scores.apply(find_task_type_winner, axis=1)
    
    winner_counts = task_winners.value_counts()
    total_tasks = df['task_name'].nunique()
    
    for mode in mode_names:
        win_count = winner_counts.get(mode, 0)
        stats[f'{mode}_task_wins'] = win_count
        stats[f'{mode}_task_win_rate'] = (win_count / total_tasks) * 100
    
    # 平局统计
    tie_count = winner_counts.get('tie', 0)
    stats['task_ties'] = tie_count
    stats['task_tie_rate'] = (tie_count / total_tasks) * 100
    
    return stats

def visualize_results_improved(df, length_win_rates, length_avg_scores, task_avg_scores):
    """改进版可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 获取mode名称（不包括tie）
    mode_columns = [col for col in length_win_rates.columns if col != 'tie']
    
    # 1. 基于任务类型的总体胜率分布
    task_winners = task_avg_scores.drop_duplicates('task_name')['task_winner'].value_counts()
    axes[0, 0].pie(task_winners.values, labels=task_winners.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Overall Win Rate by Task Type')
    
    # 2. 按任务长度的胜率分布（基于任务类型）- 只显示各个mode，不显示tie
    # 创建一个新的DataFrame只包含mode的胜率
    mode_win_rates = length_win_rates[mode_columns]
    mode_win_rates.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Mode Win Rates by Task Length (Task-Type Weighted) %')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].legend(title='Thinking Mode')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加tie率作为文本注释
    if 'tie' in length_win_rates.columns:
        for i, length in enumerate(length_win_rates.index):
            tie_rate = length_win_rates.loc[length, 'tie']
            axes[0, 1].text(i, mode_win_rates.loc[length].max() + 2, 
                          f'Tie: {tie_rate:.1f}%', 
                          ha='center', va='bottom', fontsize=9)
    
    # 3. 按任务长度的平均分数（基于任务类型）
    length_avg_scores.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Average Scores by Task Length (Task-Type Weighted)')
    axes[1, 0].set_ylabel('Average Score')
    axes[1, 0].legend(title='Thinking Mode')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 任务类型级别的分数分布
    score_columns = [col for col in task_avg_scores.columns if col.endswith('_score')]
    task_scores_melted = task_avg_scores[score_columns + ['task_name']].melt(
        id_vars='task_name', var_name='mode', value_name='score'
    )
    task_scores_melted['mode'] = task_scores_melted['mode'].str.replace('_score', '')
    
    sns.boxplot(data=task_scores_melted, x='mode', y='score', ax=axes[1, 1])
    axes[1, 1].set_title('Task-Type Score Distribution by Mode')
    axes[1, 1].set_ylabel('Average Score per Task Type')
    
    plt.tight_layout()
    plt.savefig('mode_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 配置文件路径 - 请根据实际情况修改
    file_paths = {
        'mode1': '/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/gpt-4o_mode1/1749191441/final_result.json',
        'mode2': '/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/gpt-4o_mode2/1749199883/final_result.json', 
        'mode3': '/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/gpt-4o_mode3/1749199814/final_result.json',
        'mode4': '/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/gpt-4o_mode4/1749191648/final_result.json'
    }
    print("=== 开始加载数据 ===")
    # 加载数据
    data_dict = load_data(file_paths)
    
    # 创建对比DataFrame
    df = create_comparison_dataframe(data_dict)
    
    # 各种分析
    task_analysis = analyze_by_task_type(df)
    
    # 使用改进版的任务长度分析
    length_analysis, length_win_rates, length_avg_scores, length_win_counts, task_avg_scores = analyze_by_task_length_improved(df)
    
    # 使用改进版的总体统计
    overall_stats = overall_statistics_improved(df)
    
    # 显示结果
    print("\n=== 改进版总体统计 ===")
    print("基于任务实例的统计:")
    print(f"  总任务实例数: {overall_stats['total_instances']}")
    print(f"  总任务类型数: {overall_stats['total_task_types']}")
    
    print("\n基于任务类型的平均分数:")
    for mode in data_dict.keys():
        instance_avg = overall_stats[f'{mode}_avg_score_instance']
        task_avg = overall_stats[f'{mode}_avg_score_task']
        print(f"  {mode}:")
        print(f"    实例平均: {instance_avg:.2f}")
        print(f"    任务类型平均: {task_avg:.2f}")
    
    print("\n基于任务类型的胜率:")
    for mode in data_dict.keys():
        wins = overall_stats.get(f'{mode}_task_wins', 0)
        win_rate = overall_stats.get(f'{mode}_task_win_rate', 0)
        print(f"  {mode}: {wins}/{overall_stats['total_task_types']} ({win_rate:.1f}%)")
    
    print("\n=== 改进版按任务长度分析 ===")
    print("\n按任务长度的胜率（基于任务类型）:")
    print(length_win_rates)
    
    print("\n按任务长度的平均分数（基于任务类型）:")
    print(length_avg_scores)
    
    print("\n按任务长度的任务类型分布:")
    length_task_dist = df.groupby('task_length_label')['task_name'].nunique()
    print(length_task_dist)
    
    # 保存结果
    df.to_csv('mode_comparison_detail.csv', index=False, encoding='utf-8')
    task_analysis.to_csv('task_analysis_summary.csv', encoding='utf-8')
    task_avg_scores.to_csv('task_type_averages.csv', index=False, encoding='utf-8')
    length_win_rates.to_csv('length_win_rates.csv', encoding='utf-8')
    length_avg_scores.to_csv('length_avg_scores.csv', encoding='utf-8')
    
    print(f"\n结果已保存到以下文件:")
    print(f"- 详细对比数据: mode_comparison_detail.csv")
    print(f"- 任务类型分析: task_analysis_summary.csv")
    print(f"- 任务类型平均分: task_type_averages.csv")
    print(f"- 改进版任务长度胜率: length_win_rates.csv")
    print(f"- 改进版任务长度平均分: length_avg_scores.csv")
    
    # 创建改进版可视化
    visualize_results_improved(df, length_win_rates, length_avg_scores, task_avg_scores)
    
    return df, task_analysis, length_analysis, length_win_rates, length_avg_scores, task_avg_scores

def analyze_best_mode_by_length_improved(df, length_win_rates, length_avg_scores):
    """改进版：分析各任务长度下哪种模式表现最好"""
    print("\n=== 改进版：各任务长度最佳模式分析（基于任务类型） ===")
    
    # 严格定义score列和mode名称
    mode_names = []
    for col in df.columns:
        if col.endswith('_score') and not col in ['max_score', 'best_score']:
            mode_name = col.replace('_score', '')
            if mode_name not in ['max', 'best', 'min', 'avg', 'mean']:
                mode_names.append(mode_name)
    
    score_columns = [f'{mode}_score' for mode in mode_names]
    
    for length in ['Short', 'Medium', 'Long']:
        if length not in length_avg_scores.index:
            continue
            
        print(f"\n{length} 任务:")
        
        # 平均分数
        print("  平均分数（基于任务类型）:")
        scores_dict = {}
        for col in score_columns:
            mode_name = col.replace('_score', '')
            score = length_avg_scores.loc[length, col]
            scores_dict[mode_name] = score
            print(f"    {mode_name}: {score:.3f}")
        
        # 胜率
        print("  胜率（基于任务类型）:")
        for mode in mode_names:
            if mode in length_win_rates.columns:
                win_rate = length_win_rates.loc[length, mode]
                print(f"    {mode}: {win_rate:.1f}%")
        
        if 'tie' in length_win_rates.columns:
            tie_rate = length_win_rates.loc[length, 'tie']
            print(f"    平局: {tie_rate:.1f}%")
        
        # 找出最佳模式
        best_score_mode = max(scores_dict.items(), key=lambda x: x[1])[0]
        
        # 找出胜率最高的模式（不包括tie）
        mode_win_rates = {mode: length_win_rates.loc[length, mode] 
                         for mode in mode_names if mode in length_win_rates.columns}
        if mode_win_rates:
            best_win_mode = max(mode_win_rates.items(), key=lambda x: x[1])[0]
            print(f"  最高平均分: {best_score_mode} ({scores_dict[best_score_mode]:.3f})")
            print(f"  最高胜率: {best_win_mode} ({mode_win_rates[best_win_mode]:.1f}%)")
        
        # 分析差异
        if len(mode_names) == 2:
            mode1, mode2 = mode_names[0], mode_names[1]
            score_diff = abs(scores_dict[mode1] - scores_dict[mode2])
            print(f"  分数差异: {score_diff:.3f}")

if __name__ == "__main__":
    # 使用示例数据路径 - 实际使用时请修改为您的文件路径
    file_paths = {
        'mode1': '/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/gpt-4o_mode1/1749191441/final_result.json',
        'mode2': '/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/gpt-4o_mode2/1749199883/final_result.json', 
        'mode3': '/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/gpt-4o_mode3/1749199814/final_result.json',
        'mode4': '/apdcephfs_cq11/share_1567347/share_info/rhyang/AdaAgent/results/test/sciworld/gpt-4o_mode4/1749191648/final_result.json'
    }
    
    try:
        df, task_analysis, length_analysis, length_win_rates, length_avg_scores, task_avg_scores = main()
        
        # 使用改进版分析
        analyze_best_mode_by_length_improved(df, length_win_rates, length_avg_scores)
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请修改 file_paths 字典中的文件路径为实际路径")
    except Exception as e:
        print(f"执行出错: {e}")