import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASELINES = ['GEMM', 'megablocks', 'Samoyeds', 'vLLM', 'hare']
HIDE_ANNOTATIONS = ['GEMM', 'megablocks', 'Samoyeds']

LABELS = {
    'GEMM': 'Transformer',
    'megablocks': 'MegaBlocks',
    'Samoyeds': 'Samoyeds',
    'vLLM': 'vLLM',
    'hare': 'Hare',
}
COLORS = {
    'GEMM':       '#C7C7C7',  # 银灰色
    'megablocks': '#8DD3C7',  # 淡蓝
    'Samoyeds':   '#B2DF8A',  # 淡绿
    'vLLM':       '#A6CEE3',  # 淡紫
    'hare': '#FB9A99'   # 柔和的红：虽然柔和，但在冷色中依然显眼
}
HATCHES = {
    'hare': '//',  
    'megablocks': '--',  
    'Samoyeds': 'x',     
    'vLLM': '\\',        
    'GEMM': '..'         
}

def preprocess_csv(df):
    """
    预处理数据：以 GEMM 为基准计算每个模型的 Speedup
    """
    # 提取 GEMM 的时长作为基准字典
    gemm_map = df[df['kernel_name'] == 'GEMM'].set_index('model_name')['duration'].to_dict()
    
    # 计算 Speedup: Speedup = Duration_GEMM / Duration_Current
    def get_speedup(row):
        base_val = gemm_map.get(row['model_name'])
        if base_val and row['duration'] > 0:
            return base_val / row['duration']
        return np.nan

    df['speedup'] = df.apply(get_speedup, axis=1)
    return df

def plot_bar(df, output_file, figure_size=(5, 5), subfigure_args=None, xlabel=None, fontsize=12):
    
    models = sorted(df['model_name'].unique())
    n_models = len(models)
    n_baselines = len(BASELINES)
    
    fig, ax = plt.subplots(figsize=figure_size)
    plt.subplots_adjust(**subfigure_args)
    
    group_spacing = 1.5  # 组与组之间的间距
    x = np.arange(n_models) * group_spacing
    bar_width = 0.2
    spacing = 0.02
    
    # 遍历每个 baseline 绘制柱状图
    for i, b in enumerate(BASELINES):
        # 计算每个柱子的偏移位置
        offsets = x - ((n_baselines - 1) * (bar_width + spacing)) / 2 + i * (bar_width + spacing)
        
        # 提取当前 kernel 在各模型下的 speedup 值
        yvals = []
        for m in models:
            val = df[(df['model_name'] == m) & (df['kernel_name'] == b)]['speedup'].values
            yvals.append(val[0] if len(val) > 0 else np.nan)
        
        bars = ax.bar(offsets, yvals, width=bar_width, label=LABELS[b], color=COLORS[b], edgecolor='black', hatch=HATCHES.get(b, ''))
        
        for bar, val in zip(bars, yvals):
            if np.isfinite(val) and b not in HIDE_ANNOTATIONS:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f'{val:.2f}x', ha='center', va='bottom', 
                        fontsize=fontsize-2, color=COLORS[b])

    ax.set_ylabel('Speedup Over Transformer', fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(0, 5)
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=0.05)
    # 图例放在上方
    legend_handles = [
        mpatches.Patch(
            facecolor=color,        # 色块填充颜色
            label=LABELS[label],            # 标签文本
            edgecolor='black',      # [关键] 给色块加黑色边框
            linewidth=0.8,           # [关键] 边框线条粗细
            hatch=HATCHES.get(label, '')
        )
        for label, color in COLORS.items()
    ]

    fig.legend(
        handles=legend_handles, 
        loc='upper center', 
        bbox_to_anchor=(0.525, 1), # 调整这个坐标来移动图例
        ncol=len(BASELINES),                      # 横向显示 7 个
        fontsize=11,
    )
    plt.savefig(output_file, dpi=1000)
    print(f"Successfully saved plot to {output_file}")

if __name__ == "__main__":
    file_name = "../model/result/b1_seq4096_mlp_flash.csv"
    output_file = "./figures/prefill_mlp_H800.pdf"
    raw_df = pd.read_csv(file_name)
    
    # 1. 预处理
    processed_df = preprocess_csv(raw_df)
    
    # 2. 绘图
    figure_size = (8, 3.3)
    subfigure_args = {'left':0.06, 'right':0.99, 'top':0.88, 'bottom':0.13, 'wspace':0.08}
    xlabel = "Model"
    plot_bar(processed_df, output_file, figure_size, subfigure_args, xlabel)