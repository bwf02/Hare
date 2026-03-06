import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASELINES = ['GEMM', 'megablocks', 'Samoyeds', 'vLLM', 'hare']
# HIDE_ANNOTATIONS = ['GEMM', 'megablocks', 'Samoyeds']
HIDE_ANNOTATIONS = BASELINES
LABELS = {
    'GEMM': 'Transformer',
    'megablocks': 'MegaBlocks',
    'Samoyeds': 'Samoyeds',
    'vLLM': 'vLLM',
    'hare': 'Hare',
}
COLORS = {
    'GEMM':       '#B2DF8A',  # 银灰色
    'megablocks': "#8267FF",  # 淡蓝
    'Samoyeds':   '#E58047',  # 淡绿
    'vLLM':       '#A6CEE3',  # 淡紫
    'hare': '#D34F54'   # 柔和的红：虽然柔和，但在冷色中依然显眼
}

# HATCHES = {
#     'hare': '//',    # 密集斜纹
#     'megablocks': '--',     # 反斜纹
#     'Samoyeds': 'x',       # 交叉纹
#     'vLLM': '\\',          # 竖纹
#     'GEMM': '..'           # 点纹 (为 GEMM 增加辨识度)
# }
HATCHES = {
    'hare': None,    # 密集斜纹
    'megablocks': None,     # 反斜纹
    'Samoyeds': None,       # 交叉纹
    'vLLM': None,          # 竖纹
    'GEMM': None           # 点纹 (为 GEMM 增加辨识度)
}


FONTSIZE=14

def preprocess_csv(df):
    """
    预处理数据：以 GEMM 为基准计算每个模型的Speedup
    """
    # 提取 GEMM 的时长作为基准字典
    gemm_map = df[df['kernel_name'] == 'GEMM'].set_index('model_name')['duration'].to_dict()
    
    # 计算 Speedup: Speedup = Duration_GEMM / Duration_Current
    def get_speedup(row):
        base_val = gemm_map.get(row['model_name'])
        if base_val and row['duration'] > 0:
            return base_val / row['duration']
        return 0

    df['speedup'] = df.apply(get_speedup, axis=1)
    return df

def plot_bar(df, ax, fontsize=FONTSIZE):
    
    models = sorted(df['model_name'].unique())
    n_models = len(models)
    n_baselines = len(BASELINES)
    
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

    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=fontsize, rotation=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    # ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=0.05)

    
    

if __name__ == "__main__":
    file_b32 = "../model/result/b1_seq512_model_flash.csv"
    file_b64 = "../model/result/b1_seq768_model_flash.csv"
    file_b128 = "../model/result/b1_seq1024_model_flash.csv"

    # 1. 创建画布 (1行3列)
    subfigure_args = {'left':0.025, 'right':0.99, 'top':0.88, 'bottom':0.14, 'wspace':0.06}
    figure_size = (20, 4.5)
    fig, axes = plt.subplots(1, 3, figsize=figure_size, sharey=False) # 如果需要Y轴刻度一致，改 sharey=True
    plt.subplots_adjust(**subfigure_args)

    raw_df = pd.read_csv(file_b32)
    processed_df = preprocess_csv(raw_df)
    ax = axes[0]
    plot_bar(processed_df, ax)
    max_y = 12
    ax.set_ylim(0, max_y)
    ticks = [0, 1] + list(range(2, max_y + 1, 2))
    ax.set_yticks(ticks)
    ax.text(
            0.02, 0.95,                
            f'Batch Size=32',
            transform=ax.transAxes,    
            ha='left',                 
            va='top',                  
            fontsize=FONTSIZE,
            fontweight='bold'
        )
    ax.set_ylabel('Speedup Over Transformer.', fontsize=FONTSIZE)

    raw_df = pd.read_csv(file_b64)
    processed_df = preprocess_csv(raw_df)
    ax = axes[1]
    plot_bar(processed_df, ax)
    max_y = 12
    ax.set_ylim(0, max_y)
    ticks = [0, 1] + list(range(2, max_y + 1, 2))
    ax.set_yticks(ticks)
    ax.text(
            0.02, 0.95,                
            f'Batch Size=64',
            transform=ax.transAxes,    
            ha='left',                 
            va='top',                  
            fontsize=FONTSIZE,
            fontweight='bold'
        )
    
    raw_df = pd.read_csv(file_b128)
    processed_df = preprocess_csv(raw_df)
    ax = axes[2]
    plot_bar(processed_df, ax)
    max_y = 12
    ax.set_ylim(0, max_y)
    ticks = [0, 1] + list(range(2, max_y + 1, 2))
    ax.set_yticks(ticks)
    ax.text(
            0.02, 0.95,                
            f'Batch Size=128',
            transform=ax.transAxes,    
            ha='left',                 
            va='top',                  
            fontsize=FONTSIZE,
            fontweight='bold'
        )

    fig.text(0.5, 0.01, "Model", ha='center', fontsize=FONTSIZE)

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
        bbox_to_anchor=(0.51, 1.01), # 调整这个坐标来移动图例
        ncol=len(BASELINES),                      # 横向显示 7 个
        fontsize=FONTSIZE,
    )

    save_path = 'figures/decode_model.pdf'
    plt.savefig(save_path, dpi=1000)
    print(f"Figure Save to : {save_path}")