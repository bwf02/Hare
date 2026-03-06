import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import os

FONTSIZE=16
# 自定义配色

CUSTOM_PALETTE = {
    'cuBLAS': "#C7C7C7",  
    'Sputnik': "#B2DF8A", 
    'Samoyeds': "#A6CEE3",
    'CLASP': "#E58047",   
    'Spatha': "#8267FF",  
    'SSD': "#FB9A99",
    'DSS': "#D34F54"
}
LABELS = ['cuBLAS', 'Sputnik', 'Samoyeds', 'CLASP', 'Spatha', 'Hare-SDSMM', 'Hare-SSDMM']
BASELINES = ['Sputnik', 'CLASP', 'Spatha', 'Samoyeds', 'SSD', 'DSS']

def process_standard_data(csv_path, group_col):
    if not os.path.exists(csv_path):
        print(f"Warning: 文件不存在 {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    
    # 1. 计算 Speedup
    for baseline in BASELINES:
        if baseline in df.columns:
            df[baseline] = df['cuBLAS'] / df[baseline]
    
    # 2. 转换格式
    df_melted = df.melt(id_vars=[group_col], value_vars=BASELINES, 
                        var_name='Method', value_name='Speedup')
    
    return df_melted

def draw_subplot(ax, df_melted, x_col, title, y_range=None, y_tricks=None):
    if df_melted is None:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        return

    width = 0.8
    linewidth = 2
    edge_color = '#333333'
    # 绘制箱线图
    sns.boxplot(
        x=x_col, 
        y='Speedup', 
        hue='Method', 
        data=df_melted,
        palette=CUSTOM_PALETTE,
        linewidth=1,         # 箱体的线宽
        width=width,
        gap=0.15,            # 注意：gap 参数需要 Seaborn v0.13.0+
        ax=ax,
        # 1. 修改边线颜色，不再使用纯黑
        boxprops=dict(edgecolor=edge_color, linewidth=0),
        whiskerprops=dict(color=edge_color, linewidth=linewidth),
        capprops=dict(color=edge_color, linewidth=linewidth),
        medianprops=dict(color='white', linewidth=linewidth), # 中位数保持白色
        
        # 2. 离群点美化：空心（白色填充）且边缘加粗
        fliersize=4,         # 适当调整离群点大小，方便看到空心效果
        flierprops=dict(
            marker='o',
            markerfacecolor='white',    # "空间用空白" -> 内部填充白色
            markeredgecolor=edge_color, # 边缘颜色与箱线一致
            markeredgewidth=2           # "曲线粗度为2" -> 边缘线宽
        ),
    )

    # 绘制点图（保持不变，或确保颜色协调）
    sns.pointplot(
        x=x_col, 
        y='Speedup', 
        hue='Method', 
        data=df_melted,
        palette=CUSTOM_PALETTE,
        linestyles="--",
        alpha=0.8,
        errorbar=None,
        ax=ax,
        dodge=width - width / len(CUSTOM_PALETTE), # 确保点图与箱线图对齐
        linewidth=1.5,
    )

    ax.set_yscale('symlog', linthresh=1, linscale=0.25)
    
    ax.axhline(y=1, color="black", linestyle='--', linewidth=2, alpha=0.9)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    if title:
        ax.set_title(title, fontsize=FONTSIZE, fontweight='bold', pad=10)
    if x_col == "NUM_EXPERTS":
        x_label = "Expert Counts"
    elif x_col == "K":
        x_label = "Hidden Size"
    elif x_col == "N":
        x_label = "Tokens"
    ax.set_xlabel(x_label, fontsize=FONTSIZE, labelpad=0.5)
    
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='y', which='both', labelsize=FONTSIZE-2)
    ax.tick_params(axis='x', which='both', labelsize=FONTSIZE-2)

    if y_range:
        ax.set_ylim(y_range)
    if y_tricks:
        ax.set_yticks(y_tricks)

    if ax.get_legend():
        ax.get_legend().remove()
        
    # sns.despine(ax=ax, trim=False, left=False)

if __name__ == "__main__":
    # 定义文件路径 (请修改为您实际的路径)
    path_vary_k = '../kernel/result/VaryK_5090.csv'  # 替换实际路径
    path_vary_n = '../kernel/result/VaryN_5090.csv'  # 替换实际路径
    path_vary_m = '../kernel/result/VaryNumExpert_5090.csv' # 实际路径

    # 1. 创建画布 (1行3列)
    subfigure_args = {'left':0.04, 'right':0.99, 'top':0.88, 'bottom':0.12, 'wspace':0.12}
    figure_size = (20, 4.2)    
    fig, axes = plt.subplots(1, 3, figsize=figure_size, sharey=False) # 如果需要Y轴刻度一致，改 sharey=True
    plt.subplots_adjust(**subfigure_args)

    # 处理m
    y_tricks = [0.1, 1, 4, 8, 16, 32, 45]
    y_range = [y_tricks[0], y_tricks[-1]]
    df_m = process_standard_data(path_vary_m, group_col='NUM_EXPERTS')
    draw_subplot(axes[0], df_m, x_col='NUM_EXPERTS', title=None, y_range=y_range, y_tricks=y_tricks)
    axes[0].set_ylabel('Speedup Over cuBLAS(log scale)', fontsize=FONTSIZE)
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[0].spines[spine].set_linewidth(1.2) # 稍微加粗，打印出来更清晰
        axes[0].spines[spine].set_color('#333333') # 使用深灰色比纯黑更高级

    # 2. 处理并绘制: 按 K 分组
    y_tricks = [0.1, 1, 4, 8, 16, 20, 24]
    y_range = [y_tricks[0], y_tricks[-1]]
    df_k = process_standard_data(path_vary_k, group_col='K')
    draw_subplot(axes[1], df_k, x_col='K', title=None, y_range=y_range, y_tricks=y_tricks)
    axes[1].set_ylabel('')
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[1].spines[spine].set_linewidth(1.2) # 稍微加粗，打印出来更清晰
        axes[1].spines[spine].set_color('#333333') # 使用深灰色比纯黑更高级

    # 3. 处理并绘制: 按 N 分组
    y_tricks = [0.1, 1, 4, 8, 16, 20, 24]
    y_range = [y_tricks[0], y_tricks[-1]]
    df_n = process_standard_data(path_vary_n, group_col='N')
    draw_subplot(axes[2], df_n, x_col='N', title=None, y_range=y_range, y_tricks=y_tricks)
    axes[2].set_ylabel('')
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[2].spines[spine].set_linewidth(1.2) # 稍微加粗，打印出来更清晰
        axes[2].spines[spine].set_color('#333333') # 使用深灰色比纯黑更高级

    legend_handles = [
        mpatches.Patch(
            facecolor=color,        # 色块填充颜色
            label=label,            # 标签文本
            edgecolor='black',      # [关键] 给色块加黑色边框
            linewidth=0.8           # [关键] 边框线条粗细
        )
        for label, color in zip(LABELS, CUSTOM_PALETTE.values())
    ]

    fig.legend(
        handles=legend_handles, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.01), # 调整这个坐标来移动图例
        ncol=len(LABELS),                      # 横向显示 7 个
        fontsize=14,
    )

    save_path = 'figures/synthetic_5090.pdf'
    plt.savefig(save_path, dpi=300)
    print(f"Figure Save to : {save_path}")
