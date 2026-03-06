import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

data = {
    'Input_tokens': ['32', '64', '128', '256', '2048', '4096'],
    'Gating':             [1.063, 0.774, 1.065, 1.248, 0.829, 0.988],
    'Gather':             [0.343, 0.264, 0.324, 0.393, 0.412, 0.535],
    'Topology':           [0.954, 0.744, 0.912, 1.191, 0.728, 0.883],
    'Expert Computation': [2.479, 2.252, 2.502, 2.638, 3.586, 5.787],
    'Scatter':            [0.276, 0.235, 0.320, 0.378, 0.356, 0.575]
}

df = pd.DataFrame(data)
df.set_index('Input_tokens', inplace=True)

# 计算百分比
df['Total'] = df.sum(axis=1)
df_pct = df.div(df['Total'], axis=0).drop(columns='Total')

# ============== 2. 严格顺序与配色定义 ==============

# 【关键】从下到上的顺序
stack_order = ['Scatter', 'Expert Computation', 'Topology', 'Gather', 'Gating']

# 自定义配色字典
colors_map = {
    'Gating':             "#FDB462",   # 暖橙色 (顶部)
    'Gather':             "#80B1D3",  # 柔和蓝 (IO)
    'Topology':           "#FB8072",  # 鲑鱼红 (瓶颈高亮)
    'Expert Computation': "#D9D9D9",  # 中性灰 (计算主体)
    'Scatter':            "#8DD3C7",  # 青瓷绿 (底部)

}

# 按照指定顺序筛选数据
df_plot = df_pct[stack_order]

# ============== 3. 绘图 ==============
fig, ax = plt.subplots(figsize=(10, 4.4))

# 初始化底部高度
bottom = np.zeros(len(df_plot))

# 循环绘制
for col in stack_order:
    ax.bar(df_plot.index, df_plot[col], bottom=bottom, label=col, 
           color=colors_map[col], 
           edgecolor='black', linewidth=0.6, # 细黑边增强质感
           width=0.65)
    bottom += df_plot[col]

# ============== 4. 装饰与标注 ==============

# Y轴设置
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.ylabel('Percentage (%)', fontsize=14, labelpad=-2)

# X轴设置
plt.xlabel('Input Tokens', fontsize=14, labelpad=-2)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], 
           bbox_to_anchor=(0.5, 1.12), loc='upper center', ncol=5, 
           frameon=False, fontsize=14, handlelength=1.5)

# 网格线
plt.grid(axis='y', linestyle='--', alpha=0.3, color='gray', zorder=0)
ax.set_axisbelow(True) # 确保网格线在柱子后面

# 数值标注 (自动适配颜色)
bottom_labels = np.zeros(len(df_plot))
for col in stack_order:
    values = df_plot[col].values
    for i, v in enumerate(values):
        # 阈值：占比 > 5% 才显示数字，避免文字重叠
        if v > 0.05: 
            # 字体颜色逻辑：除了浅灰色的 Computation 用黑色字，其他彩色背景用深色或白色视情况而定
            # 这里统一用黑色加粗，配合浅色系背景效果最好；如果是深色背景改为 'white'
            text_color = 'black' 
            # 如果背景色特别深(如 Topology 的红色)，可以考虑用白色
            if col == 'Topology': text_color = 'white' 
            
            ax.text(i, bottom_labels[i] + v/2, f'{v:.1%}', 
                    ha='center', va='center', fontsize=12, 
                    color=text_color, fontweight='bold')
    bottom_labels += values

# ============== 5. 保存 ==============
plt.tight_layout(pad=0.1)
plt.savefig("/root/code/fast-sparse/benchmark/plot/figures/deepseek_megablocks_breakdown.pdf", dpi=300)
