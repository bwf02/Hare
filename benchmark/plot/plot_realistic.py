import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASELINES = ['cuBLAS', 'Sputnik', 'CLASP', 'Spatha', 'Samoyeds', 'SSD', 'DSS']
# HIDE_ANNOTATIONS = ['cuBLAS','Sputnik','Samoyeds','CLASP','Spatha', "SSD"]
HIDE_ANNOTATIONS = BASELINES
COLORS = {
    'cuBLAS': "#C7C7C7",  
    'Sputnik': "#B2DF8A", 
    'Samoyeds': "#A6CEE3",
    'CLASP': "#E58047",   
    'Spatha': "#8267FF",  
    'SSD': "#FB9A99",
    'DSS': "#D34F54"
}


LABLELS = {
    'cuBLAS': 'cuBLAS',
    'Sputnik': 'Sputnik',
    'Samoyeds': 'Samoyeds',
    'CLASP': 'CLASP',
    'Spatha': 'Spatha',
    'SSD': 'Hare-SDSMM',
    'DSS': 'Hare-SSDMM'
}

# HATCHES = {
#     'cuBLAS':   '..',    # 点状：作为基准，保留低调的背景感
#     'Sputnik':  '||',    # 竖线：简单清晰，与斜线形成对比
#     'Samoyeds': 'xx',    # 交叉：视觉密度较高，适合中间位置
#     'CLASP':    '++',    # 十字：类似网格，但比 x 更正，区分度高
#     'Spatha':   '\\\\',  # 反斜杠：经典的条纹样式
#     'SSD':      '--',    # 横线：与竖线对应
#     'DSS':      '//'     # 正斜杠：最常用的高亮样式，通常留给你的方法或最强Baseline
# }

HATCHES = {
    'cuBLAS':   None,    # 点状：作为基准，保留低调的背景感
    'Sputnik':  None,    # 竖线：简单清晰，与斜线形成对比
    'Samoyeds': None,    # 交叉：视觉密度较高，适合中间位置
    'CLASP':    None,    # 十字：类似网格，但比 x 更正，区分度高
    'Spatha':  None,  # 反斜杠：经典的条纹样式
    'SSD':      None,    # 横线：与竖线对应
    'DSS':     None     # 正斜杠：最常用的高亮样式，通常留给你的方法或最强Baseline
}

MODELS = ["DeepSeek", "MiniCPM", "Mixtral-2x11B", "Mixtral-8x22B", "Qwen2"]

def get_times(row, method):

    time_ns = row[method]
    if pd.isna(time_ns) or time_ns == 0:
        return np.nan

    return time_ns


def preprocess_csv(file_name):
    """
    Preprocess the CSV file and return a cleaned DataFrame.
    """
    df = pd.read_csv(file_name)

    # Extract unique (M, N, K) triples and create shape labels
    triples = df[['M','N','K']].drop_duplicates().reset_index(drop=True)
    # print(triples)
    shape_map = {tuple(x): f"M{idx}" for idx, x in enumerate(triples.values)}
    df['shape'] = df.apply(lambda r: shape_map[(r['M'], r['N'], r['K'])], axis=1)

    # Clean the dataframe by replacing zeros with NaN for baselines
    df_clean = df.copy()
    for c in BASELINES:
        df_clean[c] = df_clean[c].replace(0, np.nan)

    # Compute TFLOPS or Time for each baseline method
    for m in BASELINES:
        df_clean[f'{m}_value'] = df_clean.apply(
            lambda r: get_times(r, m),
            axis=1
        )

    # Create speedup dataframe relative to cuBLAS
    speedup_df = df_clean.copy()
    for m in BASELINES:
        speedup_df[m] = df_clean['cuBLAS_value'] / df_clean[f'{m}_value']

    speedup_df['cuBLAS'] = 1.0
    return speedup_df, triples, shape_map

def plot_bar(speedup_df, triples, shape_map, output_file, y_args, subfigure_args, figure_size, fontsize=14, legend=True):

    sparsities = sorted(speedup_df['Sparsity'].unique())
    shapes = list(triples.apply(lambda row: shape_map[(row['M'], row['N'], row['K'])], axis=1))
    n_shapes = len(shapes)
    n_spars = len(sparsities)
    n_baselines = len(BASELINES)

    y_limits, y_trickss = y_args
    
    fig, axes = plt.subplots(nrows=1, ncols=n_spars, figsize=figure_size)

    if n_spars == 1: axes = [axes]

    plt.subplots_adjust(**subfigure_args)

    group_spacing = 1.9  # 组与组之间的间距
    x = np.arange(n_shapes) * group_spacing
    bar_width = 0.22
    spacing = 0.02
    hide_set = set(HIDE_ANNOTATIONS)

    for ax, sp in zip(axes, sparsities):
        sdf = speedup_df[speedup_df['Sparsity'] == sp].set_index('shape')
        sdf = sdf.reindex(shapes)  # keep consistent order, may introduce NaN rows

        ax.set_ylim(y_limits[sp][0], y_limits[sp][-1])
        ax.set_yticks(y_trickss[sp])

        for i, b in enumerate(BASELINES):
            offsets = x - ((n_baselines - 1) * (bar_width + spacing)) / 2 + i * (bar_width + spacing)

            yvals = sdf[b].values
            bars = ax.bar(offsets, yvals, width=bar_width, 
                          label=LABLELS[b] if sp==sparsities[0] else None, 
                          color=COLORS[b], 
                          edgecolor='black', 
                          hatch=HATCHES.get(b, ''))

            # annotate values on top of bars (only finite numbers) unless baseline is hidden
            if b not in hide_set:
                for bar, val in zip(bars, yvals):
                    if np.isfinite(val):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01, f'{val:.2f}×',
                                ha='center', va='bottom', fontsize=fontsize, rotation=0, color=COLORS[b])

        ax.set_xticks(x)
        ax.set_xticklabels(shapes, rotation=0, fontsize=fontsize)

        ax.text(
            0.02, 0.95,                
            f'Sparsity {int(sp*100)}%',
            transform=ax.transAxes,    
            ha='left',                 
            va='top',                  
            fontsize=fontsize,
            fontweight='bold'
        )
        ax.grid(axis='y', linestyle='--', alpha=0.4)


    axes[0].set_ylabel('Speedup Over cuBLAS.', fontsize=fontsize)
    handles, labels = axes[0].get_legend_handles_labels()
    if legend: fig.legend(handles, labels, loc='upper center', ncol=len(BASELINES), fontsize=fontsize)
    fig.text(0.5, 0.02, 'Matrix ID', 
            ha='center', va='center', 
            fontsize=fontsize)

    # plt.axis('off')

    plt.savefig(output_file, dpi=300)
    print(f'Saved figure to {output_file}')

if __name__ == '__main__':

    file_name = '../kernel/result/realistic_Pro6000.csv'
    output_file = './figures/realistic_speedup_Pro6000.pdf'

    y_limits = {0.5: (0, 14, ), 0.75: (0, 24), 0.9: (0, 64)}
    y_trickss = {0.5: [1, 3, 4, 6, 8, 10, 12], 0.75: [1, 4, 6, 8, 10, 12, 14, 16, 18, 20], 0.9: [1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]}
    y_args = (y_limits, y_trickss)
    subfigure_args = {'left':0.025, 'right':0.99, 'top':0.88, 'bottom':0.125, 'wspace':0.06}
    figure_size = (22, 4.7)

    speedup_df, triples, shape_map = preprocess_csv(file_name)
    plot_bar(speedup_df, triples, shape_map, output_file, y_args, subfigure_args, figure_size, 16, True)

    file_name = '../kernel/result/realistic_5090.csv'
    output_file = './figures/realistic_speedup_5090.pdf'

    y_limits = {0.5: (0, 12), 0.75: (0, 20), 0.9: (0, 44)}
    y_trickss = {0.5: [1, 2, 4, 6, 8, 10, 12, 14], 0.75: [1, 4, 8, 12, 16, 20, 24], 0.9: [1, 8, 16, 24, 32, 40, 48, 56, 64]}
    y_args = (y_limits, y_trickss)
    subfigure_args = {'left':0.03, 'right':0.99, 'top':0.96, 'bottom':0.125, 'wspace':0.08}
    figure_size = (22, 4)

    speedup_df, triples, shape_map = preprocess_csv(file_name)
    plot_bar(speedup_df, triples, shape_map, output_file, y_args, subfigure_args, figure_size, 16, False)