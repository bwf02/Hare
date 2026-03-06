import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 全局样式配置 ---
# HATCH_LIST = ['..', '||', 'xx', '++'] 
HATCH_LIST = [None, None, None, None] 

MODELS={"DeepSeek": "DeepSeek-16B", "MiniCPM": "MiniCPM-16B", "Mixtral": "Mixtral-2x11B", "qwen2_moe": "Qwen2-2.7B"}

COLORS = ['#A6CEE3', '#8267FF', '#FB9A99', '#D34F54']

def load_and_preprocess(prefix_path, file_names, labels):

    all_dfs = []
    
    for i, fname in enumerate(file_names):
        file_path = os.path.join(prefix_path, fname)
        if not os.path.exists(file_path):
            print(f"Error: 找不到文件 {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        
        # 预处理：按 model_name 去重，保留第一个出现的（如 Mixtral）
        df = df.drop_duplicates(subset=['model_name'], keep='first')
        df['version_label'] = labels[i]
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("没有加载到任何有效数据，请检查路径。")

    full_df = pd.concat(all_dfs)

    # 计算 Speedup: 以第一个 label (Naive) 为基准
    base_durations = all_dfs[0].set_index('model_name')['duration']
    
    def get_speedup(row):
        base_time = base_durations[row['model_name']]
        return base_time / row['duration']

    full_df['speedup'] = full_df.apply(get_speedup, axis=1)
    return full_df

def plot_ablation(full_df, output_path):
    """
    执行绘图逻辑并保存为PDF。
    """
    models = full_df['model_name'].unique()
    num_models = len(models)
    
    fig, axes = plt.subplots(1, num_models, figsize=(12, 4), sharey=True)
    
    if num_models == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        model_data = full_df[full_df['model_name'] == model].reset_index(drop=True)
        
        # 1. 统一设置 Y 轴范围为 0 到 3
        ax.set_ylim(0, 2.5)
        
        for j, row in model_data.iterrows():
            ax.bar(row['version_label'], row['speedup'], 
                width=0.65,  # <--- 这里设置宽度，例如 0.6
                color=COLORS[j], 
                hatch=HATCH_LIST[j], 
                edgecolor='black', 
                linewidth=1,
                alpha=0.9,
                zorder=3)
            
            # 数值标注：如果数值接近 3，稍微向下偏移以免出界
            text_y = row['speedup'] + 0.05
            if text_y > 2.9: text_y = row['speedup'] - 0.2
            
            ax.text(j, text_y, f"{row['speedup']:.2f} ×", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=4)

        ax.set_title(f"{MODELS[model]}", fontsize=14, pad=5)
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        # ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=12, rotation=20)

        # 核心逻辑：只有第一个子图保留 Y 轴刻度和标签
        if i == 0:
            ax.set_ylabel(r'Speedup Over $Hare_{naive}$.', fontsize=12)
            ax.tick_params(axis='y', labelsize=12)
            # 设置 Y 轴主刻度
            ax.set_yticks(np.arange(0, 3.1, 0.5))
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)
            # ax.spines['left'].set_visible(False)

    plt.tight_layout()

    save_dir = os.path.dirname(output_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved ablation plot to: {output_path}")

def main():
    config = {
        "prefix_path": "/root/code/fast-sparse/benchmark/ablation",
        "file_names": [
            'bw8padding_naive_pipeline.csv', 
            'bw8padding_hybrid_pipeline.csv', 
            'bw8paddingfree_hybrid_pipeline.csv', 
            'bw16paddingfree_hybrid_pipeline.csv'
        ],
        "labels": [r'$Hare_{naive}$', r'$Hare_{pipe}$', r'$Hare_{pf}$', r'$Hare_{full}$'],
        "output_file": "./figures/ablation.pdf"
    }

    try:
        data = load_and_preprocess(config["prefix_path"], config["file_names"], config["labels"])
        plot_ablation(data, config["output_file"])
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()