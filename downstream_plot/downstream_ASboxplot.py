import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
parser = argparse.ArgumentParser(description='metrics')
parser.add_argument("--document", type=str, default='dataset45_ML')
args = parser.parse_args()

Data =args.document #

data_path = 'result/' + Data
save_path = 'downstream_result/AS/' + Data
if not os.path.exists(save_path):
    os.makedirs(save_path)
methods = ['SpaDiT', 'Tangram', 'scVI', 'SpaGE', 'stPlus', 'SpaOTsc', 'novoSpaRc', 'SpatialScope', 'stDiff']
file_paths = [f"{data_path}/{Data}_{method}_gene_result.csv" for method in methods]


data_as = {}
for method, file_path in zip(methods, file_paths):
    try:
        data = pd.read_csv(file_path)
        if data['AS'].dtype.kind not in 'biufc':
            print(f"{method}的AS列包含非数值数据")
        data_as[method] = data['AS'].dropna()
    except Exception as e:
        print(f"无法加载{method}的数据: {e}")

colors = ['#1f77b4', '#ebb1c9', '#e1d64c', '#c0c738', '#82b679', '#bcadcd', '#a8c6c9', '#8bcae3', '#9b95bc']

fig, ax = plt.subplots(figsize=(10, 10))
bp = ax.boxplot([data_as[method] for method in methods if method in data_as], patch_artist=True, notch=False, labels=methods,
                boxprops=dict(linewidth=0.1, facecolor='none'),
                medianprops={'color': 'black', 'linewidth': 0.2},
                whiskerprops={'linewidth': 0.2},
                capprops={'linewidth': 0.2}
                )


for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)


for spine in ax.spines.values():
    spine.set_linewidth(0.2)


ax.xaxis.set_tick_params(width=0.2)
ax.yaxis.set_tick_params(width=0.2)

ax.set_title(f'{Data}')
ax.set_ylabel('AS')
ax.set_xlabel('Method')
plt.xticks(rotation=45)
plt.savefig(os.path.join(save_path, Data + '.pdf'))
plt.close()