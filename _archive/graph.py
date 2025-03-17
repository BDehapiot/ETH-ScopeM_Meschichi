#%% Imports -------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

#%% Parameters ----------------------------------------------------------------

# Path
# data_path = Path(Path.cwd(), "data", "local") 
data_path = Path("D:/local_Meschichi/data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")

# Path (selection)
tags_in0 = ["KA"]; tags_out0 = []
tags_str0 = f"{tags_in0}in_{tags_out0}out"
tags_in1 = ["KZ"]; tags_out1 = []
tags_str1 = f"{tags_in1}in_{tags_out1}out"

# Plot (selection)
plot_data = ["nVolume", "nMMRatio"] 
bins = 25

# Save
save = True

#%% Functions -----------------------------------------------------------------

def df_paths(root_path, tags_in, tags_out):
    paths = []
    for path in root_path.glob("**/*.csv"):
        if tags_in:
            check_tags_in = all(tag in str(path) for tag in tags_in)
        else:
            check_tags_in = True
        if tags_out:
            check_tags_out = not any(tag in str(path) for tag in tags_out)
        else:
            check_tags_out = True
        if check_tags_in and check_tags_out:
            paths.append(path)
    return paths

def df_merge(paths):
    nData, cData = [], []
    for path in paths:
        if "nData" in path.stem:
            nData.append(pd.read_csv(path))
        if "cData" in path.stem:
            cData.append(pd.read_csv(path))
    nData = pd.concat(nData, axis=0, ignore_index=True)
    cData = pd.concat(cData, axis=0, ignore_index=True)
    return nData, cData

#%% Execute -------------------------------------------------------------------

# Load, merge and save dataframes
nData0, cData0 = df_merge(df_paths(data_path, tags_in0, tags_out0))
nData1, cData1 = df_merge(df_paths(data_path, tags_in1, tags_out1))
if save:
    nData0.to_csv(data_path / f"nData_{tags_str0}.csv", index=False)
    cData0.to_csv(data_path / f"cData_{tags_str0}.csv", index=False)
    nData1.to_csv(data_path / f"nData_{tags_str1}.csv", index=False)
    cData1.to_csv(data_path / f"cData_{tags_str1}.csv", index=False)

# Plot
data0, data1 = [], []
for name in plot_data:
    if name is not None:
        if name[0] == "n": 
            data0.append(np.array(nData0[name]))
            data1.append(np.array(nData1[name]))
        if name[0] == "c": 
            data0.append(np.array(cData0[name]))
            data1.append(np.array(cData1[name]))

fig, axs = plt.subplots(2, 1, figsize=(8, 10))

if len(data0) == 1:
    
    # Plots
    axs[0].hist(data0[0], bins=bins)
    axs[0].set_title(f"{plot_data[0]}\n{tags_str0}")
    axs[0].set_xlabel(f"{plot_data[0]}")
    axs[0].set_ylabel("Occurrence")
    axs[1].hist(data1[0], bins=bins)
    axs[1].set_title(f"{plot_data[0]}\n{tags_str1}")
    axs[1].set_xlabel(f"{plot_data[0]}")
    axs[1].set_ylabel("Occurrence")

    # Axis limits
    x_min = min(np.nanmin(data0[0]), np.nanmin(data1[0]))
    x_max = max(np.nanmax(data0[0]), np.nanmax(data1[0]))
    y_max = max(
        np.histogram(data0[0], bins=bins)[0].max(), 
        np.histogram(data1[0], bins=bins)[0].max()
        )
    axs[0].set_xlim([x_min, x_max])
    axs[1].set_xlim([x_min, x_max])
    axs[0].set_ylim([0, y_max])
    axs[1].set_ylim([0, y_max])

elif len(data0) == 2:
    
    # Plots
    axs[0].scatter(data0[0], data0[1], s=10)
    axs[0].set_title(f"{plot_data[0]} vs. {plot_data[1]}\n{tags_str0}")
    axs[0].set_xlabel(f"{plot_data[0]}")
    axs[0].set_ylabel(f"{plot_data[1]}")
    axs[1].scatter(data1[0], data1[1], s=10)
    axs[1].set_title(f"{plot_data[0]} vs. {plot_data[1]}\n{tags_str1}")
    axs[1].set_xlabel(f"{plot_data[0]}")
    axs[1].set_ylabel(f"{plot_data[1]}")
    
    # Axis limits
    x_min = min(np.nanmin(data0[0]), np.nanmin(data1[0]))
    x_max = max(np.nanmax(data0[0]), np.nanmax(data1[0]))
    y_min = min(np.nanmin(data0[1]), np.nanmin(data1[1]))
    y_max = max(np.nanmax(data0[1]), np.nanmax(data1[1]))
    axs[0].set_xlim([x_min, x_max])
    axs[1].set_xlim([x_min, x_max])
    axs[0].set_ylim([y_min, y_max])
    axs[1].set_ylim([y_min, y_max])
    
plt.tight_layout()
plt.show()
