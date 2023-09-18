#%% Imports -------------------------------------------------------------------

import nd2
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

#%% Parameters ----------------------------------------------------------------

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path(Path.cwd(), 'data', 'local')

# Open data
data = []
for path in data_path.iterdir():
    if "nd2" in path.name:
        with nd2.ND2File(path) as ndfile:
            data.append(ndfile.asarray())    
            
#%%

for stack in data:
    print(
        # f'min = {np.min(stack)} | '
        # f'max = {np.max(stack)} | '
        # f'median = {np.median(stack)} | '
        f'mean = {np.mean(stack):.2f} | '
        f'sd = {np.std(stack):.2f} | '
        f'ratio = {np.mean(stack) / np.std(stack):.2f}'
        )
            
# # Z-profiles
# zprof = []
# for stack in data:
#     zprof.append(np.mean(stack, axis=(1,2)))
    
# zprof = np.array(zprof)
# zprof_avg = np.mean(zprof, axis=0)
# plt.plot(zprof_avg)
    
# for i, vector in enumerate(zprof):
#     plt.plot(vector, label=f'zprof {i+1}')
    
            
#%%
            
# values = np.concatenate([stack.flatten() for stack in data])
# # Plot the histogram
# plt.hist(values, bins=1000, edgecolor='black')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of Values in 3D Arrays')
# plt.show()

# print(np.percentile(values, 99.9))
