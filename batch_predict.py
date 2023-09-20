#%% Imports -------------------------------------------------------------------

import gc
import nd2
import time
import numpy as np
import pandas as pd
from skimage import io
from scipy import stats
from pathlib import Path
import segmentation_models as sm
from skimage.measure import label
from joblib import Parallel, delayed 
from skimage.transform import downscale_local_mean
from skimage.segmentation import clear_border
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, white_tophat, disk

#%% Initialize ----------------------------------------------------------------

# Get paths
data_path = Path(Path.cwd(), 'data', 'local')
nd2_paths = []
for path in data_path.iterdir():
    if path.suffix == '.nd2':
        nd2_paths.append(path)
        
# Model
model_name = 'model_weights_0.5.h5'
rescale_factor = float(model_name[14:17])

# Create stack_data dict
stack_data = {
    "name": [], "path": [], "voxel_size": [], "voxel_ratio": [],
    "stack": [], "stack_norm": [], "tophat": [], "probs": [],
    "nMasks": [], "nLabels": [], "cMasks": [],
    "results": [],
    }

#%% Parameters ----------------------------------------------------------------

# Nuclei nMasks
prob_min = 0.5
prob_sigma = 2 * rescale_factor
clear_nBorder = False
min_nSize = 4096 * rescale_factor

# Chromocenter nMasks 
tophat_size = 4 * rescale_factor
tophat_sigma = 2 * rescale_factor
tophat_tresh_coeff = 1
min_cSize = 32 * rescale_factor
        
#%% Functions -----------------------------------------------------------------

def pre_process(path):
    
    # Open data & metadata
    name = path.stem
    with nd2.ND2File(path) as ndfile:
        stack = ndfile.asarray()
        voxel_size = list(ndfile.voxel_size())
        voxel_ratio = (voxel_size[2] / voxel_size[0]) * rescale_factor
           
    # Downscale stack (reduce size)
    stack = downscale_local_mean(
        stack, (1, int(1 / rescale_factor), int(1 / rescale_factor)),
        ).astype("uint16")
    
    # Normalize stack
    stack_norm = stack.copy()
    pMax = np.percentile(stack_norm, 99.9)
    stack_norm[stack_norm > pMax] = pMax
    stack_norm = (stack_norm / pMax)
    
    # Tophat transform
    tophat = []
    for plane in stack:
        tophat.append(gaussian(
            white_tophat(plane, footprint=disk(tophat_size)),
            sigma=tophat_sigma, 
            preserve_range=True,
            ).astype("uint16"))
    tophat = np.stack(tophat)

    return name, path, stack, voxel_size, voxel_ratio, stack_norm, tophat

# -----------------------------------------------------------------------------

def post_process(probs, tophat, name, voxel_size, voxel_ratio):
    
    # 3D smooth probabilities
    probs = gaussian(probs, sigma=(
        prob_sigma / voxel_ratio, prob_sigma, prob_sigma)
        )  
    
    # Get filtered nMasks (border + size criteria)
    nMasks = probs > prob_min
    if clear_nBorder:
        nMasks = clear_border(nMasks)
    nMasks = remove_small_objects(nMasks, min_size=min_nSize)
        
    # Get labelled objects
    nLabels = label(nMasks)
    
    # Get filtered cMasks (size criteria)
    cMasks = np.zeros_like(tophat, dtype=bool)
    for lab in np.unique(nLabels):
        if lab > 0:
            idx = (nLabels == lab)
            values = tophat[idx]
            thresh = threshold_otsu(values)
            cMasks[idx] = (values > thresh * tophat_tresh_coeff)
    cMasks = remove_small_objects(cMasks, min_size=min_cSize)
    
    # Extract volume info
    results = []
    for lab in np.unique(nLabels):
        if lab > 0:
            idx = (nLabels == lab)
            nVolume = np.sum(idx)
            cVolume = np.sum(cMasks[idx])
            results.append((
                name, name[0:6], lab, 
                nVolume * (voxel_size[0] * voxel_size[1] * voxel_size[2]),
                cVolume * (voxel_size[0] * voxel_size[1] * voxel_size[2]), 
                cVolume / nVolume
                ))

    return nMasks, nLabels, cMasks, results

#%% Model ---------------------------------------------------------------------

# Define & compile model
model = sm.Unet(
    'resnet34', 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Load weights
model.load_weights(Path(Path.cwd(), model_name))   

#%% Pre-processing ------------------------------------------------------------

start = time.time()
print('Pre-processing')

output1 = Parallel(n_jobs=-1)(
    delayed(pre_process)(path) for path in nd2_paths
    )
stack_data["name"] = [data[0] for data in output1]
stack_data["path"] = [data[1] for data in output1]
stack_data["stack"] = [data[2] for data in output1]
stack_data["voxel_size"] = [data[3] for data in output1]
stack_data["voxel_ratio"] = [data[4] for data in output1]
stack_data["stack_norm"] = [data[5] for data in output1]
stack_data["tophat"] = [data[6] for data in output1]
del output1; gc.collect()

end = time.time()
print(f'  {(end-start):5.3f} s')   

#%% Predict -------------------------------------------------------------------

start = time.time()
print('Predict')

for stack_norm in stack_data["stack_norm"]:
    
    # Predict
    stack_data["probs"].append(model.predict(stack_norm).squeeze())

end = time.time()
print(f'  {(end-start):5.3f} s')   

#%% Post-processing -----------------------------------------------------------

start = time.time()
print('Post-processing')

output2 = Parallel(n_jobs=-1)(
    delayed(post_process)(probs, tophat, name, voxel_size, voxel_ratio) 
    for probs, tophat, name, voxel_size, voxel_ratio 
    in zip(
        stack_data["probs"],
        stack_data["tophat"],
        stack_data["name"],
        stack_data["voxel_size"],
        stack_data["voxel_ratio"]
        )
    )
stack_data["nMasks"] = [data[0] for data in output2]
stack_data["nLabels"] = [data[1] for data in output2]
stack_data["cMasks"] = [data[2] for data in output2]
stack_data["results"] = [data[3] for data in output2]
del output2; gc.collect()

end = time.time()
print(f'  {(end-start):5.3f} s')   

#%% Save data -----------------------------------------------------------------

start = time.time()
print('Save')

for i, name in enumerate(stack_data["name"]):
    
    # Create directory
    stack_path = Path(data_path, name)
    stack_path.mkdir(parents=True, exist_ok=True)
    
    # Format and save output metadata
    voxel_size = stack_data["voxel_size"][i]
    voxel_ratio = stack_data["voxel_ratio"][i]    
    metadata = pd.DataFrame({
        "voxel_x": [voxel_size[0]],
        "voxel_y": [voxel_size[1]],
        "voxel_z": [voxel_size[2]],
        "voxel_ratio": [voxel_ratio],
        })
    
    metadata.to_csv(Path(stack_path, f'{name}_metadata.csv'), index=False)
    
    # Format and save output data
    stack = stack_data["stack"][i]
    nMasks = (stack_data["nMasks"][i] * 255).astype('uint8')
    cMasks = (stack_data["cMasks"][i] * 255).astype('uint8')
    nLabels = stack_data["nLabels"][i].astype('uint8')
    tophat = (stack_data["tophat"][i]).astype('float32')
    tophat[nMasks == 0] = 0
    
    io.imsave(Path(stack_path, f'{name}_stack.tif'),
        stack, check_contrast=False,
        )
    io.imsave(Path(stack_path, f'{name}_nMasks.tif'),
        nMasks, check_contrast=False,
        )
    io.imsave(Path(stack_path, f'{name}_cMasks.tif'),
        cMasks, check_contrast=False,
        )
    io.imsave(Path(stack_path, f'{name}_nLabels.tif'),
        nLabels, check_contrast=False,
        )
    io.imsave(Path(stack_path, f'{name}_tophat.tif'),
        tophat, check_contrast=False,
        )
       
end = time.time()
print(f'  {(end-start):5.3f} s')   

#%%

results = pd.DataFrame(
    [item for sublist in stack_data["results"] for item in sublist],
    columns=['name', 'cond', 'label', 'nVolume', 'cVolume', 'cnRatio']
    )

KASind_nVolume = results[results['cond'] == 'KASind']['nVolume'].mean()
KZLind_nVolume = results[results['cond'] == 'KZLind']['nVolume'].mean()
KASind_nVolume_sd = results[results['cond'] == 'KASind']['nVolume'].std()
KZLind_nVolume_sd = results[results['cond'] == 'KZLind']['nVolume'].std()
KASind_cVolume = results[results['cond'] == 'KASind']['cVolume'].mean()
KZLind_cVolume = results[results['cond'] == 'KZLind']['cVolume'].mean()
KASind_cVolume_sd = results[results['cond'] == 'KASind']['cVolume'].std()
KZLind_cVolume_sd = results[results['cond'] == 'KZLind']['cVolume'].std()
KASind_cnRatio = results[results['cond'] == 'KASind']['cnRatio'].mean()
KZLind_cnRatio = results[results['cond'] == 'KZLind']['cnRatio'].mean()
KASind_cnRatio_sd = results[results['cond'] == 'KASind']['cnRatio'].std()
KZLind_cnRatio_sd = results[results['cond'] == 'KZLind']['cnRatio'].std()
t_stat, p_value = stats.ttest_ind(
    results[results['cond'] == 'KASind']['cnRatio'], 
    results[results['cond'] == 'KZLind']['cnRatio']
    )

print(f'KASind_nVolume = {KASind_nVolume:.3f} µm3 +- {KASind_nVolume_sd:.3f} sd')
print(f'KZLind_nVolume = {KZLind_nVolume:.3f} µm3 +- {KZLind_nVolume_sd:.3f} sd')
print(f'KASind_cVolume = {KASind_cVolume:.3f} µm3 +- {KASind_cVolume_sd:.3f} sd')
print(f'KZLind_cVolume = {KZLind_cVolume:.3f} µm3 +- {KZLind_cVolume_sd:.3f} sd')
print(f'KASind_cnRatio = {KASind_cnRatio:.3f} +- {KASind_cnRatio_sd:.3f} sd')
print(f'KZLind_cnRatio = {KZLind_cnRatio:.3f} +- {KZLind_cnRatio_sd:.3f} sd')
print(f'p-value = {p_value:.3e}')


