#%% Imports -------------------------------------------------------------------

import gc
import nd2
import time
import napari
import numpy as np
from pathlib import Path
import segmentation_models as sm
from skimage.measure import label
from joblib import Parallel, delayed 
from skimage.transform import downscale_local_mean
from skimage.segmentation import clear_border
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, white_tophat, disk, ball

#%% Initialize ----------------------------------------------------------------

# Get paths
nd2_paths = []
for path in Path(Path.cwd(), 'data', 'local').iterdir():
    if path.suffix == '.nd2':
        nd2_paths.append(path)
        
# Model
model_name = 'model_weights_0.5.h5'
rescale_factor = float(model_name[14:17])

# Create stack_data dict
stack_data = {
    "name": [],
    "path": [],
    "stack": [],
    "voxel_size": [],
    "voxel_ratio": [],
    "mask": [],
    "labels": [],
    "probs": [],
    }

#%% Parameters ----------------------------------------------------------------

# Nuclei mask (nMask)
prob_min = 0.5
prob_sigma = 2 * rescale_factor
clear_nBorder = False
min_nSize = 4096 * rescale_factor
        
#%% Functions -----------------------------------------------------------------

def pre_process(path):
    
    # Open data & metadata
    name = path.name
    with nd2.ND2File(path) as ndfile:
        stack = ndfile.asarray()
        voxel_size = list(ndfile.voxel_size())
        voxel_ratio = (voxel_size[2] / voxel_size[0]) * rescale_factor
    
    # Downscale stack (reduce size)
    stack = downscale_local_mean(
        stack, (1, int(1 / rescale_factor), int(1 / rescale_factor)),
        )
    
    # Normalize stack
    pMax = np.percentile(stack, 99.9)
    stack[stack > pMax] = pMax
    stack = (stack / pMax)
    
    return name, path, stack, voxel_size, voxel_ratio

# -----------------------------------------------------------------------------

def post_process(probs, voxel_ratio):
    
    # 3D smooth probabilities
    probs = gaussian(probs, sigma=(
        prob_sigma / voxel_ratio, prob_sigma, prob_sigma)
        )  
    
    # Get filtered mask (border & size criteria)
    mask = probs > prob_min
    if clear_nBorder:
        mask = clear_border(mask)
    mask = remove_small_objects(mask, min_size=min_nSize)
        
    # Get labelled objects
    labels = label(mask)
    
    return mask, labels

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
del output1; gc.collect()

end = time.time()
print(f'  {(end-start):5.3f} s')   

#%% Predict -------------------------------------------------------------------

start = time.time()
print('Predict')

for stack in stack_data["stack"]:
    stack_data["probs"].append(model.predict(stack).squeeze())

end = time.time()
print(f'  {(end-start):5.3f} s')   

#%% Post-processing -----------------------------------------------------------

start = time.time()
print('Post-processing')

output2 = Parallel(n_jobs=-1)(
    delayed(post_process)(probs, voxel_ratio) 
    for probs, voxel_ratio 
    in zip(stack_data["probs"], stack_data["voxel_ratio"])
    )
stack_data["mask"] = [data[0] for data in output2]
stack_data["labels"] = [data[1] for data in output2]
del output2; gc.collect()

end = time.time()
print(f'  {(end-start):5.3f} s')   

#%% Save data -----------------------------------------------------------------

# for i in len(stack_data["name"]):
#     print(stack_data["name"][i])   
    
