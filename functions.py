#%% Imports -------------------------------------------------------------------

import nd2
import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import segmentation_models as sm
from joblib import Parallel, delayed

# bdtools
from bdtools.models.unet import UNet
from bdtools.models import preprocess
from bdtools.patch import merge_patches
from bdtools.norm import norm_gcn, norm_pct

# Skimage
from skimage.feature import peak_local_max
from skimage.transform import rescale, resize
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import clear_border, expand_labels
from skimage.morphology import (
    disk, remove_small_objects, white_tophat, binary_dilation
    )

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Functions (GPU) -----------------------------------------------------------

def limit_vram(vram):

    import tensorflow as tf    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    memory_config = tf.config.experimental\
        .VirtualDeviceConfiguration(memory_limit=vram)
    
    if gpus:
        try:
            tf.config.experimental\
                .set_virtual_device_configuration(gpus[0], [memory_config])
        except RuntimeError as e:
            print(e)

#%% Functions (open_stack) ----------------------------------------------------

def open_stack(path, metadata=True):
    
    # Read nd2 file
    with nd2.ND2File(path) as ndfile:
        stack = ndfile.asarray()
        nZ, nY, nX = stack.shape
        vY, vX, vZ = ndfile.voxel_size()
    
    if metadata:
        
        metadata = {
            "nZ" : nZ, "nY" : nY, "nX" : nX, 
            "vZ" : vZ, "vY" : vY, "vX" : vX,
            }
    
        return stack, metadata
    
    return stack
    