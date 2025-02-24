#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from pathlib import Path

# functions
from functions import open_stack

# bdtools
from bdtools.models import preprocess
from bdtools.models.unet import UNet
from bdtools.patch import merge_patches

#%% Inputs --------------------------------------------------------------------

# stack_name = "FIN_01_00"
# stack_name = "KAS_01_02"
# stack_name = "KZL_01_00"
# stack_name = "KZL_01_03"
stack_name = "NEO_04_02"

# preprocess()
patch_size = 256
patch_overlap = 128
img_norm = "global"
msk_type = "edt"

#%% Initialize ----------------------------------------------------------------

data_path = Path("D:/local_Meschichi/data")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Path
    path = data_path / (stack_name + ".nd2")
    
    # Load & format data
    stack, metadata = open_stack(path)
    
    # Preprocess & predict
    unet = UNet(load_name="256_edt")
    prds = unet.predict(
        preprocess(
            stack, msks=None, 
            patch_size=patch_size, 
            patch_overlap=patch_overlap,
            img_norm=img_norm,
            msk_type=msk_type,
            )
        )
    
    # Merge patches
    prds = merge_patches(prds, stack.shape, patch_overlap)
           
    # Display
    viewer = napari.Viewer()
    viewer.add_image(stack)
    viewer.add_image(prds)

    
