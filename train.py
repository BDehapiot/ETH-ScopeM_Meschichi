#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.models.unet import UNet

#%% Inputs --------------------------------------------------------------------

# UNet build()
backbone = "resnet34"
activation = "sigmoid"
downscale_factor = 2

# UNet train()

# preprocess
patch_size = 512
patch_overlap = 0
img_norm = "global"
msk_type = "bounds"

# augment
iterations = 5000
gamma_p = 0.5
gblur_p = 0
noise_p = 0 
flip_p = 0.5 
distord_p = 0.5

# train
epochs = 100
batch_size = 32 
validation_split = 0.2
metric = "soft_dice_coef"
learning_rate = 0.0005
patience = 20

save_name = f"ds{downscale_factor}_{patch_size}_{msk_type}"

#%% Initialize ----------------------------------------------------------------

train_path = Path("data", "train")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Load data
    imgs, msks = [], []
    for path in list(train_path.glob("*.tif")):
        if "mask" in path.name:
            msks.append(io.imread(path))   
            imgs.append(io.imread(str(path).replace("_mask", "")))
    imgs = np.stack(imgs)
    msks = np.stack(msks)
     
    unet = UNet(
        save_name="",
        load_name="",
        root_path=Path.cwd(),
        backbone=backbone,
        classes=1,
        activation=activation,
        )
    
    unet.train(
        
        imgs, msks, 
        X_val=None, y_val=None,
        preview=False,
        
        # Preprocess
        img_norm=img_norm, 
        msk_type=msk_type, 
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        downscaling_factor=downscale_factor, 
        
        # Augment
        iterations=iterations,
        gamma_p=gamma_p, 
        gblur_p=gblur_p, 
        noise_p=noise_p, 
        flip_p=flip_p, 
        distord_p=distord_p,
        
        # Train
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        metric=metric,
        learning_rate=learning_rate,
        patience=patience,
        
        )