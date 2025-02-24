#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.models.metrics import *
from bdtools.models import preprocess, train


#%% Inputs --------------------------------------------------------------------

# General
target = "rscale" 

# preprocess()
patch_size = 128
patch_overlap = 0
img_norm = "global"
msk_type = "edt"

# UNet build()
save_name = f"{target}"
backbone = "resnet18"
activation = "sigmoid"
downscale_steps = 1

# UNet train()
epochs = 100
batch_size = 32 
validation_split = 0.2
metric = "soft_dice_coef"
learning_rate = 0.0005
patience = 20

#%% Initialize ----------------------------------------------------------------

train_path = Path("data", "train")

#%% Eimgsecute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Load data
    imgs, msks = [], []
    for path in list(train_path.glob(f"*{target}*")):
        if not "mask" in path.name:
            imgs.append(io.imread(path))
        else:
            msks.append(io.imread(path))
    imgs = np.stack(imgs)
    msks = np.stack(msks)
    
    # Remove empty masks
    valid = []
    for i in range(len(imgs)):
        if np.max(msks[i, ...]) > 0:
            valid.append(i)
    imgs = imgs[valid]
    msks = msks[valid]
          
    # Preprocess
    imgs, msks = preprocess(
        imgs, msks=msks, 
        patch_size=patch_size, 
        patch_overlap=patch_overlap,
        img_norm=img_norm,
        msk_type=msk_type,
        )
    
    # Train
    unet = UNet(
        save_name=save_name,
        backbone=backbone,
        activation=activation,
        downscale_steps=downscale_steps, 
        )

    unet.train(
        imgs, msks, 
        X_val=None, y_val=None,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        metric=metric,
        learning_rate=learning_rate,
        patience=patience,
        )
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(imgs)
    viewer.add_image(msks)
    
    pass