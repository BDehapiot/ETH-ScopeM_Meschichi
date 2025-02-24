#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.models import preprocess

#%% Inputs --------------------------------------------------------------------

target = "rscale" 
patch_size = 128
patch_overlap = 0

#%% Initialize ----------------------------------------------------------------

train_path = Path("data", "train")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Load data
    X, y = [], []
    for path in list(train_path.glob(f"*{target}*")):
        if not "mask" in path.name:
            X.append(io.imread(path))
        else:
            y.append(io.imread(path))
    X = np.stack(X)
    y = np.stack(y)
    
    # Remove empty masks
    valid = []
    for i in range(len(X)):
        if np.max(y[i, ...]) > 0:
            valid.append(i)
    X = X[valid]
    y = y[valid]
          
    # Preprocess
    X, y = preprocess(
        X, msks=y, 
        patch_size=patch_size, 
        patch_overlap=patch_overlap,
        img_norm="global",
        msk_type="edt",
        )
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(X)
    viewer.add_image(y)
    
    pass