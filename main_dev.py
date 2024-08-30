#%% Imports -------------------------------------------------------------------

import time
import napari
from numba import cuda
from skimage import io
from pathlib import Path
from functions import limit_vram, segment, save

#%% Comments ------------------------------------------------------------------

''' 
Since EDM are required, do measurments within segment function, 
taking advantage of the reduced-size data
'''

#%% Parameters ----------------------------------------------------------------

# Path
# data_path = Path(Path.cwd(), "data", "local") 
data_path = Path("D:/local_Meschichi/data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")

# Batch
batch = False
overwrite = True
stack_name = "KASind1" # if batch == False 

# nMask
lmax_dist = 5
lmax_prom = 0.15
prob_thresh = 0.25
clear_nBorder = True
min_nSize = 4096

# cMask
tophat_size = 3
tophat_sigma = 1
tophat_tresh_coeff = 1.25
min_cSize = 32

# GPU
vram = None # Limit vram (None to deactivate)
if vram is not None:
    limit_vram(vram)
    print(f"VRAM limited to {vram}")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Get paths
    paths = []
    for path in data_path.iterdir():
        if path.suffix != ".nd2":
            continue
        
        is_dir = (path.parent / path.stem).is_dir()
        
        if batch:
           if overwrite:
               paths.append(path)
           elif not is_dir:
               paths.append(path)
        elif path.stem == stack_name:
            if overwrite:
                paths.append(path)
            elif not is_dir:
                paths.append(path)

    # Process
    for path in paths:
           
        outputs = segment(
            path,
            # nMask
            lmax_dist=lmax_dist,
            lmax_prom=lmax_prom,
            prob_thresh=prob_thresh,
            clear_nBorder=clear_nBorder,
            min_nSize=min_nSize,
            # cMask
            tophat_size=tophat_size,
            tophat_sigma=tophat_sigma,
            tophat_tresh_coeff=tophat_tresh_coeff,
            min_cSize=min_cSize,
            )
        
        save(path, outputs)
        
#%%

    # t0 = time.time()
    # # Clear VRAM
    # cuda.select_device(0)
    # cuda.close()
    # t1 = time.time()
    # print(f"{t1 - t0:.5f}")
                
    # # Display 
    # scale = [outputs["metadata"]["vZ"] / outputs["metadata"]["vY"], 1, 1]
    # viewer = napari.Viewer()
    # viewer.add_image(outputs["stack"], scale=scale)
    # viewer.add_labels(outputs["nLabels"], scale=scale, opacity=0.75)
    # viewer = napari.Viewer()
    # viewer.add_image(outputs["tophat"], scale=scale)
    # viewer.add_labels(outputs["cLabels"], scale=scale, opacity=0.75)