#%% Imports -------------------------------------------------------------------

import napari
from pathlib import Path
from functions import limit_vram, segment

#%% Parameters ----------------------------------------------------------------

# Path
# data_path = Path(Path.cwd(), "data", "local") 
data_path = Path("D:/local_Meschichi/data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")

# GPU
vram = None # Limit vram (None to deactivate)
if vram is not None:
    limit_vram(vram)
    print(f"VRAM limited to {vram}")

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
       
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for path in data_path.iterdir():
        if path.suffix == ".nd2":
            if path.stem == "KASind1": # dev
            # if path.stem == "KZLind1": # dev
            # if path.stem == "KASind1003": # dev
            # if path.stem == "KZLind1002": # dev
            # if path.stem == "KZLind3001": # dev
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
                
    # Display 
    scale = [outputs["metadata"]["vZ"] / outputs["metadata"]["vY"], 1, 1]
    viewer = napari.Viewer()
    viewer.add_image(outputs["stack"], scale=scale)
    viewer.add_labels(outputs["nLabels"], scale=scale, opacity=0.75)
    viewer = napari.Viewer()
    viewer.add_image(outputs["tophat"], scale=scale)
    viewer.add_labels(outputs["cLabels"], scale=scale, opacity=0.75)