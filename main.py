#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path

# functions
from functions import segment, display

#%% Inputs --------------------------------------------------------------------

# stack_name = "FIN_01_00"
# stack_name = "KAS_01_02"
# stack_name = "KZL_01_00"
# stack_name = "KZL_01_03"
stack_name = "NEO_04_02"

#%% Initialize ----------------------------------------------------------------

# Path
data_path = Path("D:/local_Meschichi/data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    path = data_path / (stack_name + ".nd2")
    
    # Segment
    outputs = segment(
        path,
        # nMask
        lmax_dist=5,
        lmax_prom=0.15,
        prob_thresh=0.25,
        clear_nBorder=True,
        min_nSize=4096,
        # cMask
        tophat_size=3,
        tophat_sigma=1,
        tophat_tresh_coeff=1.25,
        min_cSize=32,
        )
    
    # Display
    display(
        outputs["stack"],
        outputs["metadata"],
        outputs["nLabels"],
        outputs["cLabels"],
        outputs["tophat"],
        )
    
    


    
