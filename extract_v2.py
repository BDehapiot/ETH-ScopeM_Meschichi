#%% Imports -------------------------------------------------------------------

import nd2
import random 
import numpy as np
from skimage import io
from pathlib import Path

# Skimage
from skimage.transform import rescale

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path('D:/local_Meschichi/data')
stock_path = Path(Path.cwd(), 'data', 'stock') 

# Parameters
random.seed(42)
nPlane = 5

#%% Extract -------------------------------------------------------------------

for path in local_path.iterdir():
    if path.suffix == '.nd2':
                
        # Read nd2 file
        with nd2.ND2File(path) as ndfile:
            stack = ndfile.asarray()
            vY, vX, vZ = ndfile.voxel_size()
            
        # Rescale & reslice (isotropic voxel)
        zRatio = vY / vZ
        rscale = rescale(stack, (1, zRatio, zRatio), order=0)
        rslice = np.swapaxes(rscale, 0, 1)
          
        # Select and save random plane (2D)
        for i in range(nPlane): 
            rand1 = random.randint(0, rscale.shape[0] - 1)
            rand2 = random.randint(0, rslice.shape[0] - 1)
            io.imsave(
                Path(stock_path, f'{path.stem}_rscale{rand1}.tif'),
                rscale[rand1,...], check_contrast=False,
                )
            io.imsave(
                Path(stock_path, f'{path.stem}_rslice{rand2}.tif'),
                rslice[rand2,...], check_contrast=False,
                )
