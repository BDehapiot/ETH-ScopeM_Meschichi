#%% Imports -------------------------------------------------------------------

import nd2
import random
from skimage import io
from pathlib import Path

random.seed(42) 

#%% Initialize ----------------------------------------------------------------

data_path = Path('D:\\local_Meschichi\\data')
train_path = Path('D:\\local_Meschichi\\data\\train')

for stack_path in data_path.iterdir():
    if stack_path.suffix == '.nd2':
        
        # open stack
        stack = nd2.imread(stack_path) 
        stack_name = stack_path.stem
        
        # extract random z_plane
        z_rand = random.randint(0, stack.shape[0] - 1)
        z_plane = stack[z_rand,...]
        
        # save extracted z_plane
        z_plane_name = f'{stack_name}_z{z_rand}.tif'        
        io.imsave(
            Path(train_path, z_plane_name),
            z_plane,
            check_contrast=False,
            )