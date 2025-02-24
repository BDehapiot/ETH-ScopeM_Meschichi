#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path

# bdmodel
from bdmodel.train import Train
from bdmodel.functions import get_paths

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Paths
    train_path = Path("data", "train")
    
    # Open data
    imgs, msks = [], []
    msk_paths = get_paths(
        train_path, 
        ext=".tif", 
        subfolders=False,
        )
    for path in msk_paths:
        imgs.append(io.imread(str(path).replace("_mask", "")))
        msks.append(io.imread(path))
        
    # Display
        
    # # Train
    # train = Train(
    #     imgs, msks,
    #     save_name="rslice_image_128",
    #     save_path=Path("bdmodel"),
    #     msk_type="normal",
    #     img_norm="image",
    #     patch_size=128,
    #     patch_overlap=0,
    #     nAugment=3000,
    #     backbone="resnet18",
    #     epochs=200,
    #     batch_size=32,
    #     validation_split=0.2,
    #     learning_rate=0.0005,
    #     patience=30,
    #     weights_path="",
    #     # weights_path=Path(Path.cwd(), "rscale_128", "weights.h5"),
    #     )
    
    # test = Path("bdmodel")
    # print(test.resolve())