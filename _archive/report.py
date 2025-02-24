#%% Imports -------------------------------------------------------------------

import os
import napari
import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

# functions
from functions import open_stack, format_stack

# bdtools 
from bdtools.norm import norm_pct

# bdmodel
from bdmodel.predict import predict

# skimage
from skimage.exposure import adjust_gamma
from skimage.morphology import binary_dilation

#%% Parameters ----------------------------------------------------------------

stack_name = "KZL_01_02" ; rscale_idx = 35 ; rslice_idx = 116
gamma = 0.5

params = {
    
    # colors
    "color0" : (229, 229, 229),
    "color1" : (153, 153, 153),
    
    # "gray_05"  : (242, 242, 242),
    # "gray_10"  : (229, 229, 229),
    # "gray_20"  : (204, 204, 204),
    # "gray_30"  : (179, 179, 179),
    # "gray_40"  : (153, 153, 153),
    # "gray_50"  : (128, 128, 128),
    # "gray_80"  : ( 51,  51,  51),
    
    }

#%% Initialize ----------------------------------------------------------------

# Path
data_path = Path("D:/local_Meschichi/data")
train_path = Path(Path.cwd(), "data", "train")
remote_path = Path("G:/Mon Drive/Resources/BDCourse_DIP")
save_path = remote_path / "0-locals/ETH-ScopeM_Meschichi"

#%% Function(s) ---------------------------------------------------------------

def img2RGB(
        img, pct_low=0, pct_high=100, 
        color0=(0, 0, 0), color1=(255, 255, 255),
        ):
        
    dtype = img.dtype
    img = img.astype(float)
            
    if dtype != bool:
        img = norm_pct(img, pct_low=pct_low, pct_high=pct_high)
    
    # Convert to RGB
    
    if color1 == "random":
        
        unique_vals = np.unique(img)
        color_map = {val: (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            ) for val in unique_vals if val != 0}
        color_map[0] = color0
        
        img_rgb = np.zeros((*img.shape, 3), dtype=np.uint8)
        for val, color in color_map.items():
            img_rgb[img == val] = color
    
    else:
        
        r = (color0[0] + img * (color1[0] - color0[0])).astype("uint8")
        g = (color0[1] + img * (color1[1] - color0[1])).astype("uint8")
        b = (color0[2] + img * (color1[2] - color0[2])).astype("uint8")
        img_rgb = np.stack((r, g, b), axis=-1)
        
    return img_rgb  

def crop(img, brd=20):
    msk = img > 0
    max_y = np.max(msk, axis=1)
    fnz_y = np.flatnonzero(max_y)
    y0, y1 = fnz_y[0] - brd, fnz_y[-1] + brd
    max_x = np.max(msk, axis=0)
    fnz_x = np.flatnonzero(max_x)
    x0, x1 = fnz_x[0] - brd, fnz_x[-1] + brd
    return img[y0:y1, x0:x1]

#%% Function : examples() -----------------------------------------------------

def examples():
    
    def save2RGB(raw, mxp, m3d, rsc, rsl, rsc_gt, rsl_gt, rsc_prd, rsl_prd):
        
        # Convert to RGB
        img_RGB = img2RGB(
            adjust_gamma(img, gamma=gamma), pct_low=0.01, pct_high=99.99)
        mxp_RGB = img2RGB(
            adjust_gamma(mxp, gamma=gamma), pct_low=0.01, pct_high=99.99)
        m3d_RGB = img2RGB(
            adjust_gamma(m3d, gamma=0.8), pct_low=0.01, pct_high=99.99)
        rsc_RGB = img2RGB(
            adjust_gamma(rsc, gamma=gamma), pct_low=0.01, pct_high=99.99)
        rsl_RGB = img2RGB(
            adjust_gamma(rsl, gamma=gamma), pct_low=0.01, pct_high=99.99)
        rsc_gt_RGB = img2RGB(
            adjust_gamma(
                rsc_gt, gamma=gamma), pct_low=0.01, pct_high=99.99,
                color0=params["color0"], color1="random")
        rsl_gt_RGB = img2RGB(
            adjust_gamma(
                rsl_gt, gamma=gamma), pct_low=0.01, pct_high=99.99,
                color0=params["color0"], color1="random")
        rsc_prd_RGB = img2RGB(
            adjust_gamma(rsc_prd, gamma=gamma), pct_low=0.01, pct_high=99.99)
        rsl_prd_RGB = img2RGB(
            adjust_gamma(rsl_prd, gamma=gamma), pct_low=0.01, pct_high=99.99)
        
        # Save
        img_name = "expl_img_RGB.tif"
        mxp_name = "expl_mxp_RGB.tif"
        m3d_name = "expl_m3d_RGB.tif"
        rsc_name = "expl_rsc_RGB.tif"
        rsl_name = "expl_rsl_RGB.tif"
        rsc_gt_name = "expl_rsc_gt_RGB.tif"
        rsl_gt_name = "expl_rsl_gt_RGB.tif"
        rsc_prd_name = "expl_rsc_prd_RGB.tif"
        rsl_prd_name = "expl_rsl_prd_RGB.tif"
        io.imsave(save_path / img_name, img_RGB, check_contrast=False)
        io.imsave(save_path / mxp_name, mxp_RGB, check_contrast=False)
        io.imsave(save_path / m3d_name, m3d_RGB, check_contrast=False)
        io.imsave(save_path / rsc_name, rsc_RGB, check_contrast=False)
        io.imsave(save_path / rsl_name, rsl_RGB, check_contrast=False)
        io.imsave(save_path / rsc_gt_name, rsc_gt_RGB, check_contrast=False)
        io.imsave(save_path / rsl_gt_name, rsl_gt_RGB, check_contrast=False)
        io.imsave(save_path / rsc_prd_name, rsc_prd_RGB, check_contrast=False)
        io.imsave(save_path / rsl_prd_name, rsl_prd_RGB, check_contrast=False)
        
    # Predict
    rscale_prds = predict(
        rscale, 
        Path("bdmodel", "model_rscale_128"), 
        img_norm="global",
        patch_overlap=0,
        )
    rslice_prds = predict(
        rslice, 
        Path("bdmodel", "model_rslice_128"),
        img_norm="global",
        patch_overlap=0,
        )
    
    # Get examples
    img = stack[rscale_idx]
    mxp = np.max(stack, axis=0)
    rsc = rscale[rscale_idx]
    rsl = rslice[rslice_idx]
    rsc_prd = rscale_prds[rscale_idx]
    rsl_prd = rslice_prds[rslice_idx]  
    
    # Merge predictions
    rslice_prds = np.swapaxes(rslice_prds, 0, 1)
    prds_avg = np.mean(np.stack((rscale_prds, rslice_prds), axis=0), axis=0) 
    prds_std = np.std(np.stack((rscale_prds, rslice_prds), axis=0), axis=0) 
    prds = prds_avg - prds_std
    
    # Get GT
    rscale_gt_name = stack_name + f"_rscale_{rscale_idx:03d}_mask.tif"
    rscale_gt = io.imread(train_path / rscale_gt_name)
    rslice_gt_name = stack_name + f"_rslice_{rslice_idx:03d}_mask.tif"
    rslice_gt = io.imread(train_path / rslice_gt_name)
    
    # Convert to RGB & save 
    save2RGB(
        img, mxp, m3d, rsc, rsl, 
        rscale_gt, rslice_gt, 
        rsc_prd, rsl_prd, 
        )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    path = data_path / (stack_name + ".nd2")
    
    # Load & format data
    stack, metadata = open_stack(path)
    rscale, rslice = format_stack(stack, metadata) 
    
    # Display -----------------------------------------------------------------
    
    # Headless napari
    # os.environ["NAPARI_OCTREE"] = "1"
    # os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
    # Initialize
    scale = [metadata["vZ"] / metadata["vY"], 1, 1]
    angles = (15.86, -22.85, -36.26)
    center = (stack.shape[0] * 2, stack.shape[1] // 2, stack.shape[2] // 2)
    
    # Viewer (labels)
    viewer = napari.Viewer()
    viewer.add_image(stack, scale=scale)
    
    # Viewer (orientation)
    viewer.dims.ndisplay = 3
    viewer.camera.angles = angles
    viewer.camera.center = center
    viewer.camera.zoom = 0.5

    # Get views
    m3d = viewer.screenshot()[..., 0]
    m3d = crop(m3d, brd=20)    
    
    viewer.close()
    
    # -------------------------------------------------------------------------
    
    # Examples
    examples()
    
#%% Display -------------------------------------------------------------------

    # def print_camera_angles(event):
    #     print(f"Camera Angles (elev, azi, roll): {viewer.camera.angles}")
    # def print_camera_center(event):
    #     print(f"Camera Center: {viewer.camera.center}")
    # viewer.camera.events.angles.connect(print_camera_angles)
    # viewer.camera.events.center.connect(print_camera_center)

    
