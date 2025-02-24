#%% Imports -------------------------------------------------------------------

import os
import warnings
import numpy as np
os.environ['NO_ALBUMENTATIONS_UPDATE'] = "1" # Don't know if it works
import albumentations as A
from joblib import Parallel, delayed 

# bdtools
from bdtools.mask import get_edt
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches

# Skimage
from skimage.filters import gaussian
from skimage.exposure import adjust_gamma
from skimage.segmentation import find_boundaries 

#%% Function: get_paths() -----------------------------------------------------

def get_paths(
        rootpath, 
        ext=".tif", 
        tags_in=[], 
        tags_out=[], 
        subfolders=False, 
        ):
    
    """     
    Retrieve file paths with specific extensions and tag criteria from a 
    directory. The search can include subfolders if specified.
    
    Parameters
    ----------
    rootpath : str or pathlib.Path
        Path to the target directory where files are located.
        
    ext : str, default=".tif"
        File extension to filter files by (e.g., ".tif" or ".jpg").
        
    tags_in : list of str, optional
        List of tags (substrings) that must be present in the file path
        for it to be included.
        
    tags_out : list of str, optional
        List of tags (substrings) that must not be present in the file path
        for it to be included.
        
    subfolders : bool, default=False
        If True, search will include all subdirectories within `rootpath`. 
        If False, search will be limited to the specified `rootpath` 
        directory only.
        
    Returns
    -------  
    selected_paths : list of pathlib.Path
        A list of file paths that match the specified extension and 
        tag criteria.
        
    """
    
    if subfolders:
        paths = list(rootpath.rglob(f"*{ext}"))
    else:
        paths = list(rootpath.glob(f"*{ext}"))
        
    selected_paths = []
    for path in paths:
        if tags_in:
            check_tags_in = all(tag in str(path) for tag in tags_in)
        else:
            check_tags_in = True
        if tags_out:
            check_tags_out = not any(tag in str(path) for tag in tags_out)
        else:
            check_tags_out = True
        if check_tags_in and check_tags_out:
            selected_paths.append(path)

    return selected_paths

#%% Function: preprocess() ----------------------------------------------------
   
def preprocess(
        imgs, msks=None,
        img_norm="global",
        msk_type="normal", 
        patch_size=256, 
        patch_overlap=0,
        ):
    
    """ 
    Preprocess images and masks for training or prediction procedures.
    
    If msks=None, only images will be preprocessed.
    Images and masks will be splitted into patches.
    
    Parameters
    ----------
    imgs : 2D ndarray or list of 2D ndarrays (int or float)
        Inputs image(s).
        
    msks : 2D ndarray or list of 2D ndarrays (bool or int), optional, default=None 
        Inputs mask(s).
        If None, only images will be preprocessed.
        
    img_norm : str, default="global"
        - "global" : 0 to 1 normalization considering the full stack.
        - "image"  : 0 to 1 normalization per image.
                
    msk_type : str, default="normal"
        - "normal" : No changes.
        - "edt"    : Euclidean distance transform of binary/labeled objects.
        - "bounds" : Boundaries of binary/labeled objects.

    patch_size : int, default=256
        Size of extracted patches.
        Should be int > 0 and multiple of 2.
    
    patch_overlap : int, default=0
        Overlap between patches.
        Should be int, from 0 to patch_size - 1.
        
    Returns
    -------  
    imgs : 3D ndarray (float32)
        Preprocessed images.
        
    msks : 3D ndarray (float32), optional
        Preprocessed masks.
        
    """
    
    valid_types = ["normal", "edt", "bounds"]
    if msk_type not in valid_types:
        raise ValueError(
            f"Invalid value for msk_type: '{msk_type}'."
            f" Expected one of {valid_types}."
            )

    valid_norms = ["global", "image"]
    if img_norm not in valid_norms:
        raise ValueError(
            f"Invalid value for img_norm: '{img_norm}'."
            f" Expected one of {valid_norms}."
            )
        
    if patch_size <= 0 or patch_size % 2 != 0:
        raise ValueError(
            f"Invalid value for patch_size: '{patch_size}'."
            f" Should be int > 0 and multiple of 2."
            )

    if patch_overlap < 0 or patch_overlap >= patch_size:
        raise ValueError(
            f"Invalid value for patch_overlap: '{patch_overlap}'."
            f" Should be int, from 0 to patch_size - 1."
            )

    # Nested function(s) ------------------------------------------------------

    def normalize(arr, sample_fraction=0.1):
        arr = norm_gcn(arr, sample_fraction=sample_fraction)
        arr = norm_pct(arr, sample_fraction=sample_fraction)
        return arr      
            
    def _preprocess(img, msk=None):

        if msks is None:
            
            img = np.array(img)
            
            img = extract_patches(img, patch_size, patch_overlap)
                 
            return img
            
        else:
            
            img = np.array(img)
            msk = np.array(msk)
            
            if msk_type == "normal":
                msk = msk > 0
            elif msk_type == "edt":
                msk = get_edt(msk, normalize="object", parallel=False)
            elif msk_type == "bounds":
                msk = find_boundaries(msk)           
            
            img = extract_patches(img, patch_size, patch_overlap)
            msk = extract_patches(msk, patch_size, patch_overlap)
                
            return img, msk
    
    # Execute -----------------------------------------------------------------        
       
    # Normalize images
    if img_norm == "global":
        imgs = normalize(imgs)
    if img_norm == "image":
        imgs = [normalize(img) for img in imgs]
   
    # Preprocess
    if msks is None:
        
        if isinstance(imgs, np.ndarray):           
            if imgs.ndim == 2: imgs = [imgs]
            elif imgs.ndim == 3: imgs = list(imgs)
        
        if len(imgs) > 1:
               
            outputs = Parallel(n_jobs=-1)(
                delayed(_preprocess)(img)
                for img in imgs
                )
            imgs = [data for data in outputs]
            imgs = np.stack([arr for sublist in imgs for arr in sublist])
                
        else:
            
            imgs = _preprocess(imgs)
            imgs = np.stack(imgs)
        
        imgs = imgs.astype("float32")
        
        return imgs
    
    else:
        
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 2: imgs = [imgs]
            elif imgs.ndim == 3: imgs = list(imgs)
        if isinstance(msks, np.ndarray):
            if msks.ndim == 2: msks = [msks]
            elif msks.ndim == 3: msks = list(msks)
        
        if len(imgs) > 1:
            
            outputs = Parallel(n_jobs=-1)(
                delayed(_preprocess)(img, msk)
                for img, msk in zip(imgs, msks)
                )
            imgs = [data[0] for data in outputs]
            msks = [data[1] for data in outputs]
            imgs = np.stack([arr for sublist in imgs for arr in sublist])
            msks = np.stack([arr for sublist in msks for arr in sublist])
            
        else:
            
            imgs, msks = _preprocess(imgs, msks)
            imgs = np.stack(imgs)
            msks = np.stack(msks)
            
        imgs = imgs.astype("float32")
        msks = msks.astype("float32")
        
        return imgs, msks
    
#%% Function: augment() -------------------------------------------------------

# def augment(imgs, msks, iterations):
        
#     if iterations <= imgs.shape[0]:
#         warnings.warn(f"iterations ({iterations}) is less than n of images")
        
#     # Nested function(s) ------------------------------------------------------
    
#     def _augment(imgs, msks):     
        
#         operations = A.Compose([
#             A.VerticalFlip(p=0.5),              
#             A.RandomRotate90(p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.Transpose(p=0.5),
#             A.GridDistortion(p=0.5),
#             ])
        
#         idx = np.random.randint(0, len(imgs) - 1)
#         outputs = operations(image=imgs[idx,...], mask=msks[idx,...])
#         return outputs["image"], outputs["mask"]
    
#     # Execute -----------------------------------------------------------------
    
#     outputs = Parallel(n_jobs=-1)(
#         delayed(_augment)(imgs, msks)
#         for i in range(iterations)
#         )
#     imgs = np.stack([data[0] for data in outputs])
#     msks = np.stack([data[1] for data in outputs])
    
#     return imgs, msks

def augment(
        imgs, msks, iterations,
        gamma_p=0.5, gblur_p=0.5, noise_p=0.5, flip_p=0.5, distord_p=0.5,
        ):
    
    """
    Augment images and masks using random transformations.
    
    The following transformation are applied:
        
        - adjust gamma (image only)      
        - apply gaussian blur (image only) 
        - add noise (image only) 
        - flip (image & mask)
        - grid distord (image & mask)
    
    If required, image transformations are applied to their correponding masks.
    Transformation probabilities can be set with function arguments.
    Transformation random parameters can be tuned with the params dictionnary.
    Grid distortions are applied with the `albumentations` library.
    https://albumentations.ai/

    Parameters
    ----------
    imgs : 3D ndarray (float)
        Input image(s).
        
    msks : 3D ndarray (float) 
        Input corresponding mask(s).
        
    iterations : int
        The number of augmented samples to generate.
    
    gamma_p, gblur_p, noise_p, flip_p, distord_p : float (0 to 1) 
        Probability to apply the transformation.
    
    Returns
    -------
    imgs : 3D ndarray (float)
        Augmented image(s).
        
    msks : 3D ndarray (float) 
        Augmented corresponding mask(s).
    
    """
    
    # Parameters --------------------------------------------------------------
    
    params = {
        
        # Gamma
        "gamma_low"  : 0.75,
        "gamma_high" : 1.25,
        
        # Gaussian blur
        "sigma_low"  : 1,
        "sigma_high" : 3,
        
        # Noise
        "sgain_low"   : 20,
        "sgain_high"  : 50,
        "rnoise_low"  : 2,
        "rnoise_high" : 4,
        
        # Grid distord
        "nsteps_low"  : 1,
        "nsteps_high" : 10,
        "dlimit_low"  : 0.1,
        "dlimit_high" : 0.5,
        
        }
    
    # Nested functions --------------------------------------------------------
    
    def _gamma(img, gamma=1.0):
        img_mean = np.mean(img)
        img = adjust_gamma(img, gamma=gamma)
        img = img * (img_mean / np.mean(img))
        return img
    
    def _noise(img, shot_gain=0.1, read_noise_std=5):
        img_std = np.std(img) 
        # img = np.random.poisson(img * shot_gain) / shot_gain
        img += np.random.normal(
            loc=0.0, scale=img_std / read_noise_std, size=img.shape)
        return img
    
    def _flip(img, msk):
        if np.random.rand() < 0.5:
            img, msk = np.flipud(img), np.flipud(msk)
        if np.random.rand() < 0.5:
            img, msk = np.fliplr(img), np.fliplr(msk)
        if img.shape[0] == img.shape[1]:
            if np.random.rand() < 0.5:
                k = np.random.choice([-1, 1])
                img = np.rot90(img, k=k)
                msk = np.rot90(msk, k=k)
        return img, msk
    
    def _augment(img, msk):
        
        img = img.copy()
        msk = msk.copy()
        
        if np.random.rand() < gamma_p:
            gamma = np.random.uniform(
                params["gamma_low"], params["gamma_high"])
            img = _gamma(img, gamma=gamma)
            
        if np.random.rand() < gblur_p:
            sigma = np.random.randint(
                params["sigma_low"], params["sigma_high"])
            img = gaussian(img, sigma=sigma)
            
        if np.random.rand() < noise_p:
            shot_gain = np.random.uniform(
                params["sgain_low"], params["sgain_high"])
            read_noise_std = np.random.randint(
                params["rnoise_low"], params["rnoise_high"])
            img = _noise(
                img, shot_gain=shot_gain, read_noise_std=read_noise_std)
            
        if np.random.rand() < flip_p:
            img, msk = _flip(img, msk)
            
        if np.random.rand() < distord_p:
            num_steps = np.random.randint(
                params["nsteps_low"], params["nsteps_high"])
            distort_limit = np.random.uniform(
                params["dlimit_low"], params["dlimit_high"])
            spatial_transforms = A.Compose([
                A.GridDistortion(
                    num_steps=num_steps, 
                    distort_limit=distort_limit, 
                    p=1
                    )
                ])
            outputs = spatial_transforms(image=img, mask=msk)
            img, msk = outputs["image"], outputs["mask"]
        
        return img, msk
        
    # Execute -----------------------------------------------------------------
    
    # Initialize
    imgs = imgs.astype("float32")
    idxs = np.random.choice(
        np.arange(0, imgs.shape[0]), size=iterations)
    
    outputs = Parallel(n_jobs=-1, backend="threading")(
        delayed(_augment)(imgs[i], msks[i])
        for i in idxs
        )
    
    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
        
    return imgs, msks