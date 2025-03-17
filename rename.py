#%% Imports -------------------------------------------------------------------

from pathlib import Path

#%% Inputs --------------------------------------------------------------------

data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Meschichi\data")
paths = list(data_path.glob("**/*.nd2"))
names = [path.stem for path in paths]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    for path, name in zip(paths, names):
        cnd = name[:3].upper() 
        mnt = int(name[6])
        if "NEO Second batch" in str(path):
            mnt += 4         
        if len(name) == 7:
            img = 0
        else:
            img = int(name[-1])
        new_name = (f"{cnd}_{mnt:02d}_{img:02d}")
        path.rename(data_path / (new_name + ".nd2"))