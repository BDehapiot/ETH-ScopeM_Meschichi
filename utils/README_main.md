## Outputs

### nData.csv - nuclei data

```bash
# Nuclei data
    - nLabel    # IDs
    - nVolume   # volume (µm3)  
    - nCtrd     # centroid coords (zyx)
    - nMajor    # fitted ellipse major axis length (µm)
    - nMinor    # fitted ellipse minor axis length (µm)
    - nMMRatio  # minor / major axis ratio

# Associated chromocenters data
    - n_cLabel  # associated chromocenter IDs
    - n_cNumber # number of cromocenters
    - n_cVolume # cumulative volume of chromocenters (µm3)
    - n_cnRatio # chromocenter / nuclei volume ratio
```

### cData.csv - chromocenters data

```bash
# Chromocenters data data
    - cLabel    # IDs
    - nVolume   # volume (µm3)  
    - cCtrd     # centroid coords (zyx)
    - cMajor    # fitted ellipse major axis length (µm)
    - cMinor    # fitted ellipse minor axis length (µm)
    - cMMRatio  # minor / major axis ratio
    - cInt      # intensity mean (raw intensities)
    - cEDMb     # distance from nucleus periphery (µm)
    - cEDMc     # distance from nucleus centroid (µm)

# Associated nuclei data
    - c_nLabel  # associated nucleus ID
```