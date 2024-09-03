## Outputs

### nData.csv - nuclei data

```bash
# Nuclei data
    - nLabel  # nuclei IDs
    - nVolume # volume (µm3)  
    - nCtrd
    - nMajor
    - nMinor
    - nMMRatio


# Associated chromocenters data
    - n_cLabel # associated chromocenter IDs
    - n_cNumber 
    - n_cVolume
    - n_cnRatio
```

### cData.csv - chromocenters data

```bash
# Chromocenters data data
    - cLabel  # chromocenter IDs
    - nVolume # volume (µm3)  
    - cCtrd
    - cMajor
    - cMinor
    - cMMRatio
    - cInt
    - cEDMb
    - cEDMc

# Associated nuclei data
    - c_nLabel # associated chromocenter IDs
```