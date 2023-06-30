# LICOS
Learning Image Compression On board a Satellite constellation

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#installation">Installation</a>
    <ul>
      <li><a href="#data-preparation">Data preparation</a></li>
    </ul>
    </li>
  </ol>
</details>

## Installation

### Dataset preparation
This work is based on Sentinel-2 Raw data. included in the dataset [THRawS](https://zenodo.org/record/7908728#.ZGxSMHZBy3A).
To prepare your data, proceed as follows. 

1. Navigate to the `data_preparation` directory and clone [PyRawS](https://github.com/ESA-PhiLab/PyRawS) in it with `git clone https://github.com/ESA-PhiLab/PyRawS.git` . PyRaWS provide APIs to process Sentinel-2 raw data.
2. Install PyRaWs as indicated in its [README](https://github.com/ESA-PhiLab/PyRawS#installation).
3. Download [THRawS](https://zenodo.org/record/7908728#.ZGxSMHZBy3A). Please, notice the entire dataset size if of **147.6 GB**.
4. Place all the downaloded ZIP files into `data_preparation\data\THRAWS\raw`. There is an empty file called `put_THRAWS_here.txt` to give you indication of the right location. 
5. Decompress all the zip files in `data_preparation\data\THRAWS\raw`. 
6. Update the variables `PYRAWS_HOME_PATH` and `DATA_PATH` variables in `data_preparation\sys_cfg.py` with the absolute path to `PyRawS` and `data` directories. 
 For more information, please refer to [Data directory](https://github.com/ESA-PhiLab/PyRawS#data-directory).
7. Move `data_preparation\sys_cfg.py` to `data_preparation\PyRawS\pyraws\sys_cfg.py`.
8. Activate the `pyraws` environment through:

```conda activate pyraws```

9. From `data_preparation` launch the `create_tif.py`. For more information, please launch: 

```python create_tif.py --help```

10. `PyRawS` is not needed anymore and can be now removed.

