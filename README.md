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

1. Navigate to the `data_preparation` directory and clone [PyRawS](https://github.com/ESA-PhiLab/PyRawS) on it. PyRaWS provide APIs to process Sentinel-2 raw data.
2. Install PyRaWs as indicated in its [README](https://github.com/ESA-PhiLab/PyRawS#installation).
3. Download [THRawS](https://zenodo.org/record/7908728#.ZGxSMHZBy3A). 
4. Create a directory called `THRAWS`, including a subdirectory called `raw`. 
5. Decompress all the zip files in `raw`. 
6. Place `THRAWS` into the `data` directory and update the `sys_cfg.py` of `PYRawS` to point the `data` directory. For more information, please refer to [Data directory](https://github.com/ESA-PhiLab/PyRawS#data-directory).
7. Activate the `pyraws` environment. 
8. From `data_preparation` launch the `create_tif.py`. For more information, please launch: 

```python create_tif --help```



