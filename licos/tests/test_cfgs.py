import os
from dotmap import DotMap
import toml

cfgs = ["cfg/default_cfg.toml","cfg/raw_merged.toml","cfg/raw_split.toml"]

def test_cfgs():
    for path in cfgs:
      if not os.path.exists(path):
          raise Exception(f"No cfg file found at {path}.")
      print(f"Loading cfg from {path}")
      with open(path) as cfg:
          # dynamic=False inhibits automatic generation of non-existing keys
          cfg = DotMap(toml.load(cfg), _dynamic=False)
      print(cfg)