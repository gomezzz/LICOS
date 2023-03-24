from licos.main import main

import os
from dotmap import DotMap
import toml


def test_run():
    path = "cfg/default_cfg.toml"
    if not os.path.exists(path):
        raise Exception(f"No cfg file found at {path}.")
    print(f"Loading cfg from {path}")
    with open(path) as cfg:
        # dynamic=False inhibits automatic generation of non-existing keys
        cfg = DotMap(toml.load(cfg), _dynamic=False)
    print(cfg)
    cfg.simulation_time = 1
    main(cfg)
