from mmengine.config import Config

def print_mmdetection_config(fp: str):
    cfg = Config.fromfile(fp)
    print(cfg.pretty_text)