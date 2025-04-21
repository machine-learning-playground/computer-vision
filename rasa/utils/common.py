from easydict import EasyDict

import ruamel.yaml as yaml
import torch.distributed as dist


def parse_config(config_path):
    yaml_loader = yaml.YAML()
    with open(config_path, "r") as file:
        config = yaml_loader.load(file)
    config = EasyDict(config)
    return config


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
