import torch
from rl_algorithms.utils import Registry, build_from_cfg
from rl_algorithms.utils.config import ConfigDict

from rl_algorithms.common.networks.cnn import CNNLayer


AGENTS = Registry("agents")
MODELS = Registry("models")
HERS = Registry("hers")


def build_agent(cfg: ConfigDict, build_args: dict = None):
    """Build agent using config and additional arguments."""
    return build_from_cfg(cfg, AGENTS, build_args)


from rl_algorithms.dqn.utils import calculate_fc_input_size, get_fc_model

def build_model(cfg: ConfigDict, build_args: dict = None):
    """Build agent using config and additional arguments."""

    observation_dim = build_args["observation_dim"]
    action_dim = build_args["action_dim"]

    cnn_cfg = cfg.cnn_cfg
    fc_cfg = cfg.fc_cfg

    input_size = calculate_fc_input_size(observation_dim, cnn_cfg.params)
    
    fc_model = get_fc_model(fc_cfg.params, input_size, action_dim, fc_cfg["hidden_sizes"])

    model_args = dict(cnn_layers=list(map(CNNLayer, *cnn_cfg.params.values())), fc_layers=fc_model)
    Model = build_from_cfg(cnn_cfg, MODELS, model_args)

    return Model

def build_her(cfg: ConfigDict, build_args: dict = None):
    """Build her using config and additional arguments."""
    return build_from_cfg(cfg, HERS, build_args)
