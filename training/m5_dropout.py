"""Versioned transformer dropout configuration over the frozen constraint trainer.

Weights/optimizer/representation retain their checkpoint schema. Configuration,
source identity and an extra operator tag record the changed training behavior.
Inference in eval mode remains compatible with the existing physical exporters.
"""
import math
from torch import nn
from training.m5_constraints import PhysicalConstraintTrainer

VERSION = 'm5-transformer-attention-feedforward-dropout-v1'


def set_transformer_dropout(model, rate):
    counts = {'attention_modules': 0, 'feedforward_residual_dropout_modules': 0}
    for block in (model.root_model, model.body_model):
        block.dropout = rate
        for module in block.seqTransEncoder.modules():
            if isinstance(module, nn.MultiheadAttention):
                module.dropout = rate; counts['attention_modules'] += 1
            elif isinstance(module, nn.Dropout):
                module.p = rate; counts['feedforward_residual_dropout_modules'] += 1
        # Positional dropout is a separate setting, deliberately kept at zero.
        if block.sequence_pos_encoder.dropout.p != 0.:
            raise ValueError('This version requires positional dropout=0')
    if not all(counts.values()):
        raise ValueError('No supported transformer dropout modules found')
    return counts


class PhysicalDropoutTrainer(PhysicalConstraintTrainer):
    def __init__(self, config, samples, identity, stats_dir, device='cuda:0'):
        rate = float(config.get('network_dropout', 0.))
        if not math.isfinite(rate) or not 0. <= rate < 1.:
            raise ValueError('Invalid transformer dropout rate')
        if config.get('pe_dropout', 0.) != 0.:
            raise ValueError('Positional dropout needs its own versioned experiment')
        super().__init__(config, samples, identity, stats_dir, device)
        self.dropout_rate = rate
        self.dropout_modules = set_transformer_dropout(self.model, rate)
        if set_transformer_dropout(self.ema, rate) != self.dropout_modules:
            raise ValueError('Model/EMA module structure differs')

    def state(self):
        result = super().state()
        result.update(dropout_operator=VERSION, transformer_dropout_rate=self.dropout_rate)
        return result

    def load_state(self, state, continuation=False, initialization_only=False):
        operator = state.get('dropout_operator')
        if operator not in (None, VERSION):
            raise ValueError('Unsupported dropout implementation')
        if not continuation and not initialization_only:
            if operator != VERSION or state.get('transformer_dropout_rate') != self.dropout_rate:
                raise ValueError('Resume dropout configuration/operator mismatch')
        return super().load_state(state, continuation=continuation, initialization_only=initialization_only)
