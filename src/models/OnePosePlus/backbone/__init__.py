from .resnet import ResNetFPN_8_2

from math import log
from functools import lru_cache


def build_backbone(config):
    if config['type'] == 'ResNetFPN':
        if config['resolution'] == [8, 2]:
            return ResNetFPN_8_2(config['resnetfpn'])
        else:
            raise NotImplementedError
    else:
        raise ValueError("reaching this line! LOFTR_BACKBONE.TYEP and RESOLUTION are not correct")


@lru_cache(maxsize=128)
def _res2ind(resolutions, output_layers):
    """resolutions to indices of feats returned by resfpn"""
    lid2ind = {lid: ind for ind, lid in enumerate(output_layers)}
    inds = [lid2ind[int(log(r, 2))] for r in resolutions]
    return inds


def _get_win_rel_scale(config):
    try:
        min_layer_id = min(config['resnetfpn']['output_layers'])
        rel_scale = int(log(config['resolution'][1], 2)) - min_layer_id
    except KeyError as _:
        min_layer_id = min(config['RESNETFPN']['OUTPUT_LAYERS'])
        rel_scale = int(log(config['RESOLUTION'][1], 2)) - min_layer_id
    return 2**rel_scale


def _get_feat_dims(config):
    layer_dims = [1, *config['resnetfpn']['block_dims']]
    output_layers = config['resnetfpn']['output_layers']
    return [layer_dims[i] for i in output_layers]


def _split_backbone_feats(feats, bs):
    split_feats = [feat.split(bs, dim=0) for feat in feats]
    feats0 = [f[0] for f in split_feats]
    feats1 = [f[1] for f in split_feats]
    return feats0, feats1


def _extract_backbone_feats(feats, config):
    """For backwrad compatibility temporarily."""
    if config['type'] == 'ResNetFPN':
        _output_layers = tuple(config['resnetfpn']['output_layers'])
        if len(_output_layers) == 2:
            for r, l in zip(config['resolution'], _output_layers):
                assert 2 ** l == r
            return feats
        else:
            return [feats[i] for i in _res2ind(config['resolution'], _output_layers)]
    else:
        return feats
