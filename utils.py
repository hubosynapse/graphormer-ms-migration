import os
import json
import logging
import torch
import mindspore as ms
import numpy as np

# Convert pytorch_model.bin to ms_graphormer_model.ckpt
def convert_torch_to_mindspore(pth_file, **kwargs):
    """Convert pytorch model .bin to mindspore model .ckpt"""
    prefix = kwargs.get('prefix', '')

    try:
        import torch
    except Exception as exc:
        raise ImportError("'import torch' failed, please install torch by "
                          "`pip install torch` or instructions from 'https://pytorch.org'") \
        from exc

    from mindspore.train.serialization import save_checkpoint

    logging.info('Starting checkpoint conversion.')
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))

    for k, v in state_dict.items():
        if 'shared.weight' in k:
            k = k.replace('shared.weight', 'decoder.embed_tokens.embedding_table')
        if 'layer_norm' in k:
            k = k.replace('layer_norm.weight', 'layer_norm.gamma')
            k = k.replace('layer_norm.bias', 'layer_norm.beta')
        if (('encoder.weight' in k) or 
            ('graph_token_virtual_distance.weight' in k) 
            or ('graph_token.weight' in k)):
            k = k.replace('weight', 'embedding_table')
        if prefix:
            k = prefix + "." + k
        print(k)
        ms_ckpt.append({'name': k, 'data': ms.Tensor(v.numpy())})

    ms_ckpt_path = pth_file.replace('pytorch_model.bin','mindspore.ckpt')
    if not os.path.exists(ms_ckpt_path):
        try:
            save_checkpoint(ms_ckpt, ms_ckpt_path)
        except Exception as exc:
            raise RuntimeError(f'Save checkpoint to {ms_ckpt_path} failed, please checkout the path.') \
            from exc

    return ms_ckpt_path