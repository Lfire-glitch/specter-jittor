"""Map a HuggingFace BERT state dict (numpy .npz) onto the Jittor BertModel.

Key names follow the allenai/specter checkpoint, which has NO `bert.` prefix:
    embeddings.word_embeddings.weight
    encoder.layer.{i}.attention.self.{query,key,value}.{weight,bias}
    encoder.layer.{i}.attention.output.dense.{weight,bias}
    encoder.layer.{i}.attention.output.LayerNorm.{weight,bias}
    encoder.layer.{i}.intermediate.dense.{weight,bias}
    encoder.layer.{i}.output.dense.{weight,bias}
    encoder.layer.{i}.output.LayerNorm.{weight,bias}
    pooler.dense.{weight,bias}
`embeddings.position_ids` is a buffer and is ignored.
"""
import numpy as np
import jittor as jt


def build_mapping(model):
    m = {}
    e = model.embeddings
    m['embeddings.word_embeddings.weight'] = e.word_embeddings.weight
    m['embeddings.position_embeddings.weight'] = e.position_embeddings.weight
    m['embeddings.token_type_embeddings.weight'] = e.token_type_embeddings.weight
    m['embeddings.LayerNorm.weight'] = e.LayerNorm.weight
    m['embeddings.LayerNorm.bias'] = e.LayerNorm.bias
    for i, layer in enumerate(model.layers):
        p = f'encoder.layer.{i}.'
        a = layer.attention
        for name in ('query', 'key', 'value'):
            m[p + f'attention.self.{name}.weight'] = getattr(a.self_attn, name).weight
            m[p + f'attention.self.{name}.bias'] = getattr(a.self_attn, name).bias
        m[p + 'attention.output.dense.weight'] = a.output.weight
        m[p + 'attention.output.dense.bias'] = a.output.bias
        m[p + 'attention.output.LayerNorm.weight'] = a.LayerNorm.weight
        m[p + 'attention.output.LayerNorm.bias'] = a.LayerNorm.bias
        m[p + 'intermediate.dense.weight'] = layer.intermediate.weight
        m[p + 'intermediate.dense.bias'] = layer.intermediate.bias
        m[p + 'output.dense.weight'] = layer.output.weight
        m[p + 'output.dense.bias'] = layer.output.bias
        m[p + 'output.LayerNorm.weight'] = layer.LayerNorm.weight
        m[p + 'output.LayerNorm.bias'] = layer.LayerNorm.bias
    m['pooler.dense.weight'] = model.pooler.weight
    m['pooler.dense.bias'] = model.pooler.bias
    return m


def load_npz(model, npz_path, verbose=True):
    """Returns (loaded, skipped, transposed). Raises on any unmatched/mismatched key."""
    z = np.load(npz_path)
    mp = build_mapping(model)
    loaded, skipped, transposed, missing = 0, [], [], []
    for k in z.files:
        if k not in mp:
            skipped.append(k)
            continue
        arr = z[k]
        tgt = mp[k]
        if tuple(tgt.shape) != tuple(arr.shape):
            if arr.ndim == 2 and tuple(tgt.shape) == tuple(arr.T.shape):
                arr = arr.T
                transposed.append(k)
            else:
                raise ValueError(f'shape mismatch {k}: hf {arr.shape} vs jt {tgt.shape}')
        tgt.assign(jt.array(np.ascontiguousarray(arr, dtype=np.float32)))
        loaded += 1
    for k in mp:
        if k not in z.files:
            missing.append(k)
    if missing:
        raise ValueError(f'{len(missing)} jittor params had no HF counterpart, e.g. {missing[:3]}')
    if verbose:
        print(f'[load] loaded={loaded} skipped={len(skipped)} transposed={len(transposed)}')
        if skipped:
            print(f'[load] skipped keys: {skipped}')
    return loaded, skipped, transposed
