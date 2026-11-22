"""Frozen pi0.5 prefix-cache operation used by the goal readout."""
import torch
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def prefix(base,batch):
    m=base.model
    images,image_masks=base._preprocess_images(batch)
    embedding,mask,att=m.embed_prefix(images,image_masks,batch[OBS_LANGUAGE_TOKENS],batch[OBS_LANGUAGE_ATTENTION_MASK])
    att4=prepare_attention_masks_4d(make_att_2d_masks(mask,att))
    pos=torch.cumsum(mask,dim=1)-1
    m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation="eager"
    _,kv=m.paligemma_with_expert.forward(attention_mask=att4,position_ids=pos,past_key_values=None,
                                          inputs_embeds=[embedding,None],use_cache=True,adarms_cond=[None,None])
    return mask,kv
