from collections import OrderedDict

import torch

state_dict = torch.load('pretrained/epoch_160.pth')

original_key_map = {old_key: old_key for old_key in state_dict.keys()}

after_pillar_encoder_name_map = {key: value.replace('pillar_encoder', 'pillar_feature_net.pillar_encoder') for key, value in original_key_map.items()}
after_backbone_name_map = {key: value.replace('backbone', 'backbone.fpn_encoder') for key, value in after_pillar_encoder_name_map.items()}
after_neck_name_map = {key: value.replace('neck', 'backbone.fpn_decoder') for key, value in after_backbone_name_map.items()}
after_head_name_map = {key: value.replace('head', 'detection_head') for key, value in after_neck_name_map.items()}

renamed_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key in after_head_name_map:
        new_key = after_head_name_map[key]
        renamed_state_dict[new_key] = value
    else:
        renamed_state_dict[key] = value

torch.save(renamed_state_dict, 'development/reformat_state_dict.pth')