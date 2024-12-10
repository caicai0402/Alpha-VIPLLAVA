# Copyright 2023 Cruise LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob

import torch
import torch.nn as nn
from safetensors.torch import load_file

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .clip_vision_model import AlphaCLIPVisionModel

class CLIPVisionTowerMultilayer(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained('openai/clip-vit-large-patch14-336')

    def load_model(self, pretrained_model_name_or_path=None):
        self.image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        self.vision_tower = AlphaCLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')
        # self.vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')
        
        if pretrained_model_name_or_path is not None:
            safetensor_files = glob.glob(os.path.join(pretrained_model_name_or_path, "*model*.safetensors")) + glob.glob(os.path.join(pretrained_model_name_or_path, "*model*.bin"))
            
            loaded_model_weights = {}
            for model_path in safetensor_files:
                if model_path.endswith(".safetensors"):
                    loaded_weights = load_file(model_path)
                else:
                    loaded_weights = torch.load(model_path)
                loaded_model_weights.update(loaded_weights)
            
            vision_tower_weights = {}
            for target_k in self.vision_tower.state_dict().keys():
                for k, v in loaded_model_weights.items():
                    if target_k in k:
                        vision_tower_weights[target_k] = v
            
            self.vision_tower.load_state_dict(vision_tower_weights, strict=False)
            print("\nload vision_tower weights successfully!\n")
                    
        # self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = [image_forward_outs['hidden_states'][index][:, 1:] for index in [-2, -5, -8, -11, 6]]
        image_features = torch.cat(image_features, dim=-1)
        return image_features

    # @torch.no_grad()
    def forward(self, images, visual_prompt_alphas=None):
        if visual_prompt_alphas == "llava_arch prepare_inputs_labels_for_multimodal":
            raise NotImplementedError("llava_arch prepare_inputs_labels_for_multimodal")
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            if visual_prompt_alphas is None:
                visual_prompt_alphas = torch.zeros(images.size(0), 1, images.size(2), images.size(3))
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), visual_prompt_alphas.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size * 5

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
