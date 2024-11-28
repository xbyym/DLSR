import torch
import torch.nn as nn

from omegaconf import OmegaConf
import numpy as np
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import load_model,CustomGroupNorm,calculate_similarity
from Pre_train_model.models.model_helper import ModelHelper
from Pre_train_model.utils.misc_helper import update_config

class Generative_Mul_Feature(nn.Module):
    """ Reconstruct Multi-scale features and Perform OOD detection """
    def __init__(self, use_Mul_feature=True,
                 indices=None,  # Allow indices to be optionally passed in
                 use_class_label=False, 
                 pretrained_ldm_ckpt=None,
                 pretrained_ldm_cfg=None,
                 subset_timesteps_num=None,
                 similarity_type='MFsim'): 
        super().__init__()
        assert not (use_Mul_feature and use_class_label)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) 
        # Load pre-trained encoder
        self.use_Mul_feature = use_Mul_feature
        self.use_class_label = use_class_label
        self.similarity_type = similarity_type
        # --------------------------------------------------------------------------
        # Load pre-trained sampler
        if pretrained_ldm_ckpt is not None and pretrained_ldm_cfg is not None:
            ldm_config = OmegaConf.load(pretrained_ldm_cfg)
            ldm_config = update_config(ldm_config)
            self.ldm_fake_class_label = ldm_config.model.params.cond_stage_config.params.n_classes - 1
            ldm_model = load_model(ldm_config, pretrained_ldm_ckpt)
            self.ldm_sampler = DDIMSampler(ldm_model,subset_timesteps_num)
            
            # Load indices from the config if not provided
            if indices is None:
                self.indices = ldm_config.model.params.get('indices', [24, 56, 112, 272, 720])  # Default to EfficientNet_B4 if not specified
            else:
                self.indices = indices
        else:
            self.ldm_fake_class_label = 0
            # Set default indices if none provided or if config is missing
            self.indices = indices if indices is not None else [24, 56, 112, 272, 720]
            
        if pretrained_ldm_cfg is not None:
            pretrained_enc_config = ldm_config.model.params.pretrained_enc_config
            self.instantiate_pretrained_enc(pretrained_enc_config)

    def forward(self, imgs, class_label,
                gen__image=True, bsz=None, num_iter=None, choice_temperature=None,
                sampled_Multi_Feature=None, ldm_steps=100, eta=1.0, cfg=0.0, class_label_gen=None):
        if gen__image:
            self.pretrained_encoder.eval()
            with torch.no_grad():
                self.pretrained_encoder.eval()   
                input = {"image": imgs}
                outputs = self.pretrained_encoder(input)
                rep = outputs["feature_align"]
            z = self.global_avg_pool(rep)
            z = z.squeeze(-1).squeeze(-1)
            # Use the indices loaded from the configuration
            custom_group_norm = CustomGroupNorm(self.indices)
            # Apply the custom group normalization
            z = custom_group_norm(z)

            sampled_Multi_Feature, class_label = self.gen_image(64, z, num_iter, choice_temperature, sampled_Multi_Feature, ldm_steps, eta, cfg, class_label_gen)
            
            ood_score = calculate_similarity(sampled_Multi_Feature, z,self.indices,self.similarity_type)
            
            return ood_score 

    def gen_image(self, bsz=64, imgs=None, num_iter=12, choice_temperature=4.5, sampled_Multi_Feature=None, ldm_steps=100, eta=1.0,
                  cfg=0.0, class_label=None):
        """
        Generate images using the model. The DDIM sampler is used to generate representations.
        Using multi-dimensional features obtained from DDIM.
        """
        # Sample feature from LFDN
        if sampled_Multi_Feature is None: 
            with self.ldm_sampler.model.ema_scope("Plotting"):
                shape = [self.ldm_sampler.model.model.diffusion_model.in_channels,
                         self.ldm_sampler.model.model.diffusion_model.image_size,
                         self.ldm_sampler.model.model.diffusion_model.image_size]
                if self.ldm_sampler.model.class_cond:
                    cond = {"class_label": class_label}
                else:
                    class_label = self.ldm_fake_class_label * torch.ones(bsz).cuda().long()
                    cond = {"class_label": class_label}
                cond = self.ldm_sampler.model.get_learned_conditioning(cond)
                # Use DDIM to sample original representations here
                sampled_Multi_Feature, _ = self.ldm_sampler.sample(ldm_steps, conditioning=cond, x0=imgs, batch_size=bsz,
                                                          shape=shape,
                                                          eta=eta, verbose=False)
                sampled_Multi_Feature = sampled_Multi_Feature.squeeze(-1).squeeze(-1)

                # Add unconditional representation for classifier-free guidance
                if cfg > 0:
                    uncond_rep = self.fake_latent.repeat(bsz, 1)
                    sampled_Multi_Feature = torch.cat([sampled_Multi_Feature, uncond_rep], dim=0)
        # Check if class label is needed for guiding reconstruction
        if self.use_class_label:
            assert cfg == 0
            class_label = torch.randint(0, 1000, (bsz,)).cuda()

        return sampled_Multi_Feature, class_label

    def instantiate_pretrained_enc(self, pretrained_enc_cfg):
        """
        Ensure the input passed to ModelHelper is a pure Python list.
        """
        if isinstance(pretrained_enc_cfg, OmegaConf):
            pretrained_enc_cfg = OmegaConf.to_container(pretrained_enc_cfg, resolve=True)

        self.pretrained_encoder = ModelHelper(pretrained_enc_cfg) 
        self.pretrained_encoder.cuda() 
        self.pretrained_encoder.eval()
