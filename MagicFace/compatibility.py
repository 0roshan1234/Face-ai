import torch
import diffusers
if not hasattr(diffusers, 'ConfigMixin'):
    from diffusers.configuration_utils import ConfigMixin
    diffusers.ConfigMixin = ConfigMixin
if not hasattr(diffusers, 'ModelMixin'):
    from diffusers.models.modeling_utils import ModelMixin
    diffusers.ModelMixin = ModelMixin
print("Compatibility patches applied!")