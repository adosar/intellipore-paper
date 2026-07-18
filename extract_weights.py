import torch
from modules import IntelliPore

ckpt_path = 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'
ckpt = torch.load(ckpt_path)

params = ckpt['state_dict']
backbone_params = {
        k.removeprefix('backbone.'): v for k, v in params.items()
        if k.startswith('backbone')
        }

# Check that can be loaded, convert to CPU and saved them.
backbone = IntelliPore().backbone
backbone.load_state_dict(backbone_params)
backbone.cpu()
torch.save(backbone.state_dict(), 'pretrained_weights/intellipore_backbone_pretrained.pt')
