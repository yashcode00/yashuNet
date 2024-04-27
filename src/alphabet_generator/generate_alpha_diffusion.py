from alphabet_models import Diffusion, UNet_conditional
import torch
from PIL import Image
from torchvision.utils import save_image


## classes:
## ['#', '$', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n = 1
device = "cpu"
ckpt = torch.load("/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/models/diffusion/DDPM_conditional_alphs_20240427_021340/pthFiles/ema_ckpt_12.pt",
                  map_location=torch.device('cpu'))

# Remove the "module" prefix from keys
new_state_dict = {}
for key, value in ckpt.items():
    if key.startswith("module."):
        new_key = key[7:]
    else:
        new_key = key
    new_state_dict[new_key] = value

diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, num_classes=39, c_in=1, c_out=1)
diffusion.load(new_state_dict)
y = torch.Tensor([0] * n).long().to(device)
x = diffusion.sample(True, y).float()
print(x.shape)

img = x[0]
save_image(img, 'img1.png')
