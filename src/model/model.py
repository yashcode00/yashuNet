""" the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import sys
sys.path.append("/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/src/model")
from unet_parts import *
from tqdm import tqdm

"""This is the main Unet model used for segmentation"""
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


"""This is Unet encoder only for self-supervised: Rotation Prediction (ROT)"""
class UNetDense(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetDense, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_class = n_classes
        self.fc = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(2048, self.num_class),
            nn.Softmax(dim=1)  # Add softmax activation for multiclass classification
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_pool = self.Avgpool(x5)
        x6 = torch.flatten(x5_pool, 1)
        # out = self.projector(x6)
        out = self.fc(x6)
        return out

class ContrastiveSiameseUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, lr=1e-3):
        super(ContrastiveSiameseUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024 // factor, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.n_classes)  # Output a 512-dimensional feature vector
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_pool = self.Avgpool(x5)
        x6 = torch.flatten(x5_pool, 1)
        features = self.fc(x6)
        return features
        

# '''sanity check'''
# def test():
#     x = torch.randn((3, 1, 161, 161))
#     model = UNet(1, 1)
#     preds = model(x)
#     print(preds.shape)
#     print(x.shape)

#     assert preds.shape == x.shape

# def test_selfSupervised():
#     x = torch.randn((3, 3, 161, 161))
#     model = UNet(3, 4)
#     preds = model(x)
#     print(preds)
#     print(preds.shape)

# test()


#########################################################################################################
## For Score based diffusion models
#########################################################################################################


# @title Time embedding and modulation

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""

  def __init__(self,embed_dim,scale=30.):
    super().__init__()
    # Randomly sample weights (frequencies) during initialization.
    # These weights (frequencies) are fixed during optimization and are not trainable.
    self.W=nn.Parameter(torch.randn(embed_dim//2)*scale,requires_grad=False)

  def forward(self,x):
    # Cosine(2 pi freq x), Sine(2 pi freq x)
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps.
  Allow time repr to input additively from the side of a convolution layer.
  """
  def __init__(self,input_dim,output_dim):

    super().__init__()
    self.dense=nn.Linear(input_dim,output_dim)
  def forward(self,x) :

    return self.dense(x)[...,None,None]
  
class UNet_Conditional(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256,
          text_dim=256, nClass=39):
    """Initialize a time-dependent score-based network.

    Args:
    marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
    channels: The number of channels for feature maps of each resolution.
    embed_dim: The dimensionality of Gaussian random feature embeddings of time.
    text_dim:  the embedding dimension of text / digits.
    nClass:    number of classes you want to model.
    """
    super().__init__()
    # random embedding for classes
    self.cond_embed=nn.Embedding(nClass,text_dim)
    self.time_embed=nn.Sequential(

        GaussianFourierProjection(embed_dim=embed_dim),
        nn.Linear(embed_dim, embed_dim)
    )
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.t_mod1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False)
    self.t_mod2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.y_mod2 = Dense(embed_dim, channels[1])

    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=False)
    self.t_mod3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.y_mod3 = Dense(embed_dim, channels[2])

    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=1, bias=False)
    self.t_mod4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])
    self.y_mod4 = Dense(embed_dim, channels[3])

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=1, bias=False, padding=0)
    self.t_mod5 = Dense(embed_dim, channels[2])
    self.y_mod5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

    self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1, bias=False, padding=0)     #  + channels[2]
    self.t_mod6 = Dense(embed_dim, channels[1])
    self.y_mod6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

    self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=1, bias=False, padding=0)     #  + channels[1]
    self.t_mod7 = Dense(embed_dim, channels[0])
    self.y_mod7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1)

    # The swish activation function
    self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
    for module in [self.y_mod2,self.y_mod3,self.y_mod4,
                self.y_mod5,self.y_mod6,self.y_mod7]:
        nn.init.normal_(module.dense.weight, mean=0, std=0.0001)
        nn.init.constant_(module.dense.bias, 1.0)

  def forward(self, x, t, y=None):
    # print(x.shape)
    # Obtain the Gaussian random feature embedding for t
    embed = self.act(self.time_embed(t))
    y_embed = self.cond_embed(y)
    # Encoding path
    h1 = self.conv1(x) + self.t_mod1(embed)
    # print(f"Conv1: {h1.shape}")
    ## Incorporate information from t
    ## Group normalization
    h1 = self.act(self.gnorm1(h1))
    # print(f"further Conv1: {h1.shape}")
    h2 = self.conv2(h1) + self.t_mod2(embed)
    # print(f"Conv2: {h2.shape}")

    h2 = h2 * self.y_mod2(y_embed)
    h2 = self.act(self.gnorm2(h2))
    # print(f"furthern Conv2: {h2.shape}")
    h3 = self.conv3(h2) + self.t_mod3(embed)
    # print(f"Conv3: {h3.shape}")

    h3 = h3 * self.y_mod3(y_embed)
    h3 = self.act(self.gnorm3(h3))
    # print(f"further Conv3: {h3.shape}")

    h4 = self.conv4(h3) + self.t_mod4(embed)
    # print(f"Conv4: {h4.shape}")

    h4 = h4 * self.y_mod4(y_embed)
    h4 = self.act(self.gnorm4(h4))
    # print(f"further Conv4: {h4.shape}")


    # Decoding path
    h = self.tconv4(h4) + self.t_mod5(embed)
    # print(f"up conv1: {h.shape}")
    h = h * self.y_mod5(y_embed)
    ## Skip connection from the encoding path
    h = self.act(self.tgnorm4(h))
    # print(f"furhern up conv1 h: {h.shape}")
    # print(f"shape of h3: {h3.shape}, h: {h.shape}m, t_mod6: {self.t_mod6(embed).shape}")
    h = self.tconv3(h + h3) + self.t_mod6(embed)
    h = h * self.y_mod6(y_embed)
    h = self.act(self.tgnorm3(h))
    h = self.tconv2(h + h2) + self.t_mod7(embed)
    h = h * self.y_mod7(y_embed)
    h = self.act(self.tgnorm2(h))
    h = self.tconv1(h + h1)

    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h

def loss_fn_cond(model, x, y, marginal_prob_std, eps=1e-3):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t=torch.rand(x.shape[0],device=x.device)*(1.-eps)+eps
  z=torch.randn_like(x)
  std=marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t, y=y)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2,
                              dim=(1, 2, 3)))
  return loss

def marginal_prob_std(t, Lambda, device='cpu'):
  """Compute the standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    t: A vector of time steps.
    Lambda: The $\lambda$ in our SDE.

  Returns:
    std : The standard deviation.
  """
  t=t.to(device)
  std=torch.sqrt((Lambda**(2*t)-1.)/2.)/np.log(Lambda)
  return std

def diffusion_coeff(t, Lambda, device='cpu'):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    Lambda: The $\lambda$ in our SDE.

  Returns:
    diff_coeff : The vector of diffusion coefficients.
  """
  diff_coeff=Lambda**t
  return diff_coeff.to(device)

# @title Define the Sampler
def Euler_Maruyama_sampler(score_model,
              marginal_prob_std,
              diffusion_coeff,
              batch_size=64,
              x_shape=(1, 32, 32),
              num_steps=1000,
              device='cuda',
              eps=1e-3, y=None):
            
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.
  """
  t = torch.ones(batch_size).to(device)
  r = torch.randn(batch_size, *x_shape).to(device)
  init_x = r * marginal_prob_std(t)[:, None, None, None]
  init_x = init_x.to(device)
  time_steps = torch.linspace(1., eps, num_steps).to(device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y=y) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
  # Do not include any noise in the last sampling step.
  return mean_x


