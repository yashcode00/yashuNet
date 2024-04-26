import os
import torch
from modelCode.model import UNet
from modelCode.config import ModelConfig
from dotenv import load_dotenv
load_dotenv()

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
model = UNet(3,1)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, os.getenv('MODEL_NAME')), map_location='cpu'))
config = ModelConfig()