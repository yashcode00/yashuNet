from flask import Flask, request, send_file
from PIL import Image
import io
import torch
from albumentations import A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF
import os
from src.config import ModelConfig
from src.model.model import UNet

# Utility Functions
def testImage(model, image):
    model.eval()

    transform = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH, always_apply=True),  # Resize to a common size
        A.Normalize(*config.stats),
        ToTensorV2()
    ])
    
    image = transform(image).unsqueeze(0) 
    # device = next(model.parameters()).device  # Get the device of the model
    # image = image.to(device)

    with torch.no_grad():
        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()
        preds = preds.cpu().squeeze(0)
    
    return preds

def save_mask(mask):
    buffer = io.BytesIO()
    mask.save(buffer, format='PNG')
    # Rewind the buffer's file pointer to the beginning so it's ready for reading
    buffer.seek(0)

# Main Application
WORKING_DIR = os.path.join(os.path.dirname(__file__), 'source')
app = Flask(__name__)
model = UNet(3,1)
model.load_state_dict(torch.load(os.path.join(WORKING_DIR, 'models', 'final-model.pth')))
config = ModelConfig()

# Routers
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    image = Image.open(io.BytesIO(file.read()))
    tensor = testImage(model, image)  # Define your image processing function
    mask = TF.to_pil_image(tensor)
    mask = save_mask(mask)
    return send_file(mask, mimetype='image/png')


# Server starting at localhost:5000
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # Runs on http://localhost:5000
