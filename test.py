import os
from model import Dinomaly
from util import compute_anomaly_map, visualize
import torch
from PIL import Image
from torchvision import transforms


"""
Test All Images.
Save test result.
"""


model_weights_path = '/home/ohjihoon/바탕화면/dino/train_result/Col_v2/weight.pth'
image_folder = '/home/ohjihoon/바탕화면/dino/03_03_Anomaly_dataset/patch/Bottom-pin_auto_2'
result_dir = '/home/ohjihoon/바탕화면/dino/03_03_Anomaly_test/Bottom-pin_auto_2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def image_transform(image):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]

    transform_img = transforms.Compose([
        #transforms.Grayscale(num_output_channels=3),  # Grayscale → RGB (3채널)
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])
    
    return transform_img(image)

model = Dinomaly(weight=model_weights_path)
model.model.to(device)
model.model.eval()

os.makedirs(result_dir, exist_ok=True) 
image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[-1].lower() in image_extensions]

for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    

    image = Image.open(image_path).convert('RGB')
    original_image = image
    image_size = (image.height, image.width)

    image = image_transform(image)  # (C, H, W)
    image = image.unsqueeze(0).to(device)  # (1, C, H, W)
    

    en, de = model.model(image)
    anomaly_map = compute_anomaly_map(en, de, out_size=(image_size)) 
    original_image_tensor = transforms.ToTensor()(original_image).unsqueeze(0)  # (1, C, H, W)
    

    visualize(original_image_tensor, anomaly_map, save_path=result_dir, image_name=image_name)
    print(f'Processed: {image_name}')

print("All images processed successfully.")
