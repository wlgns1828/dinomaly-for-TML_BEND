import os
from model import Dinomaly
from util import StableAdamW, global_cosine_hm_percent
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch.nn as nn



class Datasets(Dataset):
    def __init__(self, images_dir, target_size=(448, 448)):
        self.images_dir = images_dir
        self.target_size = target_size


        self.image_files = []
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    self.image_files.append(os.path.join(root, f))

        self.image_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]  
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)
        return image


def train(data_path,
        output_dir,
        total_iters=2000,
        batch_size=4, 
    ):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = Datasets(images_dir=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = Dinomaly()
    model.model.to(device)
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    optimizer = StableAdamW([{'params': model.trainable.parameters()}],
                    lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    print('train image number:{}'.format(len(dataset)))
    it = 0

    for epoch in range(int(np.ceil(total_iters / len(dataloader)))):
        model.model.train()
        loss_list = []
            
        for img in dataloader:
            img = img.to(device)
            
            en, de = model.model(img)
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
                
            if (it + 1) % 10 == 0:
                print(f"Iter {it + 1}: Loss = {loss.item()}")
                torch.save(model.model.state_dict(), os.path.join(output_dir, 'weight.pth'))
            it += 1
            if it == total_iters:
                break

    

    
if __name__ == "__main__":

    train(data_path='/home/ohjihoon/바탕화면/dino/patched_train_image/Col_image_v1', 
        output_dir='/home/ohjihoon/바탕화면/dino/train_result_no_nomalize/Col_v1', 
        total_iters=5000, 
        batch_size=4)
