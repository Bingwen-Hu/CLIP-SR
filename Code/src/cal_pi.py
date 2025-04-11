import os
import torch
import pyiqa
from tqdm import tqdm
from torchvision import transforms
from PIL import Image


def calculate_average_metrics(image_folder):
    # 加载 NIQE 和 PI 评价指标
    niqe_metric = pyiqa.create_metric('niqe').cuda()
    pi_metric = pyiqa.create_metric('pi').cuda()

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    total_images = len(image_files)

    if total_images == 0:
        print("No JPG images found in the folder.")
        return

    niqe_scores = []
    pi_scores = []

    transform = transforms.ToTensor()

    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_folder, img_file)
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).cuda()

        niqe_scores.append(niqe_metric(img_tensor).item())
        pi_scores.append(pi_metric(img_tensor).item())

    avg_niqe = sum(niqe_scores) / total_images
    avg_pi = sum(pi_scores) / total_images

    print(f"Average NIQE: {avg_niqe:.4f}")
    print(f"Average PI: {avg_pi:.4f}")

    return avg_niqe, avg_pi


if __name__ == "__main__":
    image_folder = "/opt/data/private/carr/code/imgs/bird/test/TEST_bird_256_2025_03_17_11_06_02"
    calculate_average_metrics(image_folder)