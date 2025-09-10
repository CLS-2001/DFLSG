import os
import cv2
import numpy as np
from tqdm import tqdm

def add_gaussian_noise(image, noise_ratio=0.1):
    image = image.astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_ratio, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1.0)
    return (noisy_image * 255).astype(np.uint8)

def process_all_images(input_root, output_root, noise_ratio=0.1):

    os.makedirs(output_root, exist_ok=True)

    for subdir in os.listdir(input_root):
        input_subdir = os.path.join(input_root, subdir)
        output_subdir = os.path.join(output_root, subdir)

        if not os.path.isdir(input_subdir):
            continue

        os.makedirs(output_subdir, exist_ok=True)

        for filename in tqdm(os.listdir(input_subdir), desc=f"Processing {subdir}"):
            input_path = os.path.join(input_subdir, filename)
            output_path = os.path.join(output_subdir, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Warning: Failed to read {input_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            noisy_image = add_gaussian_noise(image, noise_ratio)
            noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, noisy_image)

input_root = "/home/yangxiaohui/SCL/datasets/CUHK-PEDES/imgs"
output_root = "/home/yangxiaohui/SCL/datasets/CUHK-PEDES/imgs_noise10%"

process_all_images(input_root, output_root, noise_ratio=0.1)

