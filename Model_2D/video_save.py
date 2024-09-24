# ******************************************************************************
# @author: L. I. Nurtdinova
# ******************************************************************************

import os

import cv2
import numpy as np
from PIL import Image


def save(folder_path, output_path):
    output_path = output_path + '.mp4'
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    images = []

    # Уменьшение разрешения для уменьшения размера видео
    target_size = (2565 // 2, 950 // 2)  # Можно еще уменьшить, если нужно

    for file in files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path)
        img = img.resize(target_size)
        images.append(img)

    # Используем кодек X264 для лучшего сжатия
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    fps = 25  # Уменьшаем FPS для уменьшения размера

    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    for img in images:
        # Преобразование изображения из PIL в формат, совместимый с OpenCV
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(img_cv)

    out.release()
    print(f'Анимация сохранена как {output_path}')


folder_path, output_path = '001', '001_1-1-5-without'
save(folder_path, output_path)
folder_path, output_path = '002', '001_3-1-5-without'
save(folder_path, output_path)
folder_path, output_path = '003', '001_3-0.5-5-without'
save(folder_path, output_path)
folder_path, output_path = '004', '001_3-0.5-10-without'
save(folder_path, output_path)

folder_path, output_path = '005', '001_1-1-5-with'
save(folder_path, output_path)
folder_path, output_path = '006', '001_3-1-5-with'
save(folder_path, output_path)
folder_path, output_path = '007', '001_3-0.5-5-with'
save(folder_path, output_path)
folder_path, output_path = '008', '001_3-0.5-10-with'
save(folder_path, output_path)
