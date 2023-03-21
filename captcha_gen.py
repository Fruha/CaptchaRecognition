from captcha.image import ImageCaptcha
import string
from tqdm import tqdm

import os
import random
import json
import glob

from utils import save_in_json
from utils import load_from_json

import multiprocessing
from multiprocessing import Pool, freeze_support


class CapthchaGenerator:

    def __init__(self, count, min_length=1, max_length=8, val_size=0.1, char_set=string.ascii_lowercase+string.digits, cpu_count=None, height=90, width=280):
        self.count = count
        self.min_length = min_length
        self.max_length = max_length
        self.val_size = val_size
        self.char_set = char_set
        self.height = height
        self.width = width
        if cpu_count is None:
            self.cpu_count = 1
        else:
            self.cpu_count = cpu_count

    def _generate_image(self, text, path):
        image = ImageCaptcha(self.width, self.height, fonts=self.ttfs)
        image.write(text, path)

    def _generate_folder_dataset(self, path_folder, count):
        labels = load_from_json(f'{path_folder}/labels.json')
        new_texts = []
        new_labels = []
        for i in range(0, count):
            text = ''.join(random.choices(
                self.char_set, k=random.randint(self.min_length, self.max_length)))
            labels[f'{i}'] = text
            new_texts.append(text)
            new_labels.append(str(i))

        with Pool(processes=self.cpu_count) as pool:
            pool.starmap(self._generate_image, tqdm(
                zip(new_texts, (f'{path_folder}/{i}.png' for i in labels)), total=len(new_texts)))

        save_in_json(labels, f'{path_folder}/labels.json')

    def generate_dataset(self, path_to_images='images', path_to_ttfs = None) -> None:
        """
        Create dataset train and validation in using folder
        Arguments:
            path_to_images: path to folder
        """
        if path_to_ttfs is None:
            self.ttfs = None
        else:
            self.ttfs = glob.glob(f'{path_to_ttfs}/*.ttf')

        os.makedirs(f'{path_to_images}', exist_ok=True)
        os.makedirs(f'{path_to_images}/train', exist_ok=True)
        os.makedirs(f'{path_to_images}/val', exist_ok=True)

        count_train_new = int(self.count*(1-self.val_size))
        count_val_new = int(self.count*self.val_size)

        self._generate_folder_dataset(
            f'{path_to_images}/train', count_train_new)
        self._generate_folder_dataset(f'{path_to_images}/val', count_val_new)


if __name__ == '__main__':
    freeze_support()
    char_set = string.digits + string.ascii_lowercase
    cg = CapthchaGenerator(count=120000, min_length=3, max_length=10,
                           char_set=char_set, height=90, width=280, val_size=0.1, cpu_count=2)
    cg.generate_dataset(path_to_images='images', path_to_ttfs=None)
