import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def mean(data_path):
    count = 0
    s = np.zeros((512, 512, 3), dtype=np.float64)
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith('.jpg'):
            filename = os.path.join(data_path, filename)
            img = Image.open(filename)
            img = img.resize((512, 512))
            try:
                img_np = np.asarray(img).reshape((512, 512, 3))
            except:
                continue
            img_np = (img_np / 127.5) - 1.
            s += img_np
            count += 1
    return s / count


def std(data_path, mn=None):
    if mn is None:
        mn = mean(data_path)
    count = 0
    s = np.zeros((512, 512, 3), dtype=np.float64)
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith('.jpg'):
            filename = os.path.join(data_path, filename)
            img = Image.open(filename)
            img = img.resize((512, 512))
            try:
                img_np = np.asarray(img).reshape((512, 512, 3))
            except:
                continue
            img_np = (img_np / 127.5) - 1.
            s += np.power(img_np - mn, 2)
            count += 1
    return np.sqrt(s / count)
