import os
import pandas as pd
import PIL.Image as Image
import numpy as np
import tensorflow as tf


class GTSRB_Test_Loader:
    def __init__(self, TEST_PATH, TEST_GT_PATH='evaluation/GTSRB_Test_GT.csv'):
        self.df = pd.read_csv(TEST_GT_PATH, sep=';')
        self.TEST_PATH = TEST_PATH
        self.img_size = (32, 32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = os.path.join(self.TEST_PATH, row['Filename'])
        img = Image.open(filename)
        

        img = img.resize(self.img_size)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        label = int(row['ClassId'])
        return img, label, filename
