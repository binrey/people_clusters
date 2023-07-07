import os
import pickle
import logging
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from pathlib import Path


def get_logger(logger_name):
   logger = logging.getLogger(logger_name)
   logger.setLevel(logging.DEBUG)


class DataLoader(ABC):
    @abstractmethod
    def read_gt(self, data_path):
        pass

    @staticmethod
    def read_encodings(self, data_path):
        pass

    @staticmethod
    def show_stats(self):
        pass


class MyDataLoader(DataLoader):
    def __init__(self):
        pass

    def read_gt(self, data_path):
        self.data = pd.read_csv(data_path)
        enc = LabelEncoder()
        self.data["cluster_num"] = enc.fit_transform(self.data.cluster_id)
        logging.info(f"Loaded {len(self.data)} images")
    
    def read_encodings(self, data_path):
        self.encodings = pickle.loads(open(data_path, "rb").read())
        logging.info(f"Loaded {len(self.encodings)} encodings")

    def show_stats(self):
        counter = Counter(self.data.cluster_num)
        return plt.plot(range(len(counter)), sorted(counter.values(), reverse=True), ".-")
    
    def get_xy(self):
        gt, x = [], []
        for enc in self.encodings:
            if len(enc["encoding"]):
                key = Path(enc["imagePath"]).name
                tmp = self.data[self.data.file_name == key]
                if len(tmp):
                    gt.append(tmp.cluster_num.values[0])
                    x.append(enc["encoding"])
        x, gt = np.array(x), np.array(gt)
        logging.info(f"Sync images with faces. Encodings x: {x.shape}, labeled clusters y:{gt.shape}")
        return x, gt


image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("puk")