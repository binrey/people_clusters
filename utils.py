import os
import pickle
import logging
import cv2
import json
from typing import Any
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from pathlib import Path
from random import choices


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
        self.data = pd.read_csv(data_path, usecols=["cluster_id", "file_name"])
        enc = LabelEncoder()
        self.data["cluster_num"] = enc.fit_transform(self.data.cluster_id)
        logging.info(f"Loaded {len(self.data)} images")
    
    def read_encodings(self, data_path):
        self.encodings = json.load(open(data_path, "r"))
        logging.info(f"Loaded {len(self.encodings)} encodings")

    def show_stats(self):
        counter = Counter(self.data.cluster_num)
        plt.plot(range(len(counter)), sorted(counter.values(), reverse=True), ".-")
        plt.xlabel("cluster number")
        plt.ylabel("elements count")
        plt.show();
    
    def get_xy(self):
        gt, x, ids, boxes = [], [], [], []
        for key, enc in self.encodings.items():
            if len(enc["encoding"]):
                key = Path(key).name
                tmp = self.data[self.data.file_name == key]
                if len(tmp):
                    gt.append(tmp.cluster_num.values[0])
                    x.append(enc["encoding"])
                    ids.append(key)
                    boxes.append(enc["loc"])
        x, gt, boxes = np.array(x), np.array(gt), np.array(boxes)
        logging.info(f"Sync images with faces. Encodings x: {x.shape}, labeled clusters y:{gt.shape}")
        return x, gt, ids, boxes


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


class ClustersDrawer:
    def __init__(self, imgs_path, img_size=(60, 60), ncols=10) -> None:
        self.img_size = img_size
        self.ncols = ncols
        self.imgs_path = imgs_path

    def __call__(self, labs_pred, img_names) -> None:
        clusters = set(labs_pred)
        nclust = len(clusters)

        for r in clusters:
            mask = labs_pred == r
            curr_names = [name for m, name in zip(mask, img_names) if m]
            nshot = 0
            for c in range(self.ncols):
                if nshot < len(curr_names):
                    img_path = self.imgs_path / curr_names[nshot]
                    # b = boxes[nshot]
                    img = np.array(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB))
                    img_crop = img#[b[0]:b[2], b[3]:b[1]]
                    if min(img_crop.shape) == 0:
                        img_crop = img
                    img_crop = cv2.resize(img_crop, self.img_size)
                    if img_crop.ndim == 2:
                        img_crop = np.dstack([img_crop]*3)
                else:
                    img_crop = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                nshot += 1
                if c == 0:
                    row = img_crop
                else:
                    row = np.hstack([row, img_crop])  
            if r == 0:
                wall = row
            else:
                wall = np.vstack([wall, row])


        fig, ax = plt.subplots(figsize=(2*nclust, self.ncols*2))
        plt.imshow(wall)
        plt.axis("off");