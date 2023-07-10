from abc import ABC, abstractmethod
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import logging
import numpy as np
from sklearn.metrics import v_measure_score
from utils import DataLoader
import cv2
from matplotlib import pyplot as plt
from pathlib import Path


class ClusterUsers(ABC):
    def __init__(self, data_loader:DataLoader) -> None:
        self.x, self.y, self.ids, self.boxes = data_loader.get_xy()

    @abstractmethod
    def fit(self, x, min_samples_range, eps_range):
        pass

    @abstractmethod
    def validate(self, ground):
        pass

    @property
    def labs_pred(self):
        return self._labs_pred


class MyClusterUsers(ClusterUsers):
    def __init__(self, data_loader:DataLoader) -> None:
        super().__init__(data_loader)

    def fit(self, min_samples_range, eps_range):
        output = []
        logging.debug("Start optimize DBSCAN parameters...")
        for ms in min_samples_range:
            for ep in eps_range:
                labs = DBSCAN(min_samples=ms, eps = ep).fit(self.x).labels_
                if sum(labs):
                    score = silhouette_score(self.x, labs)
                    output.append((ms, ep, score))

        min_samples, eps, score = sorted(output, key=lambda a:a[-1])[-1]
        logging.info(f"Best silhouette_score: {score}")
        logging.info(f"min_samples: {min_samples}")
        logging.info(f"eps: {eps}")

        logging.debug("Start DBSCAN with optimal parameters...")
        self._labs_pred = DBSCAN(min_samples=min_samples, eps = eps).fit(self.x).labels_
        labelIDs = np.unique(self._labs_pred)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        logging.info("Unique faces: {}".format(numUniqueFaces))

    def validate(self):
        score = v_measure_score(self.y, self._labs_pred)
        logging.info(f"V-score: {score:4.2f}")
        return score
    
    def draw(self, imgs_path, ncols=5, img_size=(60, 60)):
        if type(imgs_path) is str:
            imgs_path = Path(imgs_path)
        clusters = set(self._labs_pred)
        nclust = len(clusters)

        for r in clusters:
            mask = self._labs_pred == r
            curr_names = [name for m, name in zip(mask, self.ids) if m]
            curr_boxes = [box for m, box in zip(mask, self.boxes) if m]
            nshot = 0
            for c in range(ncols):
                if nshot < sum(mask):
                    img_path = imgs_path / curr_names[nshot]
                    b = curr_boxes[nshot]
                    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                    img_crop = img[b[1]:b[3], b[0]:b[2]]
                    if min(img_crop.shape) == 0:
                        img_crop = img
                    img_crop = cv2.resize(img_crop, img_size)
                    if img_crop.ndim == 2:
                        img_crop = np.dstack([img_crop]*3)
                else:
                    img_crop = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
                nshot += 1
                if c == 0:
                    row = img_crop
                else:
                    row = np.hstack([row, img_crop])  
            if r == 0:
                wall = row
            else:
                wall = np.vstack([wall, row])


        fig, ax = plt.subplots(figsize=(2*nclust, ncols*2))
        plt.imshow(wall)
        plt.axis("off");