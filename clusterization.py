from abc import ABC, abstractmethod
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import logging
import numpy as np
from sklearn.metrics import v_measure_score



class ClusterUsers(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, x, min_samples_range, eps_range):
        pass

    @abstractmethod
    def validate(self, ground):
        pass


class MyClusterUsers(ClusterUsers):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x, min_samples_range, eps_range):
        output = []
        logging.debug("Start optimize DBSCAN parameters...")
        for ms in min_samples_range:
            for ep in eps_range:
                labs = DBSCAN(min_samples=ms, eps = ep).fit(x).labels_
                if sum(labs):
                    score = silhouette_score(x, labs)
                    output.append((ms, ep, score))

        min_samples, eps, score = sorted(output, key=lambda x:x[-1])[-1]
        logging.info(f"Best silhouette_score: {score}")
        logging.info(f"min_samples: {min_samples}")
        logging.info(f"eps: {eps}")

        logging.debug("Start DBSCAN with optimal parameters...")
        self.labs_pred = DBSCAN(min_samples=min_samples, eps = eps).fit(x).labels_
        labelIDs = np.unique(self.labs_pred)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        logging.info("Unique faces: {}".format(numUniqueFaces))

    def validate(self, ground):
        score = v_measure_score(ground, self.labs_pred)
        logging.info(f"V-score: {score:4.2f}")
        return score