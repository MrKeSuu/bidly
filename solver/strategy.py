import abc
from typing import Iterable, Tuple

import sklearn


Coord = Tuple[float, float]


class ICoreFinder(abc.ABC):

    @abc.abstractmethod
    def find_core(self, coords: Iterable[Coord]):
        """Find core objects on 2-D plane represented by (YOLO) relative coordinates."""
        pass


class CoreFinderDbscan(ICoreFinder):
    """CoreFinder with DBSCAN clutering."""
    EPS = 0.1  # assumes records in relative coords valued in (0, 1)
    MIN_SAMPLES = 3

    DBSCAN_NOISY_LABEL = -1

    def __init__(self):
        self.dbscan = sklearn.cluster.DBSCAN(
            eps=self.EPS,
            min_samples=self.MIN_SAMPLES,
        )

    def find_core(self, coords: Iterable[Coord]):
        self.dbscan.fit(list(coords))

        _labels = self.dbscan.labels_
        is_core = [label != self.DBSCAN_NOISY_LABEL for label in _labels]
        return is_core
