import abc
from typing import Iterable, Tuple

import numpy as np
import scipy.spatial

from solver import dbscan


Coord = Tuple[float, float]


class ICoreFinder(abc.ABC):

    @abc.abstractmethod
    def find_core(self, coords: Iterable[Coord]):
        """Find core objects on 2-D plane represented by (YOLO) relative coordinates."""
        pass


class CoreFinderDbscanPy(ICoreFinder):
    """CoreFinder with pure-python DBSCAN clutering."""
    EPS = 0.1  # assumes records in relative coords valued in (0, 1)
    MIN_SAMPLES = 3

    DBSCAN_NOISY_LABEL = None

    def find_core(self, coords: Iterable[Coord]):
        X = np.asarray(list(coords)).T  # transpose required by the pure-py dbscan code
        _labels = dbscan.dbscan(X, eps=self.EPS, min_points=self.MIN_SAMPLES)
        is_core = [label != self.DBSCAN_NOISY_LABEL for label in _labels]
        return is_core


class ILinkage(abc.ABC):

    @abc.abstractmethod
    def calc_distance(self, x, y, coords: Iterable[Coord]):
        """Calculate distance between a single point (x, y) and `coords`."""
        pass

    def _calc_pointwise_distances(self, x, y, coords: Iterable[Coord]):
        return [scipy.spatial.distance.euclidean([x, y], [x2, y2])
                for x2, y2 in coords]


class SingleLinkage(ILinkage):
    """Single Linkage (SL) calcs distance using the closest point."""

    def calc_distance(self, x, y, coords: Iterable):
        pointwise_distances = self._calc_pointwise_distances(x, y, coords)
        return min(pointwise_distances)
