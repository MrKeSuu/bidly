import abc
from typing import Iterable, Tuple

import scipy.spatial
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
