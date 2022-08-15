"""Converting .json from yolo into .pbn for pythondds."""
import abc
import json
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn.cluster
import sympy


CARD_CLASSES = [
    '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', '10s', 'Js', 'Qs', 'Ks', 'As',
    '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', '10c', 'Jc', 'Qc', 'Kc', 'Ac',
    '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d', 'Jd', 'Qd', 'Kd', 'Ad',
    '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', 'Jh', 'Qh', 'Kh', 'Ah'
]

QUADRANT_TOP = "top"
QUADRANT_BOTTOM = "bottom"
QUADRANT_LEFT = "left"
QUADRANT_RIGHT = "right"
MARGIN = "margin"


class ICoreFinder(abc.ABC):

    @abc.abstractmethod
    def find_core(records: Iterable):
        """Find core objects on 2-D plane represented by (YOLO) relative coordinates."""
        pass


class DealConverter:
    QUADRANT_MARGIN_WIDTH = 0.05

    card: pd.DataFrame
    card_: pd.DataFrame

    def __init__(self, core_finder: ICoreFinder):
        self.card = None
        self.core_finder = core_finder

        self.card_ = None

    def read_yolo(self, path):
        with open(path, 'r') as f:
            yolo_json = json.load(f)
        self.card = (
            pd.json_normalize(yolo_json[0]['objects'])  # image has one frame only
                .rename(columns={"relative_coordinates.center_x": "center_x",
                                 "relative_coordinates.center_y": "center_y",
                                 "relative_coordinates.width": "width",
                                 "relative_coordinates.height" :"height"})
        )

    def report_missing_and_fp(self):
        # report missing
        detected_classes = set(self.card.name)
        missing_classes = [name for name in CARD_CLASSES if name not in detected_classes]
        print("Missing cards:", missing_classes)

        # report FP
        fp_classes = self.card.name.value_counts()[lambda s: s > 2].index.tolist()
        print("FP cards:", fp_classes)

    def dedup(self, smart=False):
        if smart:
            self.card_ = self._dedup_smart()
        else:
            self.card_ = self._dedup_simple()

    # two cases after dedup
    def assign(self):
        """Case 1: everything is perfect -> work on assigning cards to four hands"""
        # TODO test `assign` with deal3-manual-edit.json
        self._divide_to_quadrants()

        self._mark_core_objs()
        self._drop_core_duplicates()

        remaining = self._list_remaining_objs()
        while not remaining.empty:
            obj_idx, quadrant = self._find_closest_obj(remaining)

            self._assign_one_obj(obj_idx, quadrant)
            remaining = self._drop_assigned(obj_idx, remaining)

    def infer_missing(self):
        """Case 2: missing cards -> attempt to infer"""
        pass  # *lower priority

    def write_pbn(self, path):
        pass

    def _dedup_simple(self):
        """Dedup in a simple way, only keeping the one with highest confidence."""
        return (self.card
                    .sort_values('confidence')
                    .drop_duplicates(subset='name', keep='last'))

    def _dedup_smart(self):
        """Dedup in a smart way.

        steps:  # YL: got a feeling this is to complex
        1. find out dup pairs whose dist between a range: not too far nor too close
        2. remove dup objs by keeping one with highest conf; could lead to removal of valid objs
        3. append back good dups found in 1. and leave them for assign to decide
        """
        dist = self._calc_symbol_pair_dist()

        #　[0.1, 0.3] is reasonable range of symbol pair dist on same card
        densest_dist = self._find_densest(dist.query('0.1 <= dist_ <= 0.3').dist_)
        good_dup = self._get_good_dup(self.card, dist, densest_dist)

        return (pd.concat([self._dedup_simple(),
                           good_dup])
                    .drop_duplicates())

    def _divide_to_quadrants(self):
        """Divide cards to four quadrants before finding the core objs in each."""
        self.card_ = (
            self.card_
                .pipe(self._mark_marginal, width=self.QUADRANT_MARGIN_WIDTH)
                .assign(quadrant=lambda df: df.apply(self._calc_quadrant, axis=1))
        )

    def _mark_core_objs(self):
        """Find core objects for all four quadrants by adding col 'is_core'."""
        core_objs_t = self._find_quadrant_core_objs(QUADRANT_TOP)
        core_objs_b = self._find_quadrant_core_objs(QUADRANT_BOTTOM)
        core_objs_l = self._find_quadrant_core_objs(QUADRANT_LEFT)
        core_objs_r = self._find_quadrant_core_objs(QUADRANT_RIGHT)

        self.card_["is_core"] = (
            pd.concat([core_objs_t, core_objs_b, core_objs_l, core_objs_r])
                .reindex(self.card_.index, fill_value=False)  # for marginal objs
        )

    def _drop_core_duplicates(self):
        """Drop objects duplicated with core objects."""
        pass

    def _list_remaining_objs(self) -> pd.DataFrame:
        pass

    def _find_closest_obj(self, remaining):
        """Find the closest obj to any of the *qualifying hands*.

        Qualification: each hand can have 13 objects at most."""
        pass

    def _assign_one_obj(self, obj_idx, quadrant):
        """Assign the object to a hand indicated by `quadrant`, by adding col 'hand'."""
        pass

    def _drop_assigned(self, obj_idx, remaining) -> pd.DataFrame:
        """Drop assigned object from `remaining`."""
        pass

    def _calc_symbol_pair_dist(self):
        card_filtered = (
            self.card[['name', 'confidence', 'center_x', 'center_y']]
                .rename(columns=lambda s: s.split('_')[-1])
                .query('confidence >= 0.7')  # debug
        )

        pair_dist = (
            card_filtered
                # make uniq names for pair-wise dists
                .assign(group_rank=lambda df:
                            df.groupby('name').x
                                .transform(lambda s: s.rank(method='first')))
                .assign(uniq_name=lambda df:
                            df.name.str
                                .cat([df.group_rank.astype(int).astype(str)], sep='_'))
                .drop(columns=['group_rank']).set_index('uniq_name')

                .pipe(self._make_pair_wise)
                .query('name_1 == name_2')
                .assign(dist_=lambda df: df.apply(
                    lambda row: self._euclidean_dist(row.x_1, row.y_1, row.x_2, row.y_2),
                    axis=1))
        )
        return pair_dist

    @staticmethod
    def _find_densest(X, min_size=3):
        """Find densest subset of dists.

        As part of smart dedup, the idea is to find the 'mode' of
        all pair-wise distances, in order to filter out bad pairs.

        - `eps` tuned for dist between two symbols on the same card
        - only works for 1-d array currently"""
        X = np.array(X).reshape(-1, 1)
        dbscan = sklearn.cluster.DBSCAN(eps=0.01, min_samples=min_size)
        clt_id = dbscan.fit(X).labels_

        clustered = pd.DataFrame(dict(dist_=X.ravel(), clt_id_=clt_id))
        if clustered.clt_id_.max() > 0:
            print("WARNING: more than one cluster found")
        densest = clustered.loc[clustered.clt_id_ == 0, 'dist_']
        return densest

    @staticmethod
    def _get_good_dup(card, dist, densest_dist):
        good_pair = dist.loc[dist.dist_.isin(densest_dist)].filter(regex='^(name|x|y)')
        is_good_dup = pd.concat([
            good_pair.filter(regex='_1$').rename(columns=lambda s: s.replace('_1', '')),
            good_pair.filter(regex='_2$').rename(columns=lambda s: s.replace('_2', '')),
        ])

        all_dup = card.loc[card.name.isin(card.name.value_counts().loc[lambda s: s > 1].index)]

        return (all_dup
                    .merge(is_good_dup.set_index(['name', 'x', 'y']),
                           left_on=['name', 'center_x', 'center_y'],
                           right_index=True))

    @staticmethod
    def _make_pair_wise(df: pd.DataFrame):
        # pair wise -> `from_product`
        midx = pd.MultiIndex.from_product([df.index, df.index], names=['n1', 'n2'])
        # need paired x & y info -> reindex and align with `concat`
        pair = pd.concat(
            [df.add_suffix('_1').reindex(midx, level='n1'),
             df.add_suffix('_2').reindex(midx, level='n2')], axis=1)
        # order doesn't matter -> combination instead of permutation
        return pair.loc[midx[midx.get_level_values(0) < midx.get_level_values(1)]]

    @staticmethod
    def _euclidean_dist(x1, y1, x2, y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    @staticmethod
    def _mark_marginal(card: pd.DataFrame, width) -> pd.DataFrame:
        """Mark a card as marginal based on (x, y).

        OK to ignore the top-left positioned origin, due to symmetricity.
        """
        line_up = sympy.Line((0, 0), (1, 1))
        line_dn = sympy.Line((0, 1), (1, 0))

        def _calc_dist_to_border(row: pd.Series):
            dist1 = float(line_up.distance((row.center_x, row.center_y)).evalf())
            dist2 = float(line_dn.distance((row.center_x, row.center_y)).evalf())
            return min(dist1, dist2)

        return (card.assign(_dist_to_border=card.apply(_calc_dist_to_border, axis=1))
                    .assign(is_marginal=lambda df: df._dist_to_border <= width)
                    .drop(columns="_dist_to_border"))

    @staticmethod
    def _calc_quadrant(c: pd.Series):
        """Determine quadrant of cards based on (x, y) and whether marginal."""
        if c.is_marginal:
            return MARGIN

        # Note: origin is at top left corner, instead of bottom left
        if c.center_y > c.center_x and 1 - c.center_y < c.center_x:
            return QUADRANT_BOTTOM
        if c.center_y < c.center_x and 1 - c.center_y > c.center_x:
            return QUADRANT_TOP
        if c.center_y < c.center_x and 1 - c.center_y < c.center_x:
            return QUADRANT_RIGHT
        if c.center_y > c.center_x and 1 - c.center_y > c.center_x:
            return QUADRANT_LEFT

    def _find_quadrant_core_objs(self, quadrant) -> pd.Series:
        """Find core objects for a specific quadrant"""
        #  a method: list Top Q objs; transform to records; core_finder function call; transform to series with the right index
        subframe = self.card_.loc[lambda df: df.quadrant == quadrant, ["center_x", "center_y"]]

        _obj_records = subframe.itertuples(index=False)
        bool_seq = self.core_finder.find_core(_obj_records)

        return pd.Series(bool_seq, index=subframe.index)


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

    def find_core(self, records: Iterable):
        self.dbscan.fit(list(records))

        _labels = self.dbscan.labels_
        is_core = [label != self.DBSCAN_NOISY_LABEL for label in _labels]
        return is_core
