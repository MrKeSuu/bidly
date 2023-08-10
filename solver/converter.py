"""Converting .json from yolo into .pbn for pythondds."""
import abc
import json
import logging as log
import pathlib
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import scipy.spatial

from solver import dbscan
from solver import strategy
from solver import util


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

HAND_N = 'north'
HAND_S = 'south'
HAND_W = 'west'
HAND_E = 'east'
HAND_MAP = {
    QUADRANT_TOP: HAND_N,
    QUADRANT_BOTTOM: HAND_S,
    QUADRANT_LEFT: HAND_W,
    QUADRANT_RIGHT: HAND_E
}
HAND_SHORTNAME_MAP = {
    HAND_N: 'N',
    HAND_S: 'S',
    HAND_W: 'W',
    HAND_E: 'E'
}
SUIT_S, SUIT_H, SUIT_D, SUIT_C = "s", "h", "d", "c"
RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]


util.setup_basic_logging()


class IPredReader(abc.ABC):
    @abc.abstractmethod
    def read(self, src) -> pd.DataFrame:
        pass


class IAssigner(abc.ABC):
    core_finder: strategy.ICoreFinder
    linkage: strategy.ILinkage

    def __init__(self, core_finder, linkage) -> None:
        self.core_finder = core_finder
        self.linkage = linkage

    @abc.abstractmethod
    def assign(self, transformed_cards) -> List[Dict]:
        pass


class DealConverter:
    card: pd.DataFrame
    card_: pd.DataFrame

    reader = IPredReader
    assigner = IAssigner

    def __init__(self, reader, assigner):
        self.card = None
        self.reader = reader
        self.assigner = assigner

        self.card_ = None

    def read(self, path):
        self.card = self.reader.read(path)

        vcnt = self.card.name.value_counts()
        msg = ','.join(f"{name} {f'({cnt})' if cnt > 1 else ''}" for name, cnt in vcnt.iteritems())
        log.info("Read %s detected objs: %s", len(self.card), msg)

    def report_missing_and_fp(self):
        # report missing
        detected_classes = set(self.card.name)
        missing_classes = [name for name in CARD_CLASSES if name not in detected_classes]
        print("Missing cards:", missing_classes)

        # report FP
        fp_classes = self.card.name.value_counts()[lambda s: s > 2].index.tolist()
        print("FP cards:", fp_classes)

        return missing_classes, fp_classes

    def dedup(self, smart=False):
        log.info("Running dedup (smart=%s) ..", smart)
        if smart:
            self.card_ = self._dedup_smart()
        else:
            self.card_ = self._dedup_simple()

    # two cases after dedup
    def assign(self, transformed_cards):
        """Case 1: everything is perfect -> work on assigning cards to four hands"""
        log.info("Assigning cards to hands..")
        assigned_cards = self.assigner.assign(transformed_cards)
        return assigned_cards

    def infer_missing(self):
        """Case 2: missing cards -> attempt to infer"""
        pass  # *lower priority

    def write_pbn(self, path):
        hands = self._build_pbn_hands()
        deal = self._build_pbn_deal(hands)
        self._write_pbn_deal(deal, path)

    def format_pbn(self, assigned_cards) -> bytes:
        log.info("Formatting hand to PBN..")
        self.card_ = pd.DataFrame(assigned_cards)  # not best practice
        hands = self._build_pbn_hands()
        deal = self._build_pbn_deal(hands)
        return deal.encode("ascii")

    def list_assigned_cards(self):
        return self.assigner.objs[["name", "hand"]].dropna(subset=["hand"]).to_dict('records')

    def _dedup_simple(self):
        """Dedup in a simple way, only keeping the one with highest confidence."""
        return (self.card
                    .sort_values('confidence')
                    .drop_duplicates(subset='name', keep='last'))

    def _dedup_smart(self):
        """Dedup in a smart way.

        steps:  # YL: got a feeling this is too complex
        1. find out dup pairs whose dist between a range: not too far nor too close
        2. remove dup objs by keeping one with highest conf; could lead to removal of valid objs
        3. append back good dups found in 1. and leave them for assign to decide
        """
        dist = self._calc_symbol_pair_dist()

        #　[0.1, 0.3] is reasonable range of symbol pair dist on same card
        densest_dist = self._find_densest(dist.query('0.1 <= dist_ <= 0.3').dist_)
        good_dup = self._get_good_dup(self.card, dist, densest_dist)

        log.debug("Found %s 'good' dups: %s", len(good_dup), good_dup[["name"]].to_dict("records"))
        # TODO From later experiments, here we might want to replace with mean of each pair.
        return (pd.concat([self._dedup_simple(),
                           good_dup])
                    .drop_duplicates())

    def _build_pbn_hands(self):
        """Build four hands according to pbn format. (See deals.py)

        Now only supports a 4-tuple of (W, N, E, S)."""
        hands = (
            self._build_pbn_hand(HAND_W),
            self._build_pbn_hand(HAND_N),
            self._build_pbn_hand(HAND_E),
            self._build_pbn_hand(HAND_S)
        )
        return hands

    def _build_pbn_hand(self, hand_name):
        card_names = self.card_[self.card_.hand == hand_name].name.copy()
        if len(card_names) != 13:
            raise ValueError(f"{hand_name.upper()} did not have 13 cards")

        suits = (
            self._build_pbn_suit(card_names, SUIT_S),
            self._build_pbn_suit(card_names, SUIT_H),
            self._build_pbn_suit(card_names, SUIT_D),
            self._build_pbn_suit(card_names, SUIT_C)
        )
        formatted_hand = ".".join(suits)
        return formatted_hand

    @staticmethod
    def _build_pbn_suit(card_names: pd.Series, suit_name):
        cards = (
            card_names.to_frame()
                .assign(rank=lambda df: df.name.str.slice(stop=-1),
                        suit=lambda df: df.name.str.slice(start=-1))
                .assign(rank_order=lambda df: df["rank"].map(RANKS.index),
                        short_rank=lambda df: df["rank"].map(lambda r: "T" if r == "10" else r))
        )

        suit_cards = cards[cards.suit == suit_name]

        sorted_cards = suit_cards.sort_values("rank_order")

        formatted_suit = "".join(sorted_cards.short_rank)
        return formatted_suit

    def _build_pbn_deal(self, hands: tuple, first_hand=HAND_W):
        """Build deal given four hands according to pbn format. (See deals.py)"""
        if first_hand != HAND_W:
            raise NotImplementedError

        first = HAND_SHORTNAME_MAP[first_hand]

        assert len(hands) == 4
        formatted_hands = " ".join(hands)

        deal = f"{first}:{formatted_hands}"
        return deal

    def _write_pbn_deal(self, deal: str, path, force=False):
        """Write pbn-formatted deal to path."""
        path = pathlib.Path(path)
        if path.exists():
            if force:
                log.warning("Overwriting existing file at %s", path)
            else:
                raise IOError("File exists; use force=True to overwrite.")

        with open(path, 'w') as fo:
            fo.write(deal)

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
                    lambda s: scipy.spatial.distance.euclidean([s.x_1, s.y_1], [s.x_2, s.y_2]),
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
        X = np.array(X).reshape(1, -1)
        clt_id = dbscan.dbscan(X, eps=0.01, min_points=min_size)

        clustered = pd.DataFrame(dict(dist_=X.ravel(), clt_id_=clt_id))
        if clustered.clt_id_.max() > 1:
            print("WARNING: more than one cluster found")
        densest = clustered.loc[clustered.clt_id_ == 1, 'dist_']
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


class Yolo4Reader(IPredReader):
    def read(self, src):
        log.info("Reading from yolov4 pred: %s", src)
        with open(src, 'r') as f:
            yolo_json = json.load(f)

        return (
            pd.json_normalize(yolo_json[0]['objects'])  # image has one frame only
                .rename(columns={"relative_coordinates.center_x": "center_x",
                                 "relative_coordinates.center_y": "center_y",
                                 "relative_coordinates.width": "width",
                                 "relative_coordinates.height" :"height"})
        )

class Yolo5Reader(IPredReader):
    def read(self, src):
        """Read from `src`, records of detection."""
        return (
            pd.DataFrame(src)
                .assign(name=lambda df: df.class_id.map(CARD_CLASSES[::-1].__getitem__))  # dev
                .rename(columns={'x': 'center_x',
                                 'y': 'center_y',
                                 'w': 'width',
                                 'h': 'height'})
        )


class Assigner(IAssigner):
    QUADRANT_MARGIN_WIDTH = 0.05
    MAX_ASSIGNMENT_DISTANCE = 0.3  # from observations of misassignment, subject to change

    def assign(self, transformed_cards) -> List[Dict]:
        self._load(transformed_cards)

        self._divide_to_quadrants()

        self._mark_core_objs()
        self._drop_core_duplicates()
        self._assign_core_objs()

        remaining = self._list_remaining_objs()
        while not remaining.empty and self._hands_to_assign():
            obj_idx, hand = self._find_closest_obj(remaining)

            self._assign_one_obj(obj_idx, hand)
            remaining = self._drop_assigned(obj_idx, remaining)

        assigned_cards = self.list_assigned_cards()
        return assigned_cards

    #
    def _load(self, transformed_cards):
        self.objs = pd.DataFrame(transformed_cards)

    def _divide_to_quadrants(self):
        """Divide cards to four quadrants before finding the core objs in each."""
        self.objs = (
            self.objs
                .pipe(self._mark_marginal, width=self.QUADRANT_MARGIN_WIDTH)
                .assign(quadrant=lambda df: df.apply(self._calc_quadrant, axis=1))
        )
        log.debug("Divided to quadrants with %s marginal cards.",
                  self.objs.query("quadrant == 'margin'").shape[0])

    def _mark_core_objs(self):
        """Find core objects for all four quadrants by adding col 'is_core'."""
        core_objs_t = self._find_quadrant_core_objs(QUADRANT_TOP)
        core_objs_b = self._find_quadrant_core_objs(QUADRANT_BOTTOM)
        core_objs_l = self._find_quadrant_core_objs(QUADRANT_LEFT)
        core_objs_r = self._find_quadrant_core_objs(QUADRANT_RIGHT)

        self.objs["is_core"] = (
            pd.concat([core_objs_t, core_objs_b, core_objs_l, core_objs_r])
                .reindex(self.objs.index, fill_value=False)  # for marginal objs
        )
        log.info("Marked %s core objs in total", self.objs['is_core'].sum())

    def _drop_core_duplicates(self):
        """Drop objects duplicated with core objects, both inside core and outside core."""
        core = self.objs[self.objs.is_core]

        in_core_dups = core[core.duplicated("name")]
        self.objs = self.objs.drop(index=in_core_dups.index)

        out_core_dups = self.objs[lambda df: (~df.is_core) & (df.name.isin(core.name.values))]
        self.objs = self.objs.drop(index=out_core_dups.index)

        log.debug("Dropped %s duplicates inside core: %s",
                  len(in_core_dups), in_core_dups[["name", "quadrant"]].to_dict("records"))
        log.debug("Dropped %s duplicates outside core: %s",
                  len(out_core_dups), out_core_dups[["name", "quadrant"]].to_dict("records"))

    def _assign_core_objs(self):
        """Assign each core obj to hand based on quadrant, by adding col 'hand'. """
        def _to_hand(row: pd.Series):
            if not row.is_core:
                return None

            assert row.quadrant != MARGIN, f"Unexpected 'margin' core card: {row['name']}"
            return HAND_MAP[row.quadrant]

        self.objs["hand"] = self.objs.apply(_to_hand, axis=1)
        hand_card_cnts = self.objs.hand.value_counts()
        log.debug("Core card counts: %s", hand_card_cnts.to_dict())
        if hand_card_cnts.max() > 13:
            raise ValueError("Can't assign cards to hands; "
                             "try aligning hands better with the guide lines")

    def _list_remaining_objs(self) -> pd.DataFrame:
        """Return a df containing unassigned objs."""
        assert 'hand' in self.objs, "Required col 'hand' not in `self.card_`!"
        remaining = self.objs[self.objs.hand.isna()].copy()
        return remaining

    def _hands_to_assign(self):
        """Return a list of hands available for assignment (< 13 cards assigned)."""
        hand_size = self.objs.hand.value_counts()
        return hand_size[hand_size < 13].index.tolist()

    def _find_closest_obj(self, remaining: pd.DataFrame) -> Tuple[int, str]:
        """Find the closest obj to any of the *qualified hands*.

        Qualification: each hand can have 13 objects at most."""
        min_distance = 2  # large enough assuming coordinates are in (0, 1)
        closest_obj_idx = None
        closest_hand = None
        for hand in self._hands_to_assign():
            _hand_cards = self.objs[self.objs.hand == hand]
            hand_coords = list(_hand_cards[["center_x", "center_y"]].itertuples(index=False))

            for obj_idx, x, y in remaining[["center_x", "center_y"]].itertuples():
                distance = self.linkage.calc_distance(x, y, hand_coords)
                if distance < min_distance:
                    min_distance = distance
                    closest_obj_idx = obj_idx
                    closest_hand = hand

        closest_obj = remaining.loc[closest_obj_idx, ["name", "quadrant"]].to_dict()
        log.debug(
            "Found a closest obj(%s) to '%s': %s",
            closest_obj, closest_hand, min_distance)
        if min_distance > self.MAX_ASSIGNMENT_DISTANCE:
            raise ValueError("Can't assign cards to hands; "
                             "try separating hands a bit further")

        return closest_obj_idx, closest_hand

    def _assign_one_obj(self, obj_idx, hand):
        """Assign object to `hand`, by updating col 'hand'."""
        assert obj_idx in self.objs.index, f"{obj_idx} not found in `card_.index`!"
        self.objs.at[obj_idx, 'hand'] = hand

    def _drop_assigned(self, obj_idx, remaining: pd.DataFrame) -> pd.DataFrame:
        """Drop assigned object(s) from `remaining`."""
        _assigned_name = remaining.at[obj_idx, 'name']
        assigned_cards = remaining[remaining.name == _assigned_name]

        log.debug(
            "Dropping %s assigned cards: %s",
            len(assigned_cards),
            assigned_cards[["name", "quadrant"]].to_dict("records"))
        return remaining.drop(index=assigned_cards.index)

    def list_assigned_cards(self):
        return self.objs.to_dict('records')

    ##
    @staticmethod
    def _mark_marginal(card: pd.DataFrame, width) -> pd.DataFrame:
        """Mark a card as marginal based on (x, y).

        OK to ignore the top-left positioned origin, due to symmetricity.
        """
        bottom_l, top_r = (0, 0), (1, 1)
        bottom_r, top_l = (1, 0), (0, 1)

        def _calc_dist_to_border(row: pd.Series):
            p = (row.center_x, row.center_y)
            dist1 = util.point_line_dist(p, bottom_l, top_r)
            dist2 = util.point_line_dist(p, bottom_r, top_l)
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
        quadrant_card = self.objs[lambda df: df.quadrant == quadrant]

        _obj_coords = quadrant_card[["center_x", "center_y"]].itertuples(index=False)
        core_bool_seq = self.core_finder.find_core(_obj_coords)

        # Unmark core if dup exists outside its quadrant
        outside_card = self.objs[lambda df: df.quadrant != quadrant]
        no_dup_bool_seq = ~quadrant_card.name.isin(outside_card.name.values)

        return pd.Series(core_bool_seq & no_dup_bool_seq, index=quadrant_card.index)


def get_deal_converter(reader=Yolo4Reader()) -> DealConverter:
    assigner = _get_assigner()
    deal_converter = DealConverter(reader, assigner)
    return deal_converter


def _get_assigner() -> IAssigner:
    dbscan = strategy.CoreFinderDbscanPy()
    single_linkage = strategy.SingleLinkage()
    assigner = Assigner(core_finder=dbscan, linkage=single_linkage)
    return assigner
