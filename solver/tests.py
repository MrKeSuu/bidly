import pathlib

import pandas as pd
import pytest

from pythondds_min import adapter
from pythondds_min import calc_ddtable_pbn
import converter
import main
import strategy


@pytest.fixture(scope='module')
def pbn_hand() -> bytes:
    return b'W:9432.AT72.K98.JT KQ65.KJ.A52.9632 7.Q86.QJ763.AK84 AJT8.9543.T4.Q75'


@pytest.mark.verbose
class TestBasic:
    def test_min_sample_run(self, capsys):
        calc_ddtable_pbn.main()

        captured = capsys.readouterr()
        assert "KQJT82                  7" in captured.out
        assert "KJ                      AQ8" in captured.out
        assert "NT     7     7     6     6\n" in captured.out
        assert " C     9     9     3     4\n" in captured.out
        assert " H    13    13     0     0\n" in captured.out

    def test_main(self):
        main.main()


class TestConverter:
    DEAL1_YOLO_FILEPATH = pathlib.Path('fixtures/deal1-result-md.json')
    MANUAL_EDIT_YOLO_FILEPATH = pathlib.Path('fixtures/deal3-manual-edit.json')

    PBN_FILEPATH = pathlib.Path('fixtures/deal3-manual-edit.pbn')

    @pytest.fixture
    def core_finder(self):
        return strategy.CoreFinderDbscan()

    @pytest.fixture
    def linkage(self):
        return strategy.SingleLinkage()

    @pytest.fixture
    def deal_converter(self, core_finder: strategy.ICoreFinder, linkage: strategy.ILinkage):
        deal_converter = converter.DealConverter(core_finder, linkage)
        deal_converter.read_yolo(self.DEAL1_YOLO_FILEPATH)
        return deal_converter

    @pytest.fixture
    def card_names(self):
        return pd.Series(
            ['8s', 'Js', 'As', '5c', '10s', '3h', '4h', '7c', 'Qc', '10c', '4s', '5h', '9h'],
            name="name")

    # TODO to speed up tests, run complete deal_converter methods and test each step

    def test_read_yolo(self, deal_converter):
        assert isinstance(deal_converter.card, pd.DataFrame)
        assert deal_converter.card.shape == (51, 7)
        assert set(deal_converter.card.columns) == {
            'center_x', 'center_y',
            'height', 'width',
            'class_id', 'name',
            'confidence',
        }

    def test_simple_dedup(self, deal_converter):
        deal_converter.dedup()

        assert isinstance(deal_converter.card_, pd.DataFrame)
        assert deal_converter.card_.shape == (41, 7)
        assert deal_converter.card_.name.value_counts().max() == 1
        assert deal_converter.card_.query('name == "5c"').confidence.iloc[0] == 0.999394

    def test_smart_dedup(self, deal_converter):
        deal_converter.dedup(smart=True)

        assert isinstance(deal_converter.card_, pd.DataFrame)
        assert deal_converter.card_.shape == (48, 7)
        assert deal_converter.card_.name.value_counts().max() == 2
        assert deal_converter.card_.query('name == "5h"').shape[0] == 1  # bad dup dropped
        assert deal_converter.card_.query('name == "5c"').shape[0] == 2  # good dup appended back

    def test_report_missing_and_fp(self, deal_converter: converter.DealConverter, capsys):
        deal_converter.dedup()
        deal_converter.report_missing_and_fp()

        captrued = capsys.readouterr()
        assert captrued.out == (
            "Missing cards: ['2s', '6s', '9s', '3c', '10c', '4d', '5d', '10d', 'Qd', '3h', '9h']\n"
            "FP cards: ['5c']\n")

    def test_assign(self, deal_converter: converter.DealConverter):
        deal_converter.read_yolo(self.MANUAL_EDIT_YOLO_FILEPATH)
        deal_converter.dedup(smart=True)
        deal_converter.assign()

        card_ = deal_converter.card_.dropna(subset=["hand"]).set_index("name")
        assert isinstance(card_, pd.DataFrame)
        assert card_.index.is_unique
        assert card_.hand.value_counts().eq(13).all()
        assert card_.at['7s', 'hand'] == 'east'
        assert card_.at['Ah', 'hand'] == 'west'
        assert card_.at['6c', 'hand'] == 'north'
        assert card_.at['10d', 'hand'] == 'south'

    def test_divide_to_quadrants_basic(self, deal_converter: converter.DealConverter):
        deal_converter.dedup(smart=True)
        deal_converter._divide_to_quadrants()

        expected_north_cards = {'3d', '7c', '5s', '6c', '6h', 'Jh'}
        actual_quadrants = deal_converter.card_.loc[
            lambda df: df.name.isin(expected_north_cards),
            'quadrant']
        assert actual_quadrants.to_list() == ['top'] * 6

        expected_marginal_cards = {"8c", "Jc"}
        actual_quadrants = deal_converter.card_.loc[
            lambda df: df.name.isin(expected_marginal_cards),
            'quadrant']
        assert actual_quadrants.to_list() == ['margin'] * 2

    def test_mark_core_objs(self, deal_converter: converter.DealConverter):
        deal_converter.dedup(smart=True)
        deal_converter._divide_to_quadrants()
        deal_converter._mark_core_objs()

        card = deal_converter.card_
        assert "is_core" in card.columns
        assert card.is_core.notna().all()

        assert card.query("name == 'As' and quadrant == 'left'").is_core.to_list() == [True]
        assert card.query("name == '8h' and quadrant == 'right'").is_core.to_list() == [False]

        expected_north_core_cards = {'3d', '7c', '5s', '6c', '6h'}
        actual_mark = card.loc[card.name.isin(expected_north_core_cards), 'is_core']
        assert actual_mark.to_list() == [True] * 5

    def test_drop_core_duplicates(self, deal_converter: converter.DealConverter):
        deal_converter.dedup(smart=True)
        deal_converter._divide_to_quadrants()
        deal_converter._mark_core_objs()
        deal_converter._drop_core_duplicates()

        card = deal_converter.card_
        assert card.shape == (44, 10)
        assert card.query("name == '6d' and is_core == True").shape[0] == 1
        assert card.query("name == 'Qc'").shape[0] == 1
        assert card.query("name == 'Js' and quadrant == 'margin'").empty

    def test_assign_core_objs(self, deal_converter: converter.DealConverter):
        deal_converter.dedup(smart=True)
        deal_converter._divide_to_quadrants()
        deal_converter._mark_core_objs()
        deal_converter._drop_core_duplicates()
        deal_converter._assign_core_objs()

        card = deal_converter.card_
        assert "hand" in card
        assert card.query("name == 'As'").hand.to_list() == ["west"]
        assert card.query("name == '6h'").hand.to_list() == ["north"]
        assert card.query("quadrant == 'margin'").hand.isna().all()

    def test_find_closest_obj(self, deal_converter: converter.DealConverter):
        deal_converter.dedup(smart=True)
        deal_converter._divide_to_quadrants()
        deal_converter._mark_core_objs()
        deal_converter._drop_core_duplicates()
        deal_converter._assign_core_objs()

        remaining = deal_converter._list_remaining_objs()
        obj_idx, hand = deal_converter._find_closest_obj(remaining)

        assert obj_idx == 46
        assert hand == 'south'

    def test_build_pbn_suit(self, deal_converter: converter.DealConverter, card_names):
        formatted_suit = deal_converter._build_pbn_suit(card_names, "s")
        assert formatted_suit == "AJT84"

        formatted_suit = deal_converter._build_pbn_suit(card_names, "d")
        assert formatted_suit == ""

    def test_build_pbn_deal(self, deal_converter: converter.DealConverter):
        deal_converter.read_yolo(self.MANUAL_EDIT_YOLO_FILEPATH)
        deal_converter.dedup(smart=True)
        deal_converter.assign()

        hands = deal_converter._build_pbn_hands()
        formatted_deal = deal_converter._build_pbn_deal(hands)

        expected = 'W:9432.AT72.K98.JT KQ65.KJ.A52.9632 7.Q86.QJ763.AK84 AJT8.9543.T4.Q75'
        assert formatted_deal == expected

    def test_write_pbn(self, deal_converter: converter.DealConverter, pbn_hand: bytes):
        deal_converter.read_yolo(self.MANUAL_EDIT_YOLO_FILEPATH)
        deal_converter.dedup(smart=True)
        deal_converter.assign()

        assert not self.PBN_FILEPATH.exists()

        deal_converter.write_pbn(self.PBN_FILEPATH)

        assert self.PBN_FILEPATH.exists()

        with open(self.PBN_FILEPATH, 'rb') as fi:
            content = fi.read()
        assert content == pbn_hand

        self.PBN_FILEPATH.unlink()


@pytest.mark.single
class TestDdsAdapter:

    @pytest.fixture(scope='class')
    def result(self, pbn_hand):
        return adapter.solve_hand(pbn_hand)

    def test_format_hand(self, pbn_hand):
        formatted = adapter.format_hand(pbn_hand)

        assert "Hand: Unnamed Hand\n" in formatted
        assert "\nK98                     QJ763 " in formatted

    def test_solve_hand(self, result):
        formatted = adapter.format_result(result)

        assert "NT" in formatted
        assert "South" in formatted

    def test_result_to_df(self, result):
        result_df = adapter.result_to_df(result)

        assert isinstance(result_df, pd.DataFrame)
        assert result_df.shape == (20, 1)
        assert result_df.columns.tolist() == ['tricks']
        assert result_df.at[('N', 'H'), 'tricks'] == 5
        assert result_df.at[('S', 'C'), 'tricks'] == 7
        assert result_df.at[('E', 'N'), 'tricks'] == 7
        assert result_df.at[('W', 'D'), 'tricks'] == 9

    def test_calc_par_none_vul(self, result):
        par_result = adapter.calc_par(result, vul=0)
        formatted = adapter.format_par(par_result)

        assert "NS score: NS -100\n" in formatted
        assert "NS list : NS:NS 3Sx" in formatted

    def test_calc_par_both_vul(self, result):
        par_result = adapter.calc_par(result, vul=1)
        formatted = adapter.format_par(par_result)

        assert "NS score: NS -110\n" in formatted
        assert "NS list : NS:EW 3D" in formatted

    def test_calc_par_ew_vul(self, result):
        par_result = adapter.calc_par(result, vul=3)
        formatted = adapter.format_par(par_result)

        assert "NS score: NS -100\n" in formatted
        assert "NS list : NS:NS 3Sx" in formatted
