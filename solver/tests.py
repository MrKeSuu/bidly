import pathlib

import pandas as pd
import pytest

from pythondds_min import calc_ddtable_pbn
import converter
import main


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
    YOLO_FILEPATH = pathlib.Path('fixtures/deal1-result-md.json')

    @pytest.fixture
    def core_finder(self):
        return converter.CoreFinderDbscan()

    @pytest.fixture
    def deal_converter(self, core_finder: converter.ICoreFinder):
        deal_converter = converter.DealConverter(core_finder)
        deal_converter.read_yolo(self.YOLO_FILEPATH)
        return deal_converter

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

        assert "is_core" in deal_converter.card_.columns
        assert deal_converter.card_["is_core"].notna().all()

        expected_north_core_cards = {'3d', '7c', '5s', '6c', '6h'}
        actual_mark = deal_converter.card_.loc[
            lambda df: df.name.isin(expected_north_core_cards),
            'is_core']
        assert actual_mark.to_list() == [True] * 5
