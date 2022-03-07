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
    def deal_converter(self):
        deal_converter = converter.DealConverter()
        deal_converter.read_yolo(self.YOLO_FILEPATH)
        return deal_converter

    def test_read_yolo(self, deal_converter):
        assert isinstance(deal_converter.card, pd.DataFrame)
        assert deal_converter.card.shape == (51, 7)
        assert 'name' in deal_converter.card.columns
        assert 'relative_coordinates.center_x' in deal_converter.card.columns

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
