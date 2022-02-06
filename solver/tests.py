import pathlib

import pandas as pd
import pytest

from pythondds_min import calc_ddtable_pbn
import converter
import main


class TestBasic:
    def test_min_sample_run(self):
        calc_ddtable_pbn.main()

    def test_main(self):
        main.main()


# TODO add tests verifying ddtable output


@pytest.mark.single
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
