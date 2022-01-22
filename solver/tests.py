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
    def converter(self):
        return converter.DealConverter()

    def test_read_yolo(self, converter):
        converter.read_yolo(self.YOLO_FILEPATH)

        assert isinstance(converter.card, pd.DataFrame)
        assert converter.card.shape == (51, 7)
        assert 'name' in converter.card.columns
        assert 'relative_coordinates.center_x' in converter.card.columns

    def test_simple_dedup(self, converter):
        converter.read_yolo(self.YOLO_FILEPATH)
        converter.dedup()

        assert isinstance(converter.card_, pd.DataFrame)
        assert converter.card_.shape == (41, 7)
        assert converter.card_.name.value_counts().max() == 1
        assert converter.card_.query('name == "5c"').confidence.iloc[0] == 0.999394
