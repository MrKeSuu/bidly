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
