import pytest

from pythondds_min import calc_ddtable_pbn
import main


class TestBasic:
    def test_min_sample_run(self):
        calc_ddtable_pbn.main()

    def test_main(self):
        main.main()


# TODO add tests verifying ddtable output
