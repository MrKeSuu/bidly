import pytest

from pythondds_min import calc_ddtable_pbn


class TestBasic:
    def test_min_sample_run(self):
        calc_ddtable_pbn.main()


# TODO add tests verifying ddtable output
