"""Converting .json from yolo into .pbn for pythondds."""


class DealConverter:
    def __init__(self):
        pass

    def read_yolo(self, path):
        pass

    def dedup(self):
        pass

    # three cases after dedup
    def assign(self):
        """Case 1: everything is perfect -> work on assign cards to four hands"""
        pass

    def report_missing_and_fp(self):
        """Case 2: missing cards or FP cards -> report them"""
        pass

    def infer_missing(self):
        """Case 3: missing cards -> attempt to infer"""
        pass  # * lower priority

    def write_pbn(self, path):
        pass
