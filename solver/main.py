import argparse

import pythondds_min.adapter as dds_adapter
import converter


def main():
    parser = argparse.ArgumentParser("bidly solver")
    parser.add_argument("yolo_path", help="Path to yolo output json file")
    args = parser.parse_args()

    deal_converter = converter.get_deal_converter()
    deal_converter.read_yolo(args.yolo_path)
    deal_converter.dedup(smart=True)
    deal_converter.assign()
    pbn_hand = deal_converter.format_pbn()

    formatted_hand = dds_adapter.format_hand(pbn_hand, title="Example Hand")
    print(formatted_hand)

    dds_result = dds_adapter.solve_hand(pbn_hand)
    formatted_dd_result = dds_adapter.format_result(dds_result)
    print(formatted_dd_result)

    print(dds_adapter.result_to_df(dds_result).head(5))


if __name__ == '__main__':
    main()
