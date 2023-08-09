import argparse

import solver.pythondds_min.adapter as dds_adapter
from solver import converter


def main():
    parser = argparse.ArgumentParser("bidly solver")
    parser.add_argument("yolo_path", help="Path to yolo output json file")
    args = parser.parse_args()

    deal_converter = converter.get_deal_converter()
    deal_converter.read(args.yolo_path)
    deal_converter.dedup(smart=True)

    transformed_cards = deal_converter.card_.to_dict("records")
    assigned_cards = deal_converter.assign(transformed_cards)
    pbn_hand = deal_converter.format_pbn(assigned_cards)

    formatted_hand = dds_adapter.format_hand(pbn_hand, title="Example Hand")
    print(formatted_hand)

    dds_result = dds_adapter.solve_hand(pbn_hand)
    formatted_dd_result = dds_adapter.format_result(dds_result)
    print(formatted_dd_result)

    print(dds_adapter.result_to_df(dds_result).head(5))


if __name__ == '__main__':
    main()
