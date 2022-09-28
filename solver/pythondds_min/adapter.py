"""Helpers as extensions to pythondds_min/functions.py"""
import contextlib
import ctypes
import io
import logging as log

from pythondds_min import dds
from pythondds_min import functions

PbnHand = bytes


def solve_hand(hand: PbnHand):
    deal = _init_deal(hand)
    result = _init_result()

    # call CalcDDtablePBN
    ret_code = dds.CalcDDtablePBN(deal, result)

    if ret_code != dds.RETURN_NO_FAULT:
        msg = ctypes.create_string_buffer(80)
        dds.ErrorMessage(ret_code, msg)
        log.error("DDS error: %s", msg)

    return result


def format_hand(hand: PbnHand, title="Unnamed Hand"):
    title_line = f"Hand: {title}"
    deal = _init_deal(hand)

    # Capture stdout
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        functions.PrintPBNHand(title_line, deal.cards)
        formatted_hand = buf.getvalue()

    return formatted_hand


def format_table(table):
    # TODO
    # takes pointer from solve_hand
    # https://stackoverflow.com/questions/1218933/can-i-redirect-the-stdout-into-some-sort-of-string-buffer
    pass


def _init_deal(hand: PbnHand):
    deal_obj = dds.ddTableDealPBN()
    deal_obj.cards = hand
    return deal_obj


def _init_result():
    res = dds.ddTableResults()
    results = ctypes.pointer(res)
    return results
