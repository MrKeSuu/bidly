"""Helpers as extensions to pythondds_min/functions.py"""
import contextlib
import ctypes
import io
import logging as log

import numpy as np
import pandas as pd

from pythondds_min import dds
from pythondds_min import functions
from pythondds_min import hands

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


def format_result(result):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        functions.PrintTable(result)
        formatted_result = buf.getvalue()

    return formatted_result


def to_result_df(result) -> pd.DataFrame:
    """Convert result to a DataFrame indexed by (player, suit).

    `resTable` doc: Encodes the solution of a deal for combinations of denomination and declarer.
    First index is denomination. Suit encoding. Second index is declarer. Hand encoding.
    Each entry is a number of tricks.
    """
    orig_table = result.contents.resTable
    orig_values = np.asarray([orig_table[s][p] for s in range(5) for p in range(4)]).reshape(5, 4)
    df_table = pd.DataFrame(
        orig_values,
        index=pd.Series(hands.dcardSuit, name='suit'),
        columns=pd.Series(hands.dcardHand, name='player'))

    result_df = (
        df_table.stack().rename('tricks')
            .reset_index().set_index(['player', 'suit']))
    return result_df


def tricks_to_level(tricks):
    assert 0 <= tricks <= 13
    return max(0, tricks - 6)


def _init_deal(hand: PbnHand):
    deal_obj = dds.ddTableDealPBN()
    deal_obj.cards = hand
    return deal_obj


def _init_result():
    res = dds.ddTableResults()
    results = ctypes.pointer(res)
    return results
