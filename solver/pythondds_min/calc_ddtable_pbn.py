#! /usr/bin/python
import ctypes

import deals
from pythondds_min import dds
from pythondds_min import functions


HANDS = [
    deals.DEAL_1,
    deals.DEAL_2,
    deals.DEAL_3,
]


def main():
    tableDealPBN = dds.ddTableDealPBN()
    table = dds.ddTableResults()
    myTable = ctypes.pointer(table)

    line = ctypes.create_string_buffer(80)

    dds.SetMaxThreads(0)

    for handno, hand in enumerate(HANDS):
        tableDealPBN.cards = hand

        res = dds.CalcDDtablePBN(tableDealPBN, myTable)

        if res != dds.RETURN_NO_FAULT:
            dds.ErrorMessage(res, line)
            print("DDS error: {}".format(line.encode("utf-8")))

        match = functions.CompareTable(myTable, handno)

        line = "CalcDDtable, hand {}".format(handno + 1)

        functions.PrintPBNHand(line, tableDealPBN.cards)

        functions.PrintTable(myTable)


if __name__ == '__main__':
    main()
