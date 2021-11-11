#! /usr/bin/python
import ctypes

import dds
import functions


def main():
    tableDealPBN = dds.ddTableDealPBN()
    table = dds.ddTableResults()
    myTable = ctypes.pointer(table)

    line = ctypes.create_string_buffer(80)

    dds.SetMaxThreads(0)

    HAND_1 = b"E:7.AQ8.J7654.JT94 A65.T642.AKQT.A7 KQJT82.KJ.82.Q83 943.9753.93.K652"
    HAND_2 = b"W:KT5.A876.J54.984 Q962.JT952.K987. AJ743.K.QT3.6532 8.Q43.A62.AKQJT7"
    HAND_3 = b"W:QT3.863.63.T9543 9865.KQ4.K.KQ872 42.J92.JT87542.6 AKJ7.AT75.AQ9.AJ"
    HANDS = [HAND_1, HAND_2, HAND_3]

    for handno in range(3):
        tableDealPBN.cards = HANDS[handno]

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
