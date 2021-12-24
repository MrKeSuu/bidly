"""
see: https://www.tistis.nl/pbn/

3.4.11  The Deal tag
--------------------
  The Deal tag value gives the cards of each hand.  The tag value is
defined as "<first>:<1st_hand> <2nd_hand> <3rd_hand> <4th_hand>".  The 4
hands are given in clockwise rotation.  A space character exists between
two consecutive hands.  The direction of the <1st_hand> is indicated by
<first>, being W (West), N (North), E (East), or S (South).  The cards of
each hand are given in the order:  spades, hearts, diamonds, clubs.  A dot
character "." exists between two consecutive suits of a hand.  The cards of
a suit are given by their ranks.  The ranks are defined as (in descending
order):
    A , K , Q , J , T , 9 , 8 , 7 , 6 , 5 , 4 , 3 , 2.
  Note that the 'ten' is defined as the single character "T". If a hand
contains a void in a certain suit, then no ranks are entered at the place
of that suit.
  Not all 4 hands need to be given. A hand whose cards are not given, is
indicated by "-" . For example, only the east/west hands are given:
[Deal "W:KQT2.AT.J6542.85 - A8654.KQ5.T.QJT6 -"]
  In import format, the ranks of a suit can be given in any order; the
value of <first> is free.  In export format, the ranks must be given in
descending order; <first> is equal to the dealer.
"""

# see other PBN hands: https://github.com/Afwas/python-dds/blob/master/examples/hands.py#L55
DEAL_1 = b"E:7.AQ8.J7654.JT94 A65.T642.AKQT.A7 KQJT82.KJ.82.Q83 943.9753.93.K652"
DEAL_2 = b"W:KT5.A876.J54.984 Q962.JT952.K987. AJ743.K.QT3.6532 8.Q43.A62.AKQJT7"
DEAL_3 = b"W:QT3.863.63.T9543 9865.KQ4.K.KQ872 42.J92.JT87542.6 AKJ7.AT75.AQ9.AJ"
