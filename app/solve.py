import abc
import dataclasses
import logging

import detect
from solver import converter
import solver.pythondds_min.adapter as dds_adapter


lgr = logging


# Abstract #

@dataclasses.dataclass
class Solution():
    hand: object
    dds_result: object


class IPresenter(abc.ABC):

    @abc.abstractmethod
    def present(self, solution: Solution):
        pass


class BridgeSolverBase(abc.ABC):
    cards: detect.CardDetection
    presenter: IPresenter

    solution_: Solution

    def __init__(self, cards, presenter) -> None:
        self.cards = cards
        self.presenter = presenter

    @abc.abstractmethod
    def transform(self):
        pass

    @abc.abstractmethod
    def assign(self):
        pass

    @abc.abstractmethod
    def list_unsure(self):
        pass

    @abc.abstractmethod
    def solve(self):
        pass

    def present(self):
        return self.presenter.present(self.solution_)


# Impl. #

class BridgeSolver(BridgeSolverBase):
    def __init__(self, cards, presenter) -> None:
        super().__init__(cards, presenter)
        self.converter = converter.get_deal_converter(reader=converter.Yolo5Reader())

    def transform(self):
        self.converter.read(self.cards)
        self.converter.dedup(smart=True)

    def assign(self):
        self.converter.assign()

    def list_unsure(self):
        pass  # TODO

    def solve(self):
        pbn_hand = self.converter.format_pbn()

        dds_result = dds_adapter.solve_hand(pbn_hand)
        lgr.debug("DDS result: %s", dds_result)

        self.solution_ = Solution(
            hand=pbn_hand,
            dds_result=dds_result
        )


class StringPresenter(IPresenter):
    def present(self, solution: Solution):
        formatted_hand = dds_adapter.format_hand(solution.hand)

        formatted_dd_result = dds_adapter.format_result(solution.dds_result)
        return formatted_hand, formatted_dd_result


class PrintPresenter(IPresenter):  # TODO ideally have another separate `View` and this only transforms
    def present(self, solution: Solution):
        formatted_hand = dds_adapter.format_hand(solution.hand)
        print(formatted_hand)

        formatted_dd_result = dds_adapter.format_result(solution.dds_result)
        print(formatted_dd_result)

        par_result = dds_adapter.calc_par(solution.dds_result)
        print(dds_adapter.format_par(par_result))
