import abc
import dataclasses


from . import detect


@dataclasses.dataclass
class Solution():
    pass  # TODO


class IPresenter(abc.abstractmethod):

    @abc.abstractmethod
    def present(self, solution: Solution):
        pass


class BridgeSolver(abc.ABC):
    presenter: IPresenter

    def __init__(self, presenter) -> None:
        self.presenter = presenter

    @abc.abstractmethod
    def validate_transform(self, cards: detect.CardDetection):
        pass

    @abc.abstractmethod
    def assign(self, transformed_cards):
        pass

    @abc.abstractmethod
    def solve(self, assigned_cards):
        pass

    def present(self, solution: Solution):
        self.presenter.present(solution)
