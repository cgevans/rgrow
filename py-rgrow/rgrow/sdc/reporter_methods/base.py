from ..anneal import AnnealOutputs
from abc import ABCMeta, abstractmethod


class ReportingMethod(metaclass=ABCMeta):
    @abstractmethod
    @property
    def desc(self) -> str: ...

    @abstractmethod
    def reporter_method(self, anneal_outp: AnnealOutputs): ...
