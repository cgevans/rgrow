from ..anneal import AnnealOutputs
from abc import ABCMeta, abstractmethod


class ReportingMethod(metaclass=ABCMeta):
    @property
    @abstractmethod
    def desc(self) -> str: ...

    @abstractmethod
    def reporter_method(self, anneal_outp: AnnealOutputs): ...
