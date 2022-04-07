from random import Random
from typing import Any, Tuple


class BenfordRandom(Random):
    def __init__(
        self,
        seed=None,
        adjustments=5,
    ):
        super().__init__(x=seed)
        self.adjustments = adjustments

    def random(self) -> float:
        result = 1

        for i in range(self.adjustments):
            result *= super().random()

        return result

    def seed(
        self,
        a: Any = ...,
        version: int = ...,
    ) -> None:
        super().seed(a, version)

    def getstate(self) -> Tuple[Any, ...]:
        return super().getstate()

    def setstate(
        self,
        state: Tuple[Any, ...],
    ) -> None:
        super().setstate(state)
