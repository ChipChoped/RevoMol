from typing import Callable, Any


class Distance:
    def __init__(self, name: str, function: Callable[[str, str], Any]) -> None:
        self._name = name
        self.function = function

    @property
    def name(self) -> str:
        """Return the name of the evaluation.

        Returns:
            str: The name of the evaluation.
        """
        return self._name

    def distance(self, molecule_1: str, molecule_2: str) -> float:
        distance = self.function(molecule_1, molecule_2)
        return distance
