from typing import TypeVar, Generic
from math import floor, sqrt

T = TypeVar("T")


class SquareGrid(Generic[T]):
    def __init__(self, elements: list[T]):
        self._elements = elements
        self._side_length = self._largest_square_number_below(len(elements))
        if self._side_length ** 2 != len(elements):
            raise ValueError("Number of elements has to be a square number.")

    @staticmethod
    def _largest_square_number_below(number) -> int:
        return int(floor(sqrt(number)))

    def _coordinate_to_index(self, row: int, col: int) -> int:
        return row * self._side_length + col

    def _index_to_coordinate(self, index: int) -> tuple[int, int]:
        return index // self._side_length, index % self._side_length

    def get_all_coordinates(self) -> list[tuple[int, int]]:
        return [self._index_to_coordinate(index) for index in range(len(self._elements))]

    def get(self, row: int, col: int) -> T:
        return self._elements[self._coordinate_to_index(row, col)]

    def set(self, row: int, col: int, element: T):
        self._elements[self._coordinate_to_index(row, col)] = element

    def swap(self, row1: int, col1: int, row2: int, col2: int):
        idx1, idx2 = self._coordinate_to_index(row1, col1), self._coordinate_to_index(row2, col2)
        self._elements[idx1], self._elements[idx2] = self._elements[idx2], self._elements[idx1]

    @property
    def shape(self) -> tuple[int, int]:
        return self._side_length, self._side_length

    def to_list(self) -> list[T]:
        return self._elements
