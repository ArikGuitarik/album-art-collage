import pytest
from grid import SquareGrid

dummy_objects = [{i} for i in range(9)]


def test_grid_creation():
    grid = SquareGrid(dummy_objects)
    assert grid.shape == (3, 3)
    assert grid.to_list() == dummy_objects


def test_reject_non_square_number_of_elements():
    with pytest.raises(ValueError):
        SquareGrid(dummy_objects + [{10}])


def test_getter():
    grid = SquareGrid(dummy_objects)
    assert grid.get(0, 0) == {0}
    assert grid.get(2, 2) == {8}
    assert grid.get(1, 2) == {5}
    assert grid.get(2, 0) == {6}
    with pytest.raises(IndexError):
        grid.get(3, 1)
        grid.get(0, 3)


def test_setter():
    grid = SquareGrid(dummy_objects)
    element_to_add = {42}
    grid.set(2, 1, element_to_add)
    assert grid.get(2, 1) == element_to_add
    with pytest.raises(IndexError):
        grid.set(3, 0, element_to_add)


def test_swapping():
    grid = SquareGrid(dummy_objects)
    grid.swap(0, 0, 0, 0)
    assert grid.to_list() == dummy_objects
    grid.swap(1, 2, 0, 1)
    assert grid.get(0, 1) == {5}
    assert grid.get(1, 2) == {1}
    with pytest.raises(IndexError):
        grid.swap(3, 0, 1, 3)
