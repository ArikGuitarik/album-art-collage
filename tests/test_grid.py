import pytest
from grid import SquareGrid


class TestSquareGrid:
    @pytest.fixture
    def dummy_objects(self):
        return [{i} for i in range(9)]

    def test_grid_creation(self, dummy_objects):
        grid = SquareGrid(dummy_objects)
        assert grid.shape == (3, 3)
        assert grid.to_list() == dummy_objects

    def test_reject_non_square_number_of_elements(self, dummy_objects):
        with pytest.raises(ValueError):
            SquareGrid(dummy_objects + [{10}])

    def test_getter(self, dummy_objects):
        grid = SquareGrid(dummy_objects)
        assert grid.get(0, 0) == {0}
        assert grid.get(2, 2) == {8}
        assert grid.get(1, 2) == {5}
        assert grid.get(2, 0) == {6}
        with pytest.raises(IndexError):
            grid.get(3, 1)
        with pytest.raises(IndexError):
            grid.get(0, 3)

    def test_setter(self, dummy_objects):
        grid = SquareGrid(dummy_objects)
        element_to_add = {42}
        grid.set(2, 1, element_to_add)
        assert grid.get(2, 1) == element_to_add
        with pytest.raises(IndexError):
            grid.set(3, 0, element_to_add)

    def test_swapping(self, dummy_objects):
        grid = SquareGrid(dummy_objects)
        grid.swap(0, 0, 0, 0)
        assert grid.to_list() == dummy_objects
        grid.swap(1, 2, 0, 1)
        assert grid.get(0, 1) == {5}
        assert grid.get(1, 2) == {1}
        with pytest.raises(IndexError):
            grid.swap(3, 0, 1, 3)

    def test_iteration(self, dummy_objects):
        grid = SquareGrid(dummy_objects)
        elements = []
        for row, col in grid.get_all_coordinates():
            elements.append(grid.get(row, col))
        assert elements == dummy_objects

    def test_truncation(self, dummy_objects):
        extended_dummy_objects = dummy_objects + [{10}, {11}]
        truncated_grid = SquareGrid.create_by_truncating_excess_elements(extended_dummy_objects)
        assert truncated_grid.to_list() == dummy_objects
        truncated_but_unchanged_grid = SquareGrid.create_by_truncating_excess_elements(dummy_objects)
        assert truncated_but_unchanged_grid.to_list() == dummy_objects
