import unittest
import numpy as np

from ..model.ifmap_pipes import Pipe
class TestPipe(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipe((0, 0), 28)

    def test_pad_with_zeros(self):
        assert self.pipe.empty

        self.pipe.pad_with_zeros(1)

        assert len(self.pipe) == 1

        assert np.array_equal(
                self.pipe.ifmaps[0],
                np.array([0 for i in range(28)]))

        assert self.pipe.indices[0] is None

    def test_append(self):
        arr = np.array(list(range(28)))
        self.pipe.append(arr)

        assert np.array_equal(
                self.pipe.ifmaps[0],
                arr)

        assert self.pipe.indices[0] == 0

    def test_pop(self):
        arr = np.array(list(range(28, 28*2)))
        self.pipe.append(arr)

        ifmap, index = self.pipe.pop()

        assert np.array_equal(ifmap, arr)
        assert index == 1

from ..model.ifmap_pipes import PipeCoordinator
class TestPipeCoordinator(unittest.TestCase):
    class tPE:
        def __init__(self, i, j):
            self.uuid = (i, j)
            self.ifmap = None
            self.ifmap_index = None

        def set_ifmap(self, ifmap, ifmap_index=0, **kwargs):
            self.ifmap = ifmap
            self.ifmap_index = ifmap_index

        def __repr__(self):
            return "{}".format(self.uuid)

    def setUp(self):
        self.mock_pes = np.array([[self.tPE for j in range(2)] for i in range(2)])
        self.mock_pes[0][0] = self.mock_pes[0][0](0, 0)
        self.mock_pes[0][1] = self.mock_pes[0][1](0, 1)
        self.mock_pes[1][0] = self.mock_pes[1][0](1, 0)
        self.mock_pes[1][1] = self.mock_pes[1][1](1, 1)

        self.pipe_coordinator = PipeCoordinator(self.mock_pes, 28)

        self.ifmaps = np.array([
            list(range(28)),
            list(range(28, 28*2)),
            list(range(28*2, 28*3)),
            list(range(28*3, 28*4)),
            list(range(28*4, 28*5)),
        ])

    def test_ifmap_pes_property(self):
        len(self.pipe_coordinator.ifmap_pes) == 3

    def test_attach_pipes_to_pes(self):
        for pe, none in self.pipe_coordinator.ifmap_pes.items():
            assert type(pe) is self.tPE
            assert none is None

        self.pipe_coordinator._attach_pipes_to_pes()

        for pe, pipe in self.pipe_coordinator.ifmap_pes.items():
            assert type(pe) is self.tPE
            assert type(pipe) is Pipe

    def test_pad_pes_with_zeros(self):
        self.pipe_coordinator._attach_pipes_to_pes()
        self.pipe_coordinator._pad_pes_with_zeros()

        assert len(self.pipe_coordinator.ifmap_pes[self.mock_pes[0][0]]) == 0
        assert len(self.pipe_coordinator.ifmap_pes[self.mock_pes[0][1]]) == 1
        assert len(self.pipe_coordinator.ifmap_pes[self.mock_pes[1][0]]) == 0

    def test_update_pes(self):
        self.pipe_coordinator = PipeCoordinator(self.mock_pes, 3)

        self.pipe_coordinator._attach_pipes_to_pes()
        self.pipe_coordinator._pad_pes_with_zeros()

        ifmap_pes = self.pipe_coordinator.ifmap_pes

        ifmap_pes[self.mock_pes[1][0]].append(np.array([0, 1, 2]))
        ifmap_pes[self.mock_pes[0][0]].append(np.array([3, 4, 5]))

        assert len(ifmap_pes[self.mock_pes[0][0]].indices) == 1
        assert len(ifmap_pes[self.mock_pes[0][1]].indices) == 1
        assert len(ifmap_pes[self.mock_pes[1][0]].indices) == 1

        self.pipe_coordinator.update_pes()

        assert np.array_equal(
            ifmap_pes[self.mock_pes[0][0]].ifmaps[0],
            np.array([3, 4, 5]))

        assert np.array_equal(
            ifmap_pes[self.mock_pes[0][1]].ifmaps[0],
            np.array([0, 0, 0]))

        assert np.array_equal(
            ifmap_pes[self.mock_pes[1][0]].ifmaps[0],
            np.array([0, 1, 2]))

    def test_fill_pes(self):
        self.pipe_coordinator._attach_pipes_to_pes()
        self.pipe_coordinator._pad_pes_with_zeros()
        self.pipe_coordinator._fill_pes(self.ifmaps)

        ifmap_pes = self.pipe_coordinator.ifmap_pes

        assert np.array_equal(
            ifmap_pes[self.mock_pes[1][0]].indices,
            np.array(list(range(0, 5, 2))))

        assert np.array_equal(
            ifmap_pes[self.mock_pes[0][0]].indices,
            np.array(list(range(1, 5, 2))))

        assert np.array_equal(
            ifmap_pes[self.mock_pes[0][1]].indices,
            np.array([None, *list(range(2, 5, 2))]))
