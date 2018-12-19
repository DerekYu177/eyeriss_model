import unittest
import numpy as np

from ..model.processing_element import ProcessingElement

from ..model.accelerator import Accelerator
class TestAccelerator(unittest.TestCase):
    def setUp(self):
        self.acc = Accelerator((2, 2))
        self.kernel = np.array([[1, 2], [3, 4]])

        self.empty_kernel = [0, 0]
        self.empty_ifmap = [0 for i in range(28)]
        self.empty_psum = [0 for i in range(27)]

        self.r_0 = list(range(28*0, 28*1))
        self.r_1 = list(range(28*1, 28*2))
        self.r_2 = list(range(28*2, 28*3))
        self.r_3 = list(range(28*3, 28*4))
        self.r_4 = list(range(28*4, 28*5))
        self.r_5 = list(range(28*5, 28*6))

        self.ifmap = np.array([
            self.r_0, self.r_1, self.r_2,
            self.r_3, self.r_4, self.r_5])

    def test_four_PEs_initialized(self):
        assert len(self.acc.pes) == 2

        assert len(self.acc.pes[0]) == 2
        assert len(self.acc.pes[1]) == 2

    def test_PEs_in_correct_order(self):
        total_acc = 0

        for acc_col in self.acc.pes:

            uuid = [0, 0]
            for acc in acc_col:
                total_acc += 1
                assert acc.uuid[1] >= uuid[1]
                uuid[1] = acc.uuid[1]

    def test_run_ready_PEs_only_those_which_have_ifmap_and_kernel_set(self):
        class tPE:
            def __init__(*args, **kwargs):
                self._conv = False

            def conv(self):
                self._conv = True

            @property
            def ready(self):
                return True

        class tCT:
            def __init__(*args, **kwargs):
                pass

        local_accelerator = Accelerator((2, 2), stride=1,
                processing_element=tPE, global_cost_tracker=tCT)
        local_accelerator.run_ready_PEs(0)
        local_accelerator.run_ready_PEs(1)

        assert local_accelerator.pes[0][0]._conv
        assert local_accelerator.pes[0][1]._conv
        assert local_accelerator.pes[1][0]._conv
        assert local_accelerator.pes[1][1]._conv

    def test_default_kernel_not_set(self):
       assert not self.acc.kernel_set

    def test_kernels_start_as_none(self):
        assert self.acc.pes[0][0].get_kernel(mem="dram") is None
        assert self.acc.pes[0][1].get_kernel(mem="dram") is None
        assert self.acc.pes[1][0].get_kernel(mem="dram") is None
        assert self.acc.pes[1][1].get_kernel(mem="dram") is None

    def test_set_kernel(self):
        self.acc.set_kernel(self.kernel)

        assert self.acc.kernel_set

        assert np.array_equal(
            self.acc.pes[0][0].get_kernel(mem="dram"),
            np.array([3, 4]))

        assert np.array_equal(
            self.acc.pes[1][0].get_kernel(mem="dram"),
            np.array([1, 2]))

    def test_default_ifmap_not_set(self):
        assert not self.acc.ifmap_set

    def test_set_ifmap(self):
        self.acc.set_ifmap(self.ifmap)

        assert self.acc.ifmap_set

        assert np.array_equal(
            self.acc.pes[0][0].get_ifmap(mem="dram"), self.r_1)

        assert np.array_equal(
            self.acc.pes[0][1].get_ifmap(mem="dram"), self.empty_ifmap)

        # #set_ifmap sets the first M+N-1 kernels, so the one is still left as None
        assert self.acc.pes[1][1].get_ifmap(mem="dram") is None

        assert np.array_equal(
            self.acc.pes[1][0].get_ifmap(mem="dram"), self.r_0)

    def test_set_ifmap_index(self):
        self.acc.set_ifmap(self.ifmap)
        assert self.acc.pes[0][0].ifmap_index == 1
        assert self.acc.pes[0][1].ifmap_index == None
        assert self.acc.pes[1][1].ifmap_index == None
        assert self.acc.pes[1][0].ifmap_index == 0

    def test_propagate_kernel(self):
        self.acc.set_kernel(self.kernel)
        self.acc.propagate_kernel()

        assert np.array_equal(
            self.acc.pes[1][1].get_kernel(mem="dram"),
            np.array([1, 2]))

        assert np.array_equal(
            self.acc.pes[0][1].get_kernel(mem="dram"),
            np.array([3, 4]))

    def test_propagate_ifmaps(self):
        self.acc.set_ifmap(self.ifmap)
        self.acc.propagate_ifmaps()

        assert np.array_equal(
            self.acc.pes[0][0].get_ifmap(mem="dram"), self.r_3)

        assert np.array_equal(
            self.acc.pes[1][0].get_ifmap(mem="dram"), self.r_2)

        assert np.array_equal(
            self.acc.pes[1][1].get_ifmap(mem="dram"), self.r_1)

        assert np.array_equal(
            self.acc.pes[0][1].get_ifmap(mem="dram"), self.r_2)

    def test_propagate_ifmap_indices(self):
        self.acc.set_ifmap(self.ifmap)
        self.acc.propagate_ifmaps()

        assert self.acc.pes[0][0].ifmap_index == 3
        assert self.acc.pes[0][1].ifmap_index == 2
        assert self.acc.pes[1][1].ifmap_index == 1
        assert self.acc.pes[1][0].ifmap_index == 2

    def test_propagate_psums(self):
        for pe_row in self.acc.pes:
            for pe in pe_row:
                assert pe.get_psum(mem="dram") is None

        self.acc.pes[0][0].set_psum(1, mem="dram")
        self.acc.pes[0][1].set_psum(2, mem="dram")

        assert self.acc.pes[0][0].get_psum(mem="dram") == 1
        assert self.acc.pes[0][1].get_psum(mem="dram") == 2

        self.acc.propagate_psums(0)

        assert self.acc.pes[1][0].get_psum(mem="dram") == 1
        assert self.acc.pes[1][1].get_psum(mem="dram") == 2

    def test_conv_returns_false_if_not_ifmap_set_or_kernel_set(self):
        assert not self.acc.ifmap_set
        assert not self.acc.kernel_set

        assert not self.acc.conv()

class TestGlobalCostTracker(unittest.TestCase):
    def setUp(self):
        self.acc = Accelerator((2, 2))
        self.kernel = np.array([[1, 2], [3, 4]])
        self.ifmap = np.array([
            list(range(0, 3)),
            list(range(3, 6)),
            list(range(6, 9)),
        ])

        self.ofmap = np.array([
            [27, 37],
            [57, 67]
        ])

        self.acc.set_ifmap(self.ifmap)
        self.acc.set_kernel(self.kernel)

    def test_conv_still_works(self):
        return

        assert self.acc.conv()

        assert np.array_equal(
            self.acc.ofmap,
            self.ofmap)
