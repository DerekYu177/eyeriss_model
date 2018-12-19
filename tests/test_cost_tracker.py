import unittest
import numpy as np

from ..model.processing_element import ProcessingElement as PE
class TestCostTracker(unittest.TestCase):
    def test_small_ifmap_memory(self):
        kernel = np.array([1, 2])
        small_ifmap = np.array(list(range(5)))

        pe = PE((0, 0))

        pe.set_kernel(kernel, mem='dram')
        pe.set_ifmap(small_ifmap, mem='dram')

        pe.conv()

        # one for setting the kernel, ifmap
        assert pe.cost_tracker.DRAM_writes == 2
        assert pe.cost_tracker.DRAM_reads == 0

        assert pe.cost_tracker.IPE_writes == 0
        assert pe.cost_tracker.IPE_reads == 0

        assert pe.cost_tracker.GLB_writes == 0
        assert pe.cost_tracker.GLB_reads == 0

        assert pe.cost_tracker.SPAD_writes == pe.ofmap
        # four "multiply operations" with 3 reads each: kernel, ifmap, psum
        assert pe.cost_tracker.SPAD_reads == pe.ofmap * 3

    def test_small_ifmap_ops(self):
        kernel = np.array([1, 2])
        small_ifmap = np.array(list(range(5)))

        pe = PE((0, 0))

        pe.set_kernel(kernel, mem='dram')
        pe.set_ifmap(small_ifmap, mem='dram')

        pe.conv()

        assert pe.cost_tracker.add == 4

        # mults are tracked individually
        # [1, 2] x [1, 2] is 2 separate mults, despite numpy doing it in one
        assert pe.cost_tracker.mult == 8

    def test_larger_ifmap_memory(self):
        kernel_slice = np.array([1, 2])
        ifmap_slice = np.array([i for i in range(28)])

        pe = PE((0, 0), stride=(1, 1))

        pe.set_kernel(kernel_slice, mem='dram')
        pe.set_ifmap(ifmap_slice, mem='dram')

        pe.conv()

        assert pe.cost_tracker.DRAM_writes == 2
        assert pe.cost_tracker.DRAM_reads == 0

        assert pe.cost_tracker.IPE_writes == 0
        assert pe.cost_tracker.IPE_reads == 0

        assert pe.cost_tracker.GLB_writes == 0
        assert pe.cost_tracker.GLB_reads == 0

        assert pe.cost_tracker.SPAD_writes == pe.ofmap
        assert pe.cost_tracker.SPAD_reads == pe.ofmap * 3

    def test_larger_ifmap_ops(self):
        kernel_slice = np.array([1, 2])
        ifmap_slice = np.array([i for i in range(28)])

        pe = PE((0, 0), stride=(1, 1))

        pe.set_kernel(kernel_slice, mem='dram')
        pe.set_ifmap(ifmap_slice, mem='dram')

        pe.conv()

        assert pe.cost_tracker.add == 27
        assert pe.cost_tracker.mult == 54
