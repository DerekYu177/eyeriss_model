import unittest
import numpy as np

from ..model.processing_element import ProcessingElement as pe
class TestPE(unittest.TestCase):
    def setUp(self):
        self.kernel = np.array([1, 2])
        self.small_ifmap = np.array(list(range(5)))

        self.pe = pe((0, 0))
        self.pe.set_kernel(self.kernel, mem='dram')
        self.pe.set_ifmap(self.small_ifmap, mem='dram')

    def test_ofmap(self):
        assert self.pe.ofmap == 4

    def test_ready_returns_true_if_ifmap_kernel_set(self):
       assert self.pe.ready

    def test_ready_returns_false_if_not_set(self):
        self.pe = pe((0, 0))
        assert not self.pe.ready

    def test_psum_does_vector_mac_during_conv(self):
        self.pe.conv()
        assert np.array_equal(
            self.pe.get_psum(mem="dram"),
            np.array([2, 5, 8, 11]))

class TestPEwithLargeIfmap(unittest.TestCase):
    def setUp(self):
        kernel_slice = np.array([1, 2])
        ifmap_slice = np.array([i for i in range(28)])

        self.pe = pe((0, 0), stride=(1, 1))
        self.pe.set_kernel(kernel_slice, mem='dram')
        self.pe.set_ifmap(ifmap_slice, mem='dram')

    def test_ofmap_large(self):
        assert self.pe.ofmap == 27

    def test_convolution_works_large(self):
        self.pe.conv()

        # feel free to test this yourself
        assert np.array_equal(
            self.pe.get_psum(mem='dram'),
            np.array([i for i in range(2, 83, 3)]))

class TestTwoPEsConnectedTogether(unittest.TestCase):
    def setUp(self):
        kernel_top = np.array([1, 2])
        kernel_bottom = np.array([3, 4])

        ifmap_top = np.array([i for i in range(28)])
        ifmap_bottom = np.array([i for i in range(28, 56)])

        self.pe_top = pe((0, 0), stride=(1, 1))
        self.pe_bottom = pe((0, 1), stride=(1, 1))

        self.pe_top.set_kernel(kernel_top, mem="dram")
        self.pe_top.set_ifmap(ifmap_top, mem="dram")

        self.pe_bottom.set_kernel(kernel_bottom, mem="dram")
        self.pe_bottom.set_ifmap(ifmap_bottom, mem="dram")

    def test_pe_top_works_as_before(self):
        self.pe_top.conv()
        assert np.array_equal(
            self.pe_top.get_psum(mem="dram"),
            np.array([i for i in range(2, 83, 3)]))

    def test_assert_psums_connected_correctly(self):
        self.pe_bottom.conv()
        self.pe_bottom.t_shift_psum_to(self.pe_top)
        self.pe_top.conv()

        assert np.array_equal(
            self.pe_bottom.get_psum(mem="dram"),
            np.array([i for i in range(200, 389, 7)]))

        assert np.array_equal(
            self.pe_top.get_psum(mem="dram"),
            np.array([i for i in range(202, 472, 10)]))
