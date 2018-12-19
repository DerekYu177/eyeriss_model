import unittest
import numpy as np

from .test_helper import create_array

from ..model.accelerator import Accelerator
class TestIntegrationConvolution(unittest.TestCase):
    def setUp(self):
        self.ifmap = create_array(0, 1, 28, (28, 28))

    def test_2x2_with_small_ifmap(self):
        acc = Accelerator((2, 2))

        kernel = np.array([[1, 2], [3, 4]])
        ifmap = np.array([
            list(range(0, 28)),
            list(range(28, 56))])

        acc.set_kernel(kernel)
        acc.set_ifmap(ifmap)

        assert acc.conv()

        answer = np.array([[i for i in range(202, 462+1, 10)]])

        assert np.array_equal(acc.ofmap, answer)

    def test_2x2_with_larger_ifmap(self):
        acc = Accelerator((2, 2))

        kernel = np.array([[1, 2], [3, 4]])
        ifmap = np.array([
            list(range(28*0, 28*1)),
            list(range(28*1, 28*2)),
            list(range(28*2, 28*3)),
            list(range(28*3, 28*4)),
            list(range(28*4, 28*5)),
            list(range(28*5, 28*6)),
        ])

        acc.set_ifmap(ifmap)
        acc.set_kernel(kernel)

        assert acc.conv()

        answer = np.array([
            list(range(202, 462+1, 10)),
            list(range(482, 742+1, 10)),
            list(range(762, 1022+1, 10)),
            list(range(1042, 1302+1, 10)),
            list(range(1322, 1582+1, 10)),
        ])

        assert np.array_equal(acc.ofmap, answer)

    def test_2x2_with_full_ifmap(self):
        acc = Accelerator((2, 2))

        kernel = np.array([[0, 1], [2, 3]])
        ifmap = self.ifmap

        acc.set_kernel(kernel)
        acc.set_ifmap(ifmap)

        assert acc.conv()

        # verify this yourself if you don't believe me
        # I did this in excel
        answer = create_array(144, 6, 168, (27, 27))

        assert np.array_equal(acc.ofmap, answer)

    def test_7x1(self):
        """
        This is a heavily contrived example to test psum
        """
        acc = Accelerator((1, 7))

        kernel = create_array(0, 1, 7, (7, 7))
        ifmap = create_array(588, 1, 28, (7, 28))

        acc.set_kernel(kernel)
        acc.set_ifmap(ifmap)

        assert acc.conv()

        answer = create_array(140924, 1176, 32928, (22, 22))

        assert np.array_equal(acc.ofmap[0], answer[21])

    def test_7x7_with_large_ifmap(self):
        acc = Accelerator((7, 7))

        kernel = create_array(0, 1, 7, (7, 7))
        ifmap = self.ifmap

        acc.set_kernel(kernel)
        acc.set_ifmap(ifmap)

        assert acc.conv()

        # you probably want to use the excel function 'sumproduct'
        answer = create_array(140924, 1176, 32928, (22, 22))

        assert np.array_equal(acc.ofmap, answer)

    def test_7x1_with_stride(self):
        acc = Accelerator((1, 7), stride=(3, 3))

        kernel = create_array(0, 1, 7, (7, 7))
        ifmap = create_array(588, 1, 28, (7, 28))

        acc.set_kernel(kernel)
        acc.set_ifmap(ifmap)

        assert acc.conv()

        answer = create_array(832412, 3528, 857108, (1, 8))

        assert np.array_equal(acc.ofmap, answer)

    def test_7x7_with_stride(self):
        acc = Accelerator((7, 7), stride=(3, 3))

        kernel = create_array(0, 1, 7, (7, 7))
        ifmap = self.ifmap

        acc.set_kernel(kernel)
        acc.set_ifmap(ifmap)

        assert acc.conv()

        answer = create_array(140824, 3528, 98784, (8, 8))

        assert np.array_equal(acc.ofmap, answer)
