import numpy as np

def create_array(seed, col_inc, row_inc, size):
    array = np.full(size, seed)

    for ridx, row in enumerate(array):
        for cidx, col in enumerate(row):
            array[ridx][cidx] += cidx * col_inc + ridx * row_inc

    return array

import unittest
class TestTestHelper(unittest.TestCase):
    def test_create_array(self):
        answer = np.array([
            list(range(28)),
            list(range(28, 28*2)),
            list(range(28*2, 28*3)),
        ])

        assert np.array_equal(
            answer, create_array(0, 1, 28, (3, 28)))

    def test_create_square_array(self):
        answer = np.array([
            list(range(28*0, 28*1)),
            list(range(28*1, 28*2)),
            list(range(28*2, 28*3)),
            list(range(28*3, 28*4)),
            list(range(28*4, 28*5)),
            list(range(28*5, 28*6)),
            list(range(28*6, 28*7)),
            list(range(28*7, 28*8)),
            list(range(28*8, 28*9)),
            list(range(28*9, 28*10)),
            list(range(28*10, 28*11)),
            list(range(28*11, 28*12)),
            list(range(28*12, 28*13)),
            list(range(28*13, 28*14)),
            list(range(28*14, 28*15)),
            list(range(28*15, 28*16)),
            list(range(28*16, 28*17)),
            list(range(28*17, 28*18)),
            list(range(28*18, 28*19)),
            list(range(28*19, 28*20)),
            list(range(28*20, 28*21)),
            list(range(28*21, 28*22)),
            list(range(28*22, 28*23)),
            list(range(28*23, 28*24)),
            list(range(28*24, 28*25)),
            list(range(28*25, 28*26)),
            list(range(28*26, 28*27)),
            list(range(28*27, 28*28)),
        ])

        assert np.array_equal(
            answer, create_array(0, 1, 28, (28, 28)))
