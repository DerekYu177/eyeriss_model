import numpy as np
import copy

from .processing_element import ProcessingElement as PE
from .ifmap_pipes import PipeCoordinator

class GlobalCostTracker:
    def build_result_holder(self):
        pes = self.pes.flatten()

        results = {}
        for pe in pes:
            results[pe.__repr__] = pe.cost_tracker

        return results

    def __init__(self, accelerator):
        self.pes = accelerator.pes
        self.results = self.build_result_holder()

    def record(self):
        """
        We take advantage of pass by reference
        and record the statistics for all PEs
        """
        pass

class Accelerator:
    def setup_pe_array(self):
        """
        e.g. (2, 2)

        numpy-style arrays
        [[(0, 0), (1, 0)], [(0, 1), (1, 1)]]

        is equivalent to

        [
            [(0, 1), (1, 1)],
            [(0, 0), (1, 0)]
        ]
        """

        cols = self.dimensions[0]
        rows = self.dimensions[1]

        pes = np.array(
            [[self.processing_element for j in range(cols)] for i in range(rows)],
            self.processing_element)

        for i, pe_row in enumerate(pes):
            for j, pe in enumerate(pe_row):
                pes[i][j] = pes[i][j]((i, j), stride=self.stride)

        return pes

    def __init__(self, dimensions,
            stride=(1, 1),
            processing_element=PE,
            global_cost_tracker=GlobalCostTracker,
            pipe_coordinator=PipeCoordinator):

        self.dimensions = dimensions

        # stride is given by (row, column)
        # row is vertical jump
        # column is horizontal jump
        self.stride = stride
        self.processing_element = processing_element

        self.pes = self.setup_pe_array()
        self.kernel_set = False
        self.ifmap_set = False
        self.psum_set = False

        self.ifmap = None
        self.kernel = None
        self.ofmap = None

        self.pipe_coordinator = pipe_coordinator
        self.global_cost_tracker = global_cost_tracker(self)

    def ofmap_dimensions(self):
        ifmap_width, ifmap_height = self.ifmap.shape
        kernel_width, kernel_height = self.kernel.shape
        stride_width, stride_height = self.stride

        ofmap_width = int((ifmap_width - kernel_width + stride_width) / stride_width)
        ofmap_height = int((ifmap_height - kernel_height + stride_height) / stride_height)

        return (ofmap_width, ofmap_height)

    def set_kernel(self, kernel):
        """
        We want to assign the top row of the kernel to the PE at the top
        of the PE array, lefthand side
        """
        self.kernel = kernel

        for idx, kernel_row in enumerate(kernel):
            pe = self.pes[-1-idx][0]
            pe.set_kernel(kernel_row, mem="dram")

        self.kernel_set = True

    def set_ifmap(self, ifmap):
        """
        We delegate this to the pipe coordinator
        """
        self.ifmap = ifmap

        ifmap_row_size = ifmap.shape[1]
        pe_height = self.pes.shape[0]

        self.pipe_coordinator = self.pipe_coordinator(self.pes, ifmap_row_size)
        self.pipe_coordinator.setup(ifmap)

        self.pipe_coordinator.update_pes()
        self.ifmap_set = True

    def max_iterations_per_ifmap(self):
        """
        The maximum number of iterations is related to the length of the
        longest pipe
        """
        return max([len(pipe) for pipe in self.ifmap_pipeline])

    def run_ready_PEs(self, row):
        """
        if a PE has both ifmap and kernel, it is ready
        it's our responsibility to ensure that it has the right psums
        """
        for pe in self.pes[row]:
            if not pe.ready:
                continue

            pe.conv()

    def propagate_ifmaps(self):
        """
        ifmaps propagate diagonally up and to the right
        TODO: Stride is an interesting problem

        We look at this from last column to the first column.
        Each PE will find the PE whom they copy from. If the PE exists, take
        the value. If the PE does not exist, forget it.
        """
        for pe_col in self.pes[::-1]:
            for pe in pe_col:
                stride_location = np.array(pe.uuid) - np.array(self.stride)

                if np.any(stride_location < (0, 0)):
                    continue

                stride_PE = self.pes[stride_location[0]][stride_location[1]]
                if not stride_PE.has_ifmap():
                    continue

                stride_PE.t_shift_ifmap_to(pe)

        self.pipe_coordinator.update_pes()

    def propagate_kernel(self):
        """
        kernel propagates from right to left, in each row
        """
        for row_idx, pe_row in enumerate(self.pes):
            for pe_idx in range(len(pe_row)-1):
                current_pe = self.pes[row_idx][pe_idx]
                right_pe = self.pes[row_idx][pe_idx+1]

                current_pe.t_shift_kernel_to(right_pe)

    def propagate_psums(self, row_idx):
        """
        psums propagate from bottom to top, in each column
        computed per row
        """
        for pe_idx, current_pe in enumerate(self.pes[row_idx]):
            location = np.array(current_pe.uuid) + np.array((1, 0))

            if np.any(location >= self.pes.shape):
                continue

            upper_pe = self.pes[location[0]][location[1]]

            current_pe.t_shift_psum_to(upper_pe)

    def conv(self):
        """
        order of operations:
        1. all PEs need to be fed kernels
        2. all PEs need to be fed ifmaps
        3. all bottom PEs perform #conv
        4. psums are transmitted bottom up
        5. upper level PEs perform #conv
        6. Top level PEs spit out final sums
        7. These are the output convolutions
        8. Repeat 1-7 for the entire ifmap
        9. Returns the entire ofmap for the conv section

        If anything goes wrong, we return False
        """

        if not (self.ifmap_set and self.kernel_set):
            return False

        ofmap_index_seed_scaling_factor = self.ifmap[0][0] / self.ifmap.shape[1]
        self.ofmap = np.zeros(self.ofmap_dimensions())

        for _i in self.ofmap:
            for row_idx in range(len(self.pes)):
                self.run_ready_PEs(row_idx)
                self.propagate_psums(row_idx)

            # this is a crude fix
            for pe in self.pes[0]:
                pe.set_psum_zero()

            # take the PEs at the top and find out where they belong
            for top_level_pe in self.pes[-1]:
                ofmap_index = self._scale_ofmap_index(
                    top_level_pe.ifmap_index,
                    ofmap_index_seed_scaling_factor)

                if ofmap_index is None:
                    continue

                if ofmap_index >= len(self.ofmap):
                    continue

                self.ofmap[ofmap_index] = top_level_pe.get_psum(mem="dram")

            self.propagate_kernel()
            self.propagate_ifmaps()

        self.global_cost_tracker.record()
        return True

    def _scale_ofmap_index(self, ofmap_index, seed_scale_factor):
        """
        This adjusts the ofmap_index and handles the case where we don't start
        from the 0, 0 index.

        For example, if we were to had a matrix that was:

        588, 589, ... 594
        616, 617, ... 622
        ..., ..., ... ...
        756, 757, ... 763

        We want the index to start at 588 / 28 - 1 = 20

        # TODO
        Now that I think about it, this should really belong in
        the Processing Element
        """

        if int(seed_scale_factor) == 0:
            return ofmap_index

        return int(ofmap_index / seed_scale_factor) - 1

