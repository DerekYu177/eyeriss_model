import numpy as np

class Pipe:
    """
    Tasked with delivering ifmap data
    """
    @property
    def empty(self):
        return len(self.ifmaps) == 0 and len(self.indices) == 0

    def __len__(self):
        return len(self.ifmaps)

    def __repr__(self):
        return "Pipe(len: {)}".format(len(self.ifmaps))

    def __init__(self, pe_uuid, ifmap_size):
        self.uuid = pe_uuid
        self.ifmap_size = ifmap_size

        # ifmaps and indicies must move in lockstep
        self.ifmaps = []
        self.indices = []

        self.active_pointer = 0

        self.empty_ifmap = np.zeros(ifmap_size)

    def pad_with_zeros(self, num_padded_entries):
        self.ifmaps.extend([self.empty_ifmap for i in range(num_padded_entries)])
        self.indices.extend([None for i in range(num_padded_entries)])

    def append(self, ifmap):
        self.ifmaps.append(ifmap)
        self.indices.append(int(ifmap[0] / self.ifmap_size)) # this is related to the index

    def extend(self, ifmaps):
        for ifmap in ifmaps:
            self.append(ifmap)

    def pop(self):
        """
        returns the elements at active_pointer
        for both ifmaps and indicies
        """

        if self.active_pointer >= len(self):
            return self.empty_ifmap, None

        ifmap = self.ifmaps[self.active_pointer]
        index = self.indices[self.active_pointer]

        self.active_pointer += 1

        if type(index) == float:
            index = int(index)

        return ifmap, index

class PipeCoordinator:
    def __init__(self, pes, ifmap_row_size):
        self.pes = pes
        self.pes_height = pes.shape[0]

        self.ifmap_row_size = ifmap_row_size
        self.ifmap_pes = self.ifmap_pes()

    def ifmap_pes(self):
        input_pes = {}

        # first element of row, starting from the top
        for pe_row in self.pes[::-1]:
            pe = pe_row[0]
            input_pes[pe] = None

        # bottom row minus the first element
        for pe in self.pes[0]:
            if pe.uuid == (0, 0):
                continue

            input_pes[pe] = None

        return input_pes

    def setup(self, ifmaps):
        self._attach_pipes_to_pes()
        self._pad_pes_with_zeros()
        self._fill_pes(ifmaps)

    def _attach_pipes_to_pes(self):
        for pe in self.ifmap_pes.keys():
            self.ifmap_pes[pe] = Pipe(pe.uuid, self.ifmap_row_size)

    def _pad_pes_with_zeros(self):
        for pe, pipe in self.ifmap_pes.items():
            distance_from_corner = pe.uuid[1]
            pipe.pad_with_zeros(distance_from_corner)

    def _fill_pes(self, ifmaps):
        """
        For a 3x3, the pipes look like

         pipes | ifmap row #s
        ------------------------------
        pipe 1 | 0 3 6 9 ...
        pipe 2 | 1 4 7 10 ...
        pipe 3 | 2 5 8 11 ...
        pipe 4 | X 3 6 9 ...
        pipe 5 | X X 4 7 ...

        We generate the first pipe (prime ifmap indices) and then +1
        for each subsequent pipe
        """
        self.ifmaps = ifmaps
        prime_ifmap_indices = np.array(list(range(0, len(ifmaps), self.pes_height)))

        for index, pipe in enumerate(self.ifmap_pes.values()):
            ifmap_indices = prime_ifmap_indices + index
            ifmap_indices = filter(self.in_ifmap, ifmap_indices)

            pe_ifmap = [ifmaps[i] for i in ifmap_indices]
            pipe.extend(pe_ifmap)

    def in_ifmap(self, index):
        return index < len(self.ifmaps)

    def update_pes(self):
        for pe, pipe in self.ifmap_pes.items():
            ifmap, index = pipe.pop()
            pe.set_ifmap(
                ifmap,
                ifmap_index=index,
                mem="dram")
