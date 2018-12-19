import numpy as np
import copy


"""
all memory accesses are computed by the one making the transaction
Each PE is responsible for making sure that its own trasactions are being
accounted for
"""

class CostTracker:
    """
    All computes are logged with their inputs, so you can see what is actually
    computed. All memory accesses are logged with who _made_ the call, so we
    can verify that this is what we believe to be the case
    """
    def __init__(self):
        self._spad_read = []
        self._ipe_read = []
        self._glb_read = []
        self._dram_read = []

        self._spad_write = []
        self._ipe_write = []
        self._glb_write = []
        self._dram_write = []

        self._add_operations = 0
        self._mult_operations = 0

        self.history = []

    @property
    def add(self):
        return self._add_operations

    @property
    def mult(self):
        return self._mult_operations

    @property
    def SPAD_reads(self):
        return len(self._spad_read)

    @property
    def SPAD_writes(self):
        return len(self._spad_write)

    @property
    def IPE_reads(self):
        return len(self._ipe_read)

    @property
    def IPE_writes(self):
        return len(self._ipe_write)

    @property
    def GLB_reads(self):
        return len(self._glb_read)

    @property
    def GLB_writes(self):
        return len(self._glb_write)

    @property
    def DRAM_reads(self):
        return len(self._dram_read)

    @property
    def DRAM_writes(self):
        return len(self._dram_write)

    def _get_caller_name(self, func):
        return func.__name__.split('_')[1]

    def _get_access_type(self, func):
        return func.__name__.split('_')[0]

    def record_transaction(self, memtype, func):
        translator = {
            "get": "read",
            "set": "write",
        }

        if memtype not in ['spad', 'ipe', 'glb', 'dram', 'acc']:
            raise RuntimeError('Someone forgot to assign an action a cost!')

        if memtype is 'acc':
            # the memory cost has already been paid
            return

        array_name = "_{}_{}".format(
            memtype,
            translator[self._get_access_type(func)])

        array = getattr(self, array_name)

        array.append(self._get_caller_name(func))

    def track_memory_costs(self, func):
        def inner(*args, mem=None, **kwargs):

            self.record_transaction(mem, func)

            return func(*args, **kwargs)
        return inner

    def track_compute_costs(self, func):
        def inner(*args, **kwargs):

            if func.__name__ == 'mult':
                # if the arguments will be the multiplier and the multiplicand

                # if the length of these two do not match up we'd have a
                # runtime error until it is resolved, so using args[0] is valid
                # here
                self._mult_operations += len(args[0])

            if func.__name__ == 'add':
                self._add_operations += 1

            return func(*args, **kwargs)
        return inner

class ProcessingElement:
    MEMORY_ACCESSORS = [
        'get_kernel',
        'set_kernel',
        'get_ifmap',
        'set_ifmap',
        'get_psum',
        'set_psum',
    ]

    COMPUTE_OPERATORS = [
        'mult',
        'add',
    ]

    def _dynamically_decorate_accessors(self):
        """
        Hack. In order to trick the interpreter to decorating our accessors
        and operators after initialization (so that we can point towards the
        self.cost_tracker), we need to call this after initialization

        The benefit of doing this is that we get per PE calculations of memory
        accesses and computations, AND we get to have this without having
        global shared state, which is _dangerous_

        Only modify this if you know what you are doing (no free lunch)
        """

        for accessor in self.MEMORY_ACCESSORS:
            accessor_function = getattr(self, accessor)
            setattr(
                self, accessor,
                self.cost_tracker.track_memory_costs(accessor_function)
            )

        for compute_operation in self.COMPUTE_OPERATORS:
            compute_function = getattr(self, compute_operation)
            setattr(
                self, compute_operation,
                self.cost_tracker.track_compute_costs(compute_function)
            )

    def __init__(self, uuid, cost_tracker=CostTracker,
            stride=(1, 1), autodecorate=True):

        self.cost_tracker = cost_tracker()

        self.uuid = uuid

        self.stride = stride
        self.ifmap_index = None

        self._kernel = None
        self._ifmap = None
        self._psum = None

        self.kernel_set = False
        self.ifmap_set = False

        if autodecorate:
            self.decorate()

    def decorate(self):
        self._dynamically_decorate_accessors()

    def __repr__(self):
        return "PE(uuid: {})".format(self.uuid)

    @property
    def ready(self):
        return self.ifmap_set and self.kernel_set

    @property
    def ofmap(self):
        if self.ready:
            return int((self.ifmap - self.kernel + self.stride[1]) / self.stride[1])

    # recorded
    def get_kernel(self):
        return self._kernel

    # recorded
    def set_kernel(self, kernel):
        self.kernel = len(kernel)
        self._kernel = kernel

        self.kernel_set = True
        self.set_psum_if_ready()

    # recorded
    def get_ifmap(self):
        return self._ifmap

    # recorded
    def set_ifmap(self, ifmap, ifmap_index=0):
        self.ifmap = len(ifmap)
        self.ifmap_index = ifmap_index
        self._ifmap = ifmap

        self.ifmap_set = True
        self.set_psum_if_ready()

    def has_ifmap(self):
        return self._ifmap is not None

    def set_psum_if_ready(self):
        if self.ready:
            self._psum = np.zeros(self.ofmap)

    # recorded
    def get_psum(self):
        return self._psum

    # recorded
    def set_psum(self, psum):
        self._psum = copy.deepcopy(psum)

    def set_psum_zero(self):
        self._psum = np.zeros(self.ofmap)

    # recorded
    def mult(self, multiplicand, multiplier):
        return np.dot(multiplicand, multiplier)

    # recorded
    def add(self, a, b):
        return a + b

    def conv(self):
        """
        We assume here that the execute portion has 0 memory.
        The ALU can only operate on one block (8 bits) at a time

        Therefore we have to pay the spad penalty with every iteration
        We can of course change this if we feel it is appropriate
        """

        if not self.kernel_set or not self.ifmap_set:
            return False

        if self._psum is None:
            self.set_psum_zero()

        for ifmap_idx in range(0, self.ifmap, self.stride[1]):
            if ifmap_idx + self.kernel > self.ifmap:
                continue

            kernel = self.get_kernel(mem='spad')
            ifmap = self.get_ifmap(mem='spad')

            ifmap_section = ifmap[ifmap_idx: ifmap_idx+len(kernel)]
            mult_result = self.mult(kernel, ifmap_section)

            previous_psum = self.get_psum(mem="spad")

            insert_location = int(ifmap_idx/self.stride[1])
            previous_psum[insert_location]= self.add(
                mult_result,
                previous_psum[insert_location])

            self.set_psum(previous_psum, mem="spad")

    def t_shift_kernel_to(self, pe):
        pe.set_kernel(
            self.get_kernel(mem="acc"),
            mem="ipe")

    def t_shift_ifmap_to(self, pe):
        pe.set_ifmap(
            self.get_ifmap(mem="acc"),
            ifmap_index=self.ifmap_index,
            mem="ipe")

    def t_shift_psum_to(self, pe):
        pe.set_psum(
            self.get_psum(mem="acc"),
            mem="ipe")
