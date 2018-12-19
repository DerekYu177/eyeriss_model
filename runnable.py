import xmltodict
import dicttoxml

from mapping import accelerator
from mapping import test_helper # turns out create_array is useful

OUTPUT_FILE_NAME = "output_filled.xml"

class XML_IO:
    FILE_NAME = "output.xml"

    CONV_INTERNAL_SIGNAL_TO_TAG = {
        "total_dram_reads": "dram_read",
        "total_dram_writes": "dram_write",

        # "total_glb_reads": "glb_read",
        # "total_glb_writes": "glb_write",

        "total_ipe_reads": "ipe_read",
        "total_ipe_writes": "ipe_write",

        "total_spad_reads": "spad_read",
        "total_spad_writes": "spad_write",

        "add_operations": "add",
        "mult_operations": "mult",
    }

    def __init__(self):
        self.interface = self.xml()

    def xml(self):
        xml = ""
        with open(self.FILE_NAME) as f:
            xml = f.read()

        return xmltodict.parse(xml)

    def write(self, results):
        """
        Modify self.interface here
        """
        internals = self.interface['eyeriss']
        self.results = results

        # the below are all pass by reference
        self.modify_pe_array(internals)
        self.modify_conv_layer(internals)
        self.modify_fc_layers(internals)

    def modify_pe_array(self, internals):
        data = internals['pe_array']

        data['pe_height'] = Results.PE_SIZE[0]
        data['pe_width'] = Results.PE_SIZE[1]

    def modify_conv_layer(self, internals):
        data = internals['conv_layer']

        for internal_signal, tag in self.CONV_INTERNAL_SIGNAL_TO_TAG.items():
            data[tag] = self.results[internal_signal]

        data['ifmap_height'] = Results.IFMAP_SIZE[0]
        data['ifmap_width'] = Results.IFMAP_SIZE[1]

        data['filter_height'] = Results.FILTER_SIZE[0]
        data['filter_width'] = Results.FILTER_SIZE[1]

        data['filter_channel_size'] = 1 # only support one channel atm

        data['stride_height'] = Results.STRIDE_SIZE[0]
        data['stride_width'] = Results.STRIDE_SIZE[1]

    def modify_fc_layers(self, internals):
        data = internals['fc_layer']

        data['layers'] = 3

        layer_1 = data['layer_1']
        layer_1['weight_width'] = 16
        layer_1['weight_height'] = 22 * 22 * 4

        layer_2 = data['layer_2']
        layer_2['weight_width'] = 32
        layer_2['weight_height'] = 10

        layer_3 = data['layer_3']
        layer_3['weight_width'] = 10
        layer_3['weight_height'] = 1

    def save_to_file(self, output_file_name):
        xml = dicttoxml.dicttoxml(
            self.interface['eyeriss'],
            attr_type=False,
            custom_root="eyeriss")

        with open(output_file_name, "w") as f:
            f.write(xml.decode())

class Results:
    PE_SIZE = (7, 7)
    IFMAP_SIZE = (28, 28)
    FILTER_SIZE = (7, 7)
    STRIDE_SIZE = (1, 1)

    def __init__(self):
        self.accelerator = accelerator.Accelerator(
            self.PE_SIZE,
            stride=self.STRIDE_SIZE)

        self.kernel = test_helper.create_array(0, 1, 7, self.FILTER_SIZE)
        self.ifmap = test_helper.create_array(0, 1, 28, self.IFMAP_SIZE)

    def run(self):
        self.accelerator.set_kernel(self.kernel)
        self.accelerator.set_ifmap(self.ifmap)

        self.accelerator.conv()

    def produce(self):
        results = {}

        results["total_spad_reads"]     = self.for_all_pes("SPAD_reads")
        results["total_ipe_reads"]      = self.for_all_pes("IPE_reads")
        results["total_glb_reads"]      = self.for_all_pes("GLB_reads")
        results["total_dram_reads"]    = self.for_all_pes("DRAM_reads")

        results["total_spad_writes"]    = self.for_all_pes("SPAD_writes")
        results["total_ipe_writes"]     = self.for_all_pes("IPE_writes")
        results["total_glb_writes"]     = self.for_all_pes("GLB_writes")
        results["total_dram_writes"]    = self.for_all_pes("DRAM_writes")

        results["add_operations"]       = self.for_all_pes("add")
        results["mult_operations"]      = self.for_all_pes("mult")

        return results

    def for_all_pes(self, func):
        """
        only works if all pe.cost_tracker.func return integers
        """
        tracker = 0

        for pe_row in self.accelerator.pes:
            for pe in pe_row:
                tracker += getattr(pe.cost_tracker, func)

        return tracker

def produce_results():
    r = Results()
    r.run()

    return r.produce()

def write_to_interface(results):
    io = XML_IO()
    io.write(results)

    io.save_to_file(OUTPUT_FILE_NAME)

if __name__ == "__main__":
    results = produce_results()
    write_to_interface(results)

