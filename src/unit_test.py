import unittest
from kspace import KSpaceV2, KSpace
from Util import Integer, Real
import os
import re
import numpy as np
from Util import find_dir_ver, data_dir, parse_cmd_params, count_lines, remove_lines_up_to, Dataset, Builtin, save_sparse_arff

class Tests(unittest.TestCase):
    def kspace_v2(self):
        k = 0

        space = dict(
            c=Real(0.0, 1.0, name='c'),
            d=Integer(0, 500, name='d')
        )

        ks = KSpaceV2(space, k, x_in_search_space=True)
        
        low, high = space['c'].low, space['c'].high
        for i in np.arange(low, high, 0.001):
            i_mapped = ks.kmap('c', i)
            if i_mapped != i:
                print(f"Real: x = {i}: y={i_mapped}")
        
        low, high = space['d'].low, space['d'].high
        for i in range(low, high):
            i_mapped = ks.kmap('d', i)
            if i_mapped != i:
                print(f"Integer: x = {i}: y={i_mapped}")
    
    def params_parsing(self):
        from Util.params_file_to_cmd import main as test_cmd
        test = test_cmd(data_dir("kspace_lr_ns_values.json"), [0, 1, 2])
        print(test)
        result = parse_cmd_params(test)
        print(result)
    
    def line_count(self) -> int:
        out_fp = data_dir(add="R-1111-test.out")
        assert os.path.exists(out_fp)
        n_lines = count_lines(out_fp)
        assert n_lines == 10
        return n_lines
    
    def remove_lines(self):
        out_fp = data_dir(add="R-1111-test.out")
        n_lines = self.test_line_count()
        remaining = remove_lines_up_to(out_fp, n_lines - 1)
        assert len(remaining) == 1
    
    def dir_ver(self):
        new_version = find_dir_ver(data_dir(add="test_results/KSpaceOptunaSearch[acsi;nparams=6,kparams=2]", make_add_dirs=False))
        print(new_version)
    
    def dataset_size_sorted(self):
        datasets = {
            "okcupid_stem": 50_700 * 20,
            "wave_e": 72_000 * 49,
            "accel": 153_000 * 5,
            "fps": 426_000 * 45,
            "rcv1": 698_000 * 47_000,
            "acsi": 1_660_000 * 12,
            "delays_zurich": 5_500_000 * 12,
            "higgs": 11_000_000 * 28,
            "electricity": 38474 * 7,
            "epsilon": 500_000 * 2000,
            "puf_128": 6_000_000 * 128,
            "hepmass": 10_500_000 * 28,
            "comet_mc": 7_619_400 * 5
        }
        sorted_datasets = dict(sorted(datasets.items(), key=lambda item: item[1]))
        print(sorted_datasets)
    
    def test_merge_train_test(self):
        dataset = Dataset(Builtin.EPSILON, is_test=True)
        path = os.path.join(dataset.get_dir(), f"{dataset.name}.arff")
        save_sparse_arff(path, dataset.name, dataset.load_frame)

if __name__ == "__main__":
    unittest.main()