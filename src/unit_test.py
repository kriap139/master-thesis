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
    
    def test_merge_train_test(self):
        dataset = Dataset(Builtin.EPSILON, is_test=True)
        frame = dataset.load_frame()
        path = os.path.join(dataset.get_dir(), f"{dataset.name}.arff")
        save_sparse_arff(path, dataset.name, frame)

if __name__ == "__main__":
    unittest.main()