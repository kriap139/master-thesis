import unittest
from pysequd import KSpaceV2, KSpace
from Util import Integer, Real
import os
import re
from Util import find_dir_ver, data_dir, parse_cmd_params

class Tests(unittest.TestCase):
    def kspace_v2(self):
        low, high, step = 0, 500, 10
        k = 0

        space = dict(c=Real(low, high, name='c'))
        to_scale = not (high > 0 and (high <= 1))
        ks = KSpaceV2(space, k, x_in_search_space=to_scale)

        print(f"x in search space = {to_scale}")
        for i in range(low, high, step):
            print(f"x = {i}: {ks.kmap('c', i)}")
    
    def test_params_parsing(self):
        from Util.params_file_to_cmd import main as test_cmd
        test = test_cmd(data_dir("kspace_lr_ns_values.json"), [0, 1, 2])
        print(test)
        result = parse_cmd_params(test)
        print(result)


if __name__ == "__main__":
    unittest.main()