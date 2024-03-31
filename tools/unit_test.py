import unittest
from pysequd import KSpaceV2, KSpace
from Util import Integer, Real

class Tests(unittest.TestCase):
    def test_kspace_v2(self):
        low, high, step = 0, 500, 10
        k = 0

        space = dict(c=Real(low, high, name='c'))
        to_scale = not (high > 0 and (high <= 1))
        ks = KSpaceV2(space, k, x_in_search_space=to_scale)

        print(f"x in search space = {to_scale}")
        for i in range(low, high, step):
            print(f"x = {i}: {ks.kmap('c', i)}")


if __name__ == "__main__":
    unittest.main()