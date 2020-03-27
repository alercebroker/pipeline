import pandas as pd
import numpy as np
import mhps
import unittest


class TestMHPS(unittest.TestCase):
    df = pd.read_csv("ZTF18aaajeru.csv", index_col="oid")
    t1 = 100
    t2 = 10

    def test_fid_1(self):
        data = self.df[self.df.fid == 1]
        mag = data.magpsf_corr
        magerr = data.sigmapsf_corr
        time = data.mjd
        result = mhps.statistics(mag.values, magerr.values, time.values, self.t1, self.t2)
        result = np.array(result)
        self.assertEqual(len(result), 5)

    def test_fid_2(self):
        data = self.df[self.df.fid == 2]
        mag = data.magpsf_corr
        magerr = data.sigmapsf_corr
        time = data.mjd
        result = mhps.statistics(mag.values, magerr.values, time.values, self.t1, self.t2)
        result = np.array(result)
        self.assertEqual(len(result), 5)



if __name__ == '__main__':
    unittest.main()