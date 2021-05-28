import unittest
import os
import sys

import numpy as np
#import CMEPDA_exam
from functionbox import find_tx_asic
from functionbox import T_ASIC_TEMP_EVENT
#import CMEPDA_exam.functionbox as fun
#from functionbox import find_tx_asic

infile=np.fromfile('file.dat.dec',dtype=T_ASIC_TEMP_EVENT)

with open('file_test.dat.dec','wb+') as f:
    test=infile[:15]
    test.tofile(f)


found_tx,found_ic = find_tx_asic(infile)

class Test_findtxasic(unittest.TestCase):
     def test_foundtxasic(self):
         self.assertEqual(found_tx,[12])
         self.assertEqual(found_ic,[2])



if __name__=='__main__':
    unittest.main()
