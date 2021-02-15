import unittest
import os
import sys

import numpy as np
#import CMEPDA_exam
from CMEPDA_exam.functionbox import find_tx_asic
#import CMEPDA_exam.functionbox as fun
#from functionbox import find_tx_asic

infile=np.fromfile('/home/rebecca/Scrivania/programmi/energy_tx12_asic8910.dat.dec',dtype=fun.T_ASIC_TEMP_EVENT)

found_tx,found_ic = find_tx_asic(infile)

class Test_findtxasic(unittest.TestCase):
     def test_foundtxasic(self):
         self.assertEqual(found_tx,[12])
         self.assertEqual(found_ic,[8,9,10])



if __name__=='__main__':
    unittest.main()
