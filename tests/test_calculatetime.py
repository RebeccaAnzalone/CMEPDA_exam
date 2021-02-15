import unittest
import os

import numpy as np
#import CMEPDA_exam.functionbox as fun
#from functionbox import calculate_timestamps
import functionbox as fun

infile=np.fromfile('/home/rebecca/Scrivania/programmi/energy_tx12_asic8910.dat.dec',dtype=fun.T_ASIC_TEMP_EVENT)
found_tx, found_ic = fun.find_tx_asic(infile)
for sel_tx_i, sel_tx in enumerate(found_tx):
    tx_ids = infile['tx_id']==sel_tx
    for sel_ic_i, sel_ic in enumerate(found_ic):
        tx_ic_ids = np.logical_and(tx_ids,infile['asic_id']==sel_ic)
        array_t = fun.calculate_timestamps(infile[tx_ic_ids],sel_tx,sel_ic, tdc_calibration)
        array_e = (infile[tx_ic_ids]['energy']  - pedestal['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)])
        array_2t = fun.choose_timestamp(False, array_t, array_e)


class Test_calculatetime(unittest.TestCase):
     def test_calculatetime(self):
         self.assertEqual(found_tx,[12])
         self.assertEqual(found_ic,[8,9,10])



if __name__=='__main__':
    unittest.main()
