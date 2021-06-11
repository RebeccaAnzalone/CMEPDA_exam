import unittest
import os
import sys

import numpy as np
#import CMEPDA_exam
from functionbox import choose_timestamp
#import CMEPDA_exam.functionbox as fun
#from functionbox import find_tx_asic

time_test=np.array([[0.74839902, 0.62623021, 0.63748906, 0.89671172, 0.75040784,
       0.14497528, 0.86525977, 0.0783974 , 0.63727903, 0.10471543,
       0.64974534, 0.00844584, 0.16752191, 0.31077615, 0.54330714,
       0.70031552, 0.2914229 , 0.56192653, 0.71270034, 0.54292343,
       0.77497922, 0.28793384, 0.91505306, 0.39265403, 0.04562684,
       0.08792984, 0.99157677, 0.46632578, 0.38695858, 0.72098261,
       0.93234535, 0.72385352, 0.02547302, 0.25281507, 0.91704391,
       0.80777443, 0.97468726, 0.86978349, 0.99364778, 0.68244646,
       0.20361908, 0.07019947, 0.48117102, 0.61202654, 0.37728102,
       0.05367766, 0.53809308, 0.34679234, 0.95365412, 0.25575233,
       0.67016085, 0.91919217, 0.28037343, 0.07898154, 0.19494979,
       0.77425622, 0.28823167, 0.82626201, 0.51391462, 0.67853972,
       0.13315281, 0.17523223, 0.9227037 , 0.45419402]])

charge_test=np.array([[0.76995785, 0.6355348 , 0.5756627 , 0.08969229, 0.77586248,
       0.19257917, 0.65415776, 0.21485688, 0.02510655, 0.09932094,
       0.00928316, 0.00814471, 0.66123966, 0.67368423, 0.95567624,
       0.36714587, 0.14689133, 0.67875668, 0.94154075, 0.35915744,
       0.95985554, 0.93809096, 0.23901099, 0.74184403, 0.65413792,
       0.98932719, 0.35047885, 0.50331898, 0.16662463, 0.56418033,
       0.01151614, 0.21436056, 0.12556977, 0.94464585, 0.72050024,
       0.07278645, 0.45902621, 0.82767906, 0.20667132, 0.84944818,
       0.76531206, 0.23801266, 0.70772068, 0.5015658 , 0.79940703,
       0.81600886, 7.31696158, 0.85465315, 0.85986744, 0.28303582,
       0.94044149, 0.16094009, 0.22181081, 0.4929607 , 0.19836718,
       0.16061559, 0.58834767, 0.70858657, 0.66354087, 0.65736604,
       0.92466821, 0.58518594, 0.6011225 , 0.98069179]])



timestamp = choose_timestamp(False,time_test,charge_test)
timestamp_average = choose_timestamp(True,time_test,charge_test)

class Test_findpeak(unittest.TestCase):
     def test_foundpeak(self):
         self.assertEqual(timestamp,[0.53809308])
         self.assertEqual(timestamp_average,[0.517557872350621])




if __name__=='__main__':
    unittest.main()