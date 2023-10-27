import matplotlib.pyplot as plt
import os, sys, json, zlib
import numpy as np
import functionbox as fun
from pylab import subplots, hist, plot, Circle, cm
from skimage.feature import blob_log
from skimage.segmentation import watershed
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import norm
import argparse

import cProfile

import concurrent.futures
import threading

import time
import multiprocessing as mp
from multiprocessing import Process

import line_profiler
profile = line_profiler.LineProfiler()

@profile
def pedestal_and_tdc(input_file_name,count,pedestals_filename,tdc_filename):
    """
    'pedestal_and_tdc' is a function that calculates pedestal values and find the right fine time values for every pixel computing 'functionbox.find_pedestals' and 'functionbox.find_tdc' functions.

    Parameters
    ----------
    input_file_name : .dat.dec
                      acquisition file
    count : int
            size of subfile to process
    pedestals_filename : .json
                         output file containing pedestal values
    tdc_filename : .json
                   output file containing calibrated fine time values
    Returns
    -------
    None : None
    """

    i, j = 0, 314
    pedestal_array=np.zeros((fun.N_TX,fun.N_ASIC,fun.T_ASIC_CHANNELS,fun.PEDESTAL_BINS-1))
    tdc_array=np.zeros((fun.N_TX,fun.N_ASIC,fun.T_ASIC_CHANNELS,fun.TDC_BINS))
    while (i >=0 and j >0):
        infile = np.fromfile(input_file_name,dtype=fun.T_ASIC_TEMP_EVENT,count=int(count),offset=((fun.T_ASIC_TEMP_EVENT.itemsize)*i*int(count)))
        found_tx,found_ic=fun.find_tx_asic(infile)
        if os.path.isfile(pedestals_filename):
            print('pedestals file already exist!')
        else:
            pedestal_array+=fun.hist_pedestals(infile,found_tx,found_ic)
        if os.path.isfile(tdc_filename):
            print('tdc calibration file already exist!')
        else:
            tdc_array += fun.hist_tdc(infile,found_tx,found_ic)
        print(i)
        i +=1
        if (infile.size != int(count)) or (os.path.isfile(pedestals_filename) and os.path.isfile(tdc_filename)):
            j = -200
    if not os.path.isfile(pedestals_filename):
        fun.find_pedestals(pedestal_array,found_tx,found_ic,pedestals_filename)
    if not os.path.isfile(tdc_filename):
        fun.find_tdc(tdc_array,found_tx,found_ic,tdc_filename)


@profile
def crystal_map(input_file_name,count,crystal_map_filename,pedestals_filename,showmaps):
    """
    'crystal_map' is a function that gives a file containing informations about crystal map computing 'functionbox.floodmap' and 'functionbox.find_LUT' functions.

    Parameters
    ----------
    input_file_name : .dat.dec
                      acquisition file
    count : int
            size of subfile to process
    crystal_map_filename : .zjson
                           output file with crystal map informations
    pedestals_filename : .json
                         file containing pedestal values
    showmaps : boolean variable
               when it's set to TRUE the function 'functionbox.plot_maps' is activated
    Returns
    -------
    None : None
    """
    if os.path.isfile(crystal_map_filename):
        print('LUT does already exist!')
    else:
        with open('pedestals.json') as json_file0:
            data0 = json.load(json_file0)
        floodmap_array = np.zeros((fun.N_TX,fun.N_ASIC,fun.FLOODMAP_SIZE,fun.FLOODMAP_SIZE))
        i, j = 0, 314
        while(i>=0 and j>0):
            print(i)
            infile = np.fromfile(input_file_name,dtype=fun.T_ASIC_TEMP_EVENT,count=int(count),offset=((fun.T_ASIC_TEMP_EVENT.itemsize)*i*int(count)))
            found_tx,found_ic=fun.find_tx_asic(infile)
            floodmap_array += fun.floodmap(infile, found_tx, found_ic, data0)[0]
            i +=1
            if (infile.size != int(count)):
                j = -200
        fun.find_LUT(floodmap_array,found_tx,found_ic,showmaps,crystal_map_filename)
        print('crystal map file written correctly !')



@profile

def main_timestamp(input_file_name,count,crystal_map_filename,pedestals_filename,tdc_filename,file_coincidenze,CW,file_energy_calibration,arr_resolution):
    """
    Main function that computes 'functionbox.compute_events', 'functionbox.find_energy_calibration', 'functionbox.apply_energy_calibration' and 'functionbox.find_energy_resolution' functions to obtain a file containing all the coincidences and an array with all the energy calibrated values.

    Parameters
    ----------
    input_file_name : .dat.dec
                      acquisition file
    count : int
            size of subfile to process
    crystal_map_filename : .zjson
                           file with crystal map informations
    pedestals_filename : .json
                         file containing pedestal values
    tdc_filename : .json
                   file containing calibrated fine time values
    file_coincidenze : .dat.dec
                       output file with coincidences
    CW : int
         Coincidence Window
    file_energy_calibration : .json
                              file containing energy calibration coefficients
    arr_resolution : numpy.array
                     array with all the energy calibrated values used to compute energy resolution
    Returns
    -------
    file_coincidenze : .dat.dec
                       output file with coincidences
    arr_risoluzione_energetica : numpy.array
                                 array with all the energy calibrated values
    """
    lista = []
    with open(pedestals_filename) as json_file0:
        data0 = json.load(json_file0)
    with open(tdc_filename) as json_file1:
        data1 = json.load(json_file1)
    with open(file_coincidenze,'wb+') as f:
        data2 = json.loads(zlib.decompress(open(crystal_map_filename,'rb').read()).decode('ascii'))
        i, j = 0, 314
        arr_calibrazione_energetica = np.zeros((fun.N_TX,fun.N_ASIC,fun.T_ASIC_CHANNELS+1,fun.N_BINS))   #array with energy calibration informations
        arr_risoluzione_energetica = np.zeros((fun.N_TX,fun.N_ASIC,fun.T_ASIC_CHANNELS+1,fun.N_BINS))

        while(i>=0 and j>0):
            print(i)
            infile = np.fromfile(input_file_name,dtype=fun.T_ASIC_TEMP_EVENT,count=int(count),offset=((fun.T_ASIC_TEMP_EVENT.itemsize)*i*int(count)))
            found_tx,found_ic=fun.find_tx_asic(infile)

            coincidences,arr_energy = fun.compute_events(infile,found_tx,found_ic,data0,data1,data2,CW)
            print('number of coincidences', int(np.shape(coincidences)[0]/2),'-------->', np.shape(coincidences)[0]/(int(count)/2))
            lista.append(np.shape(coincidences)[0]/(int(count)/2))
            arr_calibrazione_energetica += arr_energy
            coincidences.tofile(f)
            i +=1
            if (infile.size != int(count)):
                j = -200




    if os.path.isfile(file_energy_calibration):
        print('energy_calibration file already exist!')
    else:
        fun.find_energy_calibration(found_tx, found_ic,arr_calibrazione_energetica,file_energy_calibration)

    with open(file_energy_calibration) as json_file3:
        data3 = json.load(json_file3)

    i, j = 0, 314
    while(i>=0 and j>0):
        infile = np.fromfile(file_coincidenze,dtype=fun.T_SINGLE_EVENT_PIXELATED,count=int(count),offset=((fun.T_SINGLE_EVENT_PIXELATED.itemsize)*i*int(count)))
        found_tx,found_ic=fun.find_tx_asic(infile)
        found_pixel = list(map(int,np.unique(infile['pixel_id'])))
        arr_spectra = fun.apply_energy_calibration(infile,data3, found_tx, found_ic)
        arr_risoluzione_energetica += arr_spectra
        i +=1
        if (infile.size != int(count)):
            j = -200


    print('computing energy resolution')
    fun.find_energy_resolution(arr_risoluzione_energetica)



    return file_coincidenze, arr_risoluzione_energetica




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program takes in input a raw file and gives in output a coincidences file')
    parser.add_argument('-f','--input_file',help='input file to process')
    parser.add_argument('-c','--count_events', help='size of subfile to process',default=2_000_000)
    parser.add_argument('-cw','--coincidences_window', help = 'lenght of coincidences window in ns', default = 10)
    parser.add_argument('-p','--pedestals_filename',help='json file that stores pedestals value',default='pedestals.json')
    parser.add_argument('-tdc','--tdc_calibration_filename',help='json file that stores tdc calibration values',default='tdc_calibration.json')
    parser.add_argument('-lut','--crystal_map_filename',help='json file that stores LUT',default='crystals.zjson')
    parser.add_argument('-showmaps','--show_floodmap_LUT',help='Show floodmaps and LUT for each ASIC',default=False)
    parser.add_argument('-calib','--energy_calibration', help = 'json file that stores energy calibration values', default = 'en_calibration.json')
    parser.add_argument('-o','--outfile', help = 'output file containing coincidences events', default = 'coincidences.dat.dec')
    parser.add_argument('-ris','--resolution_array', help = 'array containing energy resolution values')
    parser.add_argument('-CTR','--coincidence_time_resolution', help='compute and show histogram of time differences', default = True)
    parser.add_argument('-spectrum', '--plot_pixel_spectra', help='this is a list where the first element is True/False and the remains stand for [[TX], [ASIC]]', default = [False,[12,13],[8,9,10]])
    parser.add_argument('-ew','--energy_window', help='this is a list where the first element is True/False and the remains stand for [energy_min, energy_max]', default = [False,fun.ENERGY_WINDOW[0],fun.ENERGY_WINDOW[1]])
    parser.add_argument('-mp','--multiprocessing',help='activate multiprocessing to compute "pedestal_and_tdc" function',default=False)


    args=parser.parse_args()


    if args.multiprocessing:
        start = time.perf_counter()

        processes=[]
        for _ in range(4):
            p = mp.Process(target=pedestal_and_tdc, args=[args.input_file,args.count_events,args.pedestals_filename,args.tdc_calibration_filename])
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        finish = time.perf_counter()
        print(f'Finished in {round(finish-start, 2)} seconds')

    else:
         pedestal_and_tdc(args.input_file,args.count_events,args.pedestals_filename,args.tdc_calibration_filename)

    crystal_map(args.input_file,args.count_events,args.crystal_map_filename,args.pedestals_filename, args.show_floodmap_LUT)
    coincidences_file,spectrum_array = main_timestamp(args.input_file,args.count_events,args.crystal_map_filename,args.pedestals_filename,args.tdc_calibration_filename, args.outfile, args.coincidences_window, args.energy_calibration, args.resolution_array)

    if args.coincidence_time_resolution:
        fun.find_CTR(coincidences_file,args.count_events, args.energy_window)
    if args.plot_pixel_spectra[0]:
        fun.plot_spectrum(spectrum_array, args.plot_pixel_spectra[1],args.plot_pixel_spectra[2])
