import matplotlib.pyplot as plt
import os, sys, json, zlib
import numpy as np
import functionbox as fun
from pylab import subplots, hist, plot, Circle, cm
from skimage.feature import blob_log
from skimage.morphology import watershed
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import norm
import argparse

"""def pedestals(input_file_name,count,pedestals_filename):
    if os.path.isfile(pedestals_filename):
        print('pedestals file already exist!')
    else:
        i = 0
        j = 314
        pedestal_array=np.zeros((18,12,64,99)) #tx,asic,channels,bins
        while (i >=0 and j >0):
            print(i)
            infile = np.fromfile(input_file_name,dtype=fun.T_ASIC_TEMP_EVENT,count=count,offset=((fun.T_ASIC_TEMP_EVENT.itemsize)*i*count))
            found_tx,found_ic=fun.find_tx_asic(infile)
            pedestal_array+=fun.hist_pedestals(infile,found_tx,found_ic)
            i += 1
            if (infile.size != count):
                j = -200

        fun.find_pedestals(pedestal_array,found_tx,found_ic,pedestals_filename)
        print('pedestals file written correctly!')

def tdc(input_file_name,count,tdc_filename):
    if os.path.isfile(tdc_filename):
        print('tdc calibration file already exist!')
    else:
        tdc_array=np.zeros((18,12,64,1024)) #tx,asic,channels,bins
        i = 0
        j = 314
        while (i >=0 and j >0):
            infile = np.fromfile(input_file_name,dtype=fun.T_ASIC_TEMP_EVENT,count=count,offset=((fun.T_ASIC_TEMP_EVENT.itemsize)*i*count))
            found_tx,found_ic=fun.find_tx_asic(infile)
            tdc_array += fun.hist_tdc(infile,found_tx,found_ic)
            i +=1
            if (infile.size != count):
                j = -200
        fun.find_tdc(tdc_array,found_tx,found_ic,tdc_filename)
        print('tdc calibration file written correctly!')"""

def pedestal_and_tdc(input_file_name,count,pedestals_filename,tdc_filename):
    i, j = 0, 314
    pedestal_array=np.zeros((18,12,64,99)) #tx,asic,channels,bins
    tdc_array=np.zeros((18,12,64,1024)) #tx,asic,channels,bins
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

def crystal_map(input_file_name,count,crystal_map_filename, pedestals_filename, showmaps):
    if os.path.isfile(crystal_map_filename):
        print('LUT does already exist!')
    else:
        with open('pedestals.json') as json_file0:
            data0 = json.load(json_file0)
        floodmap_array = np.zeros((18,12,200,200))
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

def main_timestamp(input_file_name,count, crystal_map_filename,pedestals_filename,tdc_filename,file_coincidenze, CW): #file_energy_calibration,resolution_filename
    lista = []
    with open(pedestals_filename) as json_file0:
        data0 = json.load(json_file0)
    with open(tdc_filename) as json_file1:
        data1 = json.load(json_file1)
    with open(file_coincidenze,'wb+') as f:
        data2 = json.loads(zlib.decompress(open(crystal_map_filename,'rb').read()).decode('ascii'))
        i, j = 0, 314
        #arr_calibrazione_energetica = np.zeros((18,12,113,200))     #array dove storo le informazioni di calibrazione energetica
        arr_risoluzione_energetica = np.zeros((18,12,113,200))

        while(i>=0 and j>0):
            print(i)
            infile = np.fromfile(input_file_name,dtype=fun.T_ASIC_TEMP_EVENT,count=int(count),offset=((fun.T_ASIC_TEMP_EVENT.itemsize)*i*int(count)))
            found_tx,found_ic=fun.find_tx_asic(infile)

            coincidences,arr_energy = fun.compute_events(infile,found_tx,found_ic,data0,data1,data2, CW)
            print('numero di coincidenze', int(np.shape(coincidences)[0]/2),'-------->', np.shape(coincidences)[0]/(int(count)/2))
            lista.append(np.shape(coincidences)[0]/(int(count)/2))
            arr_risoluzione_energetica += arr_energy
            coincidences.tofile(f)
            i +=1
            if (infile.size != int(count)):
                j = -200

        lista = np.asarray(lista)
        plt.figure()
        plt.plot(lista,'.')
        plt.xlabel('step [1 step = {} ev.]'.format(int(count)))
        plt.show()

        """
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
        print('infile.size',infile.size)
        i +=1
        if (infile.size != int(count)):
            j = -200
            """
    fun.find_energy_resolution(arr_risoluzione_energetica, True, False)
    #fun.find_energy_resolution(arr_risoluzione_energetica, False, True)
    #fun.find_energy_resolution(arr_risoluzione_energetica, False, False)

    """if os.path.isfile(resolution_filename):
        print('energy_resolution file already exist!')
    else:
        print('adesso trovo la risoluzione_energetica')
        fun.find_energy_resolution(arr_risoluzione_energetica, found_tx, found_ic, found_pixel, resolution_filename)
    """
    return file_coincidenze, arr_risoluzione_energetica


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program takes in input a raw file and gives in output a coincidences file')
    parser.add_argument('-f','--input_file',help='input file to process')
    parser.add_argument('-c','--count_events', help=' size of subfile to process',default=1_500_000)
    parser.add_argument('-cw','--coincidences_window', help = 'lenght of coincidences window in ns', default = 10)
    parser.add_argument('-p','--pedestals_filename',help='json file that stores pedestals value',default='pedestals.json')
    parser.add_argument('-tdc','--tdc_calibration_filename',help='json file that stores tdc calibration values',default='tdc_calibration.json')
    parser.add_argument('-lut','--crystal_map_filename',help='json file that stores LUT',default='crystals.zjson')
    parser.add_argument('-showmaps','--show_floodmap_LUT',help='Show floodmaps for each ASIC and LUT',default=False)
    parser.add_argument('-calib','--energy_calibration', help = 'json file that stores energy calibration values', default = 'en_calibration.json')
    parser.add_argument('-o','--outfile', help = 'outup file containing coincidences events', default = 'coincidences.dat.dec')
    parser.add_argument('-ris','--resolution_filename', help = 'file containing energy resolution values', default = 'energy_resolution.json')
    parser.add_argument('-CTR','--coincidence_time_resolution', help='compute and show histogram of time differences', default = True)
    parser.add_argument('-spectrum', '--plot_pixel_spectra', help='this is a list where the first element is True/False and the remains stand for [[TX], [ASIC]]', default = [False,[12,13],[8,9,10]])
    parser.add_argument('-ew','--energy_window', help='this is a list where the first element is True/False and the remains stand for [energy_min, energy_max]', default = [False, 350,650])
    parser.add_argument('-CTR_c','--coincidence_time_resolution_forASIC',help = 'compute and show histogram of time differences between given sector [TRUE/FALSE, TX1, ASIC1, TX2, ASIC2]', default = [False, 12,10,13,10])
    parser.add_argument('-CTR_all','--coincidence_time_resolution_for_ALL_ASIC',help = 'compute and show histogram of time differences between given sectors [TRUE/FALSE, [TX],[ASIC]]', default = [False, [12,13],[8,9,10]])


    args=parser.parse_args()
    #pedestals(args.input_file,args.count_events,args.pedestals_filename)
    #tdc(args.input_file,args.count_events,args.tdc_calibration_filename)
    pedestal_and_tdc(args.input_file,args.count_events,args.pedestals_filename,args.tdc_calibration_filename)
    crystal_map(args.input_file,args.count_events,args.crystal_map_filename,args.pedestals_filename, args.show_floodmap_LUT)
    coincidences_file,spectrum_array = main_timestamp(args.input_file,args.count_events,args.crystal_map_filename,args.pedestals_filename,args.tdc_calibration_filename, args.outfile, args.coincidences_window) #args.energy_calibration, args.resolution_filename,
    if args.coincidence_time_resolution:
        fun.find_CTR(coincidences_file,args.count_events, args.energy_window)
    if args.plot_pixel_spectra[0]:
        fun.plot_spectrum(spectrum_array, args.plot_pixel_spectra[1],args.plot_pixel_spectra[2])
    if args.coincidence_time_resolution_forASIC[0]:
        fun.find_CTR_perASIC(coincidences_file,args.count_events, args.energy_window,args.coincidence_time_resolution_forASIC)
    if args.coincidence_time_resolution_for_ALL_ASIC[0]:
        fun.find_CTR_perASIC_ALL(coincidences_file,args.count_events, args.energy_window,args.coincidence_time_resolution_for_ALL_ASIC[1],args.coincidence_time_resolution_for_ALL_ASIC[2])