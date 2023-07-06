import matplotlib.pyplot as plt
import os, sys, json, zlib
import numpy as np
import logging
from pylab import subplots, hist, plot, Circle, cm
from skimage.feature import blob_log
#from skimage.morphology import watershed
from skimage.segmentation import watershed
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import norm

from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import cProfile

import concurrent.futures
import threading

import line_profiler
profile = line_profiler.LineProfiler()

logging.basicConfig(level=logging.INFO)

N_TX = 18
N_ASIC = 12
PEDESTAL_BINS = 100
TDC_BINS = 1024
CORRECTION_TDC_OFFSET = 30
T_ASIC_CHANNELS = 64
TDC_CALIBRATED_BITS = 14
TOP_BITS = 16
COARSE_BITS = 10
TDC_UNIT_NS = 25 / 16384 #ns
FLOODMAP_SIZE = 200
ENERGY_RANGE = [0,2000]
ENERGY_RANGE_CALIBRATED = [0,1000]
ENERGY_WINDOW=[350,650]
N_BINS = 200
N_BINS_CTR = 120

T_ASIC_TEMP_EVENT = np.dtype([
    ('tx_id'     , np.uint8),
    ('asic_id'   , np.uint8),
    ('extra_bit' , np.uint8), #dice se top cresce o decresce
    ('top'       , np.uint16), #contatore tempi lunghi fa 0-n n-0
    ('global'    , np.uint32), #contatore tempi ancora + lunghi, +1 a reset top
    ('evt_id'    , np.uint32), #contatore eventi
    ('trg_charge', '%du1'%T_ASIC_CHANNELS), #array 64 el, trigger in carica su ogni canale
    ('energy'    , '%di2'%T_ASIC_CHANNELS), #energia ogni canale
    ('coarse'    , '%di2'%T_ASIC_CHANNELS), #tempo preciso
    ('fine'      , '%di2'%T_ASIC_CHANNELS), #divisore di clock- tempo +preciso
  ])

T_SINGLE_EVENT = np.dtype([
    ('tx_id'     , np.uint8),
    ('asic_id'   , np.uint8),
    ('energy'    , '%di2'%T_ASIC_CHANNELS), #energia ogni canale
    ('timestamp' , '%di8'%T_ASIC_CHANNELS), #tempi calibrati
  ])

T_SINGLE_EVENT_TIMESTAMPED = np.dtype([
    ('tx_id'     , np.uint8),
    ('asic_id'   , np.uint8),
    ('energy'    , '%di2'%T_ASIC_CHANNELS), #energia ogni canale
    ('timestamp' , np.int64), #tempo calibrato
  ])

T_SINGLE_EVENT_PIXELATED = np.dtype([
    ('tx_id'     , np.uint8),
    ('asic_id'   , np.uint8),
    ('pixel_id'  , np.uint8),
    ('energy'    , np.float32),
    ('timestamp' , np.int64), #tempo calibrato
  ])

@profile
def find_tx_asic(infile):
    """
    This function find tx and asic in the acquisition file

    Parameters
    ----------
    infile : .dat.dec
             name of the acquisition file

    Returns
    -------
    found_tx : list
               identification numbers of the TX found in the acquisition file
    found_ic : list
               identification numbers of the ASIC found in the acquisition file
    """
    found_tx = list(map(int,np.unique(infile['tx_id'])))
    found_ic = list(map(int,np.unique(infile['asic_id'])))
    assert(np.in1d(found_tx,np.arange(N_TX)).all())
    assert(np.in1d(found_ic,np.arange(N_ASIC)).all())
    logging.info('eventi trovati : {}'.format(infile.size))
    logging.info('TX trovate     : {}'.format(found_tx))
    logging.info('ASIC trovati   : {}'.format(found_ic))
    return found_tx, found_ic


@profile
def hist_pedestals(infile,found_tx,found_ic):
    """
    Function that returns an array with energy values for every pixel.

    Parameters
    ----------
    infile : .dat.dec
             name of the acquisition file
    found_tx : list
               tx found in the acquisition file
    found_ic : list
               asic found in the acquisition file

    Returns
    -------
    arr : numpy.array
          array with energy values for every pixel
    """
    arr=np.zeros((N_TX,N_ASIC,T_ASIC_CHANNELS,PEDESTAL_BINS-1)) #tx,asic,channels,bins
    for sel_tx_i,sel_tx in enumerate(found_tx):
        tx_ids=infile['tx_id']==sel_tx
        for sel_ic_i,sel_ic in enumerate(found_ic):
            a = infile[np.logical_and(tx_ids,infile['asic_id']==sel_ic)]
            charge=a['energy']
            for i in range(0,T_ASIC_CHANNELS): #loop sui canali
                n,b=np.histogram(charge[:,i],bins=np.linspace(0,PEDESTAL_BINS,PEDESTAL_BINS))
                arr[sel_tx,sel_ic,i,:]=n
    return arr

@profile
def hist_tdc(infile,found_tx,found_ic):
    """
    Function that returns an array with fine time values for every pixel.

    Parameters
    ----------
    infile : .dat.dec
             name of the acquisition file
    found_tx : list
               tx found in the acquisition file
    found_ic : list
               asic found in the acquisition file
    Returns
    -------
    arr : numpy.array
          array with fine time values for every pixel
    """
    arr=np.zeros((N_TX,N_ASIC,T_ASIC_CHANNELS,TDC_BINS)) #tx,asic,channels,bins
    for sel_tx_i,sel_tx in enumerate(found_tx):
        tx_ids=infile['tx_id']==sel_tx
        for sel_ic_i,sel_ic in enumerate(found_ic):
            a = infile[np.logical_and(tx_ids,infile['asic_id']==sel_ic)]
            fine=a['fine']
            for i in range(0,T_ASIC_CHANNELS): #loop sui canali
                n,b=np.histogram(fine[:,i],bins=np.arange(4.5,TDC_BINS+0.5))
                arr[sel_tx,sel_ic,i,5:]=n
    return arr

@profile
def find_pedestals(arr,found_tx,found_ic,pedestals_filename):
    """
    Function that calculate pedestal values for every pixel and write the results in a json file.

    Parameters
    ----------
    arr : numpy.array
          array with energy values for every pixel ('hist_pedestals' function output)
    found_tx : list
               tx found in the acquisition file
    found_ic : list
               asic found in the acquisition file
    pedestals_filename : .json
                         output file containing pedestal values
    Returns
    -------
    None : None
    """
    pedestals_dict={}
    #found_tx=np.nonzero(np.unique(arr[:,0]))[0]
    #found_ic=np.nonzero(np.unique(arr[:,1]))[0]
    for sel_tx_i,sel_tx in enumerate(found_tx):
        pedestals_dict['TX_{}'.format(sel_tx)]={}
        for sel_ic_i,sel_ic in enumerate(found_ic):
            pedestals_dict['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]=pedestals = [0]*(T_ASIC_CHANNELS)
            for i in range(0,T_ASIC_CHANNELS):
                max=np.where(arr[sel_tx,sel_ic,i,:]==arr[sel_tx,sel_ic,i,:].max())
                pedestals[i]=int(max[0][0]) +1

    json.dump(pedestals_dict,open(pedestals_filename,'w'),indent=4)

@profile
def find_tdc_range_to_correct(finetime,xdata,pixel):
    """
    Function that find the range of wrong fine time's values and correct them with the mean.

    Parameters
    ----------
    finetime : numpy.array
        array with fine time values for every pixel ('hist_tdc' function output)
    xdata : numpy.array
            x values of the fine time histogram (values from 0 to 1024)
    pixel : int
        pixel_id
    Returns
    -------
    range_correct : list
                    range's extremes where apply the correction
    mean : int
           value for the correction
    """
    DIFF = False
    GRAD = False
    f= np.diff(finetime)
    if DIFF:
        plt.plot(f)
    g = abs(np.gradient(finetime))
    if GRAD:
        plt.plot(g)
    peaks = xdata[g>10*g.mean()]
    if len(peaks) == 0:
        range_correct = [0]
        mean = 0
    else:
        values = xdata[int(peaks[0])-CORRECTION_TDC_OFFSET : int(peaks[0])+CORRECTION_TDC_OFFSET]
        mean = (finetime[int(peaks[0])-CORRECTION_TDC_OFFSET] + finetime[int(peaks[0])+CORRECTION_TDC_OFFSET])/2
        range_correct = [values[0],values[-1]]
    return range_correct, mean

@profile
def find_tdc(arr,found_tx,found_ic,tdc_filename):
    """
    Function that calculate and apply the calibration of fine time values for every pixel and write a json file with the results.

    Parameters
    ----------
    arr : numpy.array
          array with fine time values for every pixel ('hist_tdc' function output)
    found_tx : list
               tx found in the acquisition file
    found_ic : list
               asic found in the acquisition file
    tdc_filename : .json
                   output file containing calibrated fine time values
    Returns
    -------
    None : None
    """
    tdc_dict={}
    #found_tx=np.nonzero(np.unique(arr[:,0]))[0]
    #found_ic=np.nonzero(np.unique(arr[:,1]))[0]
    for sel_tx_i,sel_tx in enumerate(found_tx):
        tdc_dict['TX_{}'.format(sel_tx)]={}
        for sel_ic_i,sel_ic in enumerate(found_ic):
            t_cal = np.zeros((T_ASIC_CHANNELS,TDC_BINS),dtype=np.int16)-1
            count = 0
            for i in range(0,T_ASIC_CHANNELS):
                xdata = np.arange(4.5,TDC_BINS+0.5)
                xdata = 0.5*(xdata[1:]-xdata[:-1])
                try:
                    range_corrected, mean = find_tdc_range_to_correct(arr[sel_tx,sel_ic,i,:],xdata,i)
                    if mean != 0:
                        h[range_corrected[0]:range_corrected[1]] = mean
                except:
                    pass

                events = arr[sel_tx,sel_ic,i,990:]

                N_ideal_bin= sum(arr[sel_tx,sel_ic,i,:])/(TDC_BINS-5)
                calibration = ( 2**(TDC_CALIBRATED_BITS-10) * arr[sel_tx,sel_ic,i,5:]/N_ideal_bin)
                t_cal[i,5:] = ( calibration.cumsum() * (TDC_BINS/(TDC_BINS-5)) *  ( (2**TDC_CALIBRATED_BITS-1) /2**TDC_CALIBRATED_BITS) ).astype(np.int16)
                tdc_dict['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)] = t_cal.tolist()

    json.dump(tdc_dict,open(tdc_filename,'w'),indent=4)

@profile
def floodmap(infile, found_tx, found_ic, data0):
    """
    This function finds floodmap values for every asic.

    Parameters
    ----------
    infile : .dat.dec
            name of the acquisition file
    found_tx : list
               tx found in the acquisition file
    found_ic : list
               asic found in the acquisition file
    data0: .json
           json file containing pedestals values
    Returns
    -------
    floodmap_array : numpy.array
                     floodmap values
    event_x : int
              coordinate of the centroid along the x axes
    event_y : int
              coordinate of the centroid along the y axes
    """
    floodmap_array = np.zeros((N_TX,N_ASIC,FLOODMAP_SIZE,FLOODMAP_SIZE))
    for sel_tx_i,sel_tx in enumerate(found_tx):
        tx_ids=infile['tx_id']==sel_tx
        for sel_ic_i,sel_ic in enumerate(found_ic):
            charge = infile['energy'][np.logical_and(tx_ids,infile['asic_id']==sel_ic)] -  data0['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]
            charge = np.ma.masked_less_equal(charge,0).reshape(-1,int(np.sqrt(T_ASIC_CHANNELS)),int(np.sqrt(T_ASIC_CHANNELS)))
            charge_sums = charge.sum(axis=1).sum(axis=1)
            event_x = charge.sum(axis=1).dot(np.arange(charge[0,0].size))/charge_sums
            event_y = charge.sum(axis=2).dot(np.arange(charge[0,:,0].size))/charge_sums
            fmap_val, _, _ = np.histogram2d(event_y, event_x,bins=(np.linspace(0,(int(np.sqrt(T_ASIC_CHANNELS))-1),N_BINS+1),np.linspace(0,(int(np.sqrt(T_ASIC_CHANNELS))-1),N_BINS+1)))
            floodmap_array[sel_tx,sel_ic,:,:] = fmap_val

    return floodmap_array,event_x,event_y

#NOTA: image sarebbe fmap_val, img_max sarebbe blobs_log e labels sarebbe lut
@profile
def generate_maps(fmap_val,lista_cry):
    """
    This function generates the maps using watershed segmentation algorithm.

    Parameters
    ----------
    fmap_val : numpy.array
               floodmap values
    lista_cry : list
                detector's crystals for the map
    Returns
    -------
    fmap_val : numpy.array
               floodmap values
    blobs_log : numpy.array
                blobs values obtained by using scipy.ndimage.maximum_filter
    lut : numpy.array
          look-up table, results of the watershed segmentation algorithm
    """
    #AGGIUNGERE DOCUMENTAZIONE

    #applicazione dell'algoritmo di watershed

    #(cx,cy) = floodmap(infile, found_tx, found_ic, data0) #coord dei centroidi
    #print(cx, cy)
    #image,xe,ye = np.histogram2d(cx,cy, bins=[np.linspace(0,7,201),np.linspace(0,7,201)])

    #distance = ndi.distance_transform_edt(image)
    blobs_log = ndi.maximum_filter(fmap_val, size=20, mode='constant')
    #print(img_max)
    #local_maxi = peak_local_max(image, indices=False, min_distance=10)
    #print(local_maxi)
    #markers = ndi.label(local_maxi)[0]
    #print(markers)
    lut = watershed(-blobs_log,T_ASIC_CHANNELS,watershed_line=False)
    #print(np.unique(lut,return_counts=True))
    #print(len(labels))
    #print(lut)

    return fmap_val, blobs_log, lut

@profile
def find_LUT(floodmap_array,found_tx, found_ic, showmaps, crystals_filename):
    """
    Function that writes in a json file the informations about the floodmap, the blobs and the look-up table for every tx and asic.

    Parameters
    ----------
    floodmap_array : numpy.array
                     floodmap values
    found_tx : list
               tx found in the acquisition file
    found_ic : list
               asic found in the acquisition file
    showmaps : boolean variable
               when it's set to TRUE the function 'plot_maps' is activated
    crystals_filename : .json
                        output file containing floodmap, blobs and lut values for every tx and asic
    Returns
    -------
    None : None
    """
    crystalmap_dict = {}
    for sel_tx_i,sel_tx in enumerate(found_tx):
        crystalmap_dict['TX_{}'.format(sel_tx)] = {}
        for sel_ic_i, sel_ic in enumerate(found_ic):
            CRY_BOT_W, CRY_BOT_H, CRY_TOP_W, CRY_TOP_H = [int(np.sqrt(T_ASIC_CHANNELS)),int(np.sqrt(T_ASIC_CHANNELS)),0,0]

            #print(np.shape(floodmap_array[sel_tx, sel_ic,:,:]))
            fmap_val, blobs_log, lut = generate_maps(floodmap_array[sel_tx, sel_ic,:,:],[CRY_BOT_W, CRY_BOT_H, CRY_TOP_W, CRY_TOP_H])
            if showmaps:
                plot_maps(fmap_val,blobs_log,lut,sel_tx,sel_ic)
            crystalmap_dict['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)] = {
            'fmap_val': fmap_val.tolist(),
            'blobs_log': blobs_log.tolist(),
            'lut': lut.tolist()
            }
    open(crystals_filename,'wb').write(zlib.compress(json.dumps(crystalmap_dict,indent=4).encode('ascii')))

@profile
def plot_maps(fmap_val,blobs_log,lut,sel_tx,sel_ic):
    """
    Function that plots and shows the floodmap, the blobs and the look-up table for a selected tx and asic.

    Parameters
    ----------
    fmap_val : numpy.array
               floodmap values
    blobs_log : numpy.array
                blobs values obtained by using scipy.ndimage.maximum_filter
    lut : numpy.array
          look-up table, results of the watershed segmentation algorithm
    sel_tx : int
             number of the selected tx in which it plots the maps
    sel_ic : int
             number of the selected asic in which it plots the maps
    Returns
    -------
    None : None
    """
    sel_tx = sel_tx if 'TX' in str(sel_tx) else f'TX{sel_tx:02d}'
    sel_ic = sel_ic if 'ASIC' in str(sel_ic) else f'ASIC{sel_ic:02d}'
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True, squeeze = False)
    fig.suptitle(f'{sel_tx} - {sel_ic}')

    ax = axes.ravel()

    ax[0].imshow(fmap_val, cmap=plt.cm.gray)
    ax[0].set_title('Floodmap')
    ax[1].imshow(-blobs_log, cmap=plt.cm.gray)
    ax[1].set_title('Img_max')
    ax[2].imshow(lut, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for l in ax:
        l.set_axis_off()

    fig.tight_layout()
    plt.show()



@profile
def get_pixel(charge,lut):
    """
    Function that returns an array with the pixel_id for every event.

    Parameters
    ----------
    charge : numpy.array
             array containing energy values
    lut : numpy.array or list
          values of the look-up table (segmentation of the floodmap)
    Returns
    -------
    pixel_ids : numpy.array
                array with identification number for the pixels
    """
    charge = np.ma.masked_less_equal(charge,0).reshape(-1,int(np.sqrt(T_ASIC_CHANNELS)),int(np.sqrt(T_ASIC_CHANNELS)))
    charge_sums = charge.sum(axis=1).sum(axis=1)
    event_x = charge.sum(axis=1).dot(np.arange(charge[0,0].size))/charge_sums
    event_y = charge.sum(axis=2).dot(np.arange(charge[0,:,0].size))/charge_sums
    pixel_ids = lut[np.clip((event_y/(7)*lut[1].size).astype('i2'),0,FLOODMAP_SIZE-2),np.clip((event_x/(7)*lut[0].size).astype('i2'),0,FLOODMAP_SIZE-2)]
    return pixel_ids

@profile
def calculate_timestamps(infile,sel_tx,sel_ic, tdc_calibration):
    """
    Function that calculate timestamps for every channel.

    Parameters
    ----------
    infile : .dat.dec
        name of the acquisition file
    sel_tx : int
             number of the selected tx in which it calculates the timestamp of the detected event
    sel_ic : int
             number of the selected asic in which it calculates the timestamp of the detected event
    tdc_calibration : .json
                      json file containing calibrated fine time values
    Returns
    -------
    channel_timestamps : numpy.array
                         array with timestamp values for every channel
    """
    tdc_calibration['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)] = np.array(tdc_calibration['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)])
    t_cal = tdc_calibration['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]
    tfine_raw = np.ma.masked_less_equal(infile['fine'],0)
    tfine = np.ma.masked_where(tfine_raw.mask,np.take_along_axis(t_cal.T,tfine_raw,axis=0)) # apply calibration... non dovrebbe gia essere in ps?
    tglobal = infile['global'].astype(np.int64)
    ttop    = infile['top'].astype(np.int64)
    tcoarse = infile['coarse'].astype(np.int64)

    # numpy.where(tcoarse>900,axis=1)
    tbc = np.logical_and((ttop & 1) != infile['extra_bit'],tcoarse.max(axis=1) > 900)
    ttop[tbc] -= 1
    tfine  = tfine.astype(np.int64)
    channel_timestamps = (
            np.broadcast_to((tglobal << (TDC_CALIBRATED_BITS + COARSE_BITS + TOP_BITS)),    (T_ASIC_CHANNELS,tglobal.size)).T +
            np.broadcast_to((ttop    << (TDC_CALIBRATED_BITS + COARSE_BITS)),                  (T_ASIC_CHANNELS,ttop.size)).T +
            (tcoarse << TDC_CALIBRATED_BITS) -
            tfine
        )
    channel_timestamps[tbc] -= 2 << (COARSE_BITS + TDC_CALIBRATED_BITS)
    #print(np.shape(channel_timestamps))
    #print ((channel_timestamps[0,:])* (25*(10**(-9))/16384))
    return channel_timestamps

@profile
def choose_timestamp(AVERAGE, timestamp, charge):
    """
    This function chooses the timestamp associated with every event and returns an array with the results.
    The choice can be made by considering the position of the greatest charge deposition in the asic or by calculating the average.

    Parameters
    ----------
    AVERAGE : boolean variable
              when it's set to TRUE the choice of the timestamp is made by calculating the average of the charge deposition in the asic
    timestamp : numpy.array
                array with timestamp values (output of the 'calculate_timestamps' function)
    charge : numpy.array
             array with energy values
    Returns
    -------
    arr : numpy.array
          array with timestamp values for every event
    """
    th = 0 #somma*0.40
    cc = np.ma.masked_less_equal(charge,th)
    #faccio due funzioni per gestire 0 e 4.
    #se incontro 1,2 e 3 butto l'evento e lo segnalo
    if AVERAGE:
        arr = np.average(timestamp, weights = cc, axis = 1)
    else:
        arr = np.take_along_axis(timestamp,np.ma.masked_less_equal(charge,0).argmax(axis=1).reshape(-1,1),axis=1)[:,0]
        #print(np.shape(arr))
        #print ((arr)* (25*(10**(-9))/16384))
        #plt.figure()
        #plt.title('Events in TX {} ASIC {}'.format(sel_tx,sel_ic))
        #plt.xlabel('time [s]')
        #occ, edges,_= plt.hist((arr)*(25*(10**(-9))/16384), bins=np.arange(0,20,1))
        #print(np.arange(0,int(acquisition_time),1))
        #plt.show()
    return arr

@profile
def get_coincidences_all(infile,CW):
    """
    This function returns an array with the coincidences.

    Parameters
    ----------
    infile : .dat
             input file containing timestamped events
    CW : int
         Coincidence Window in which to search for coincidences
    Returns
    -------
    coinc : numpy.array
            coincidences' array
    """
    coinc_window = CW/TDC_UNIT_NS
    #infile3 = infile[np.logical_and(infile['energy']<650 ,infile['energy']>400 )]
    #infile2 = infile1[bottommask]
    #print('sto selezionando solo gli eventi del bottom')

    coincident_first  = np.nonzero(np.diff(infile['timestamp']) < coinc_window)[0]
    coincident_second = coincident_first + 1
    etero_coincidence = infile['tx_id'][coincident_first] != infile['tx_id'][coincident_second]

    c1 = infile[coincident_first][etero_coincidence]
    c2 = infile[coincident_second][etero_coincidence]

    coincidences = np.vstack([c1,c2])
    coincidences.sort(order='tx_id',axis=0)
    coinc = np.dstack([coincidences[0], coincidences[1]]).flatten().ravel()

    return coinc

@profile
def hist_energy(array_temporaneo,pixel_ids,sel_tx,sel_ic):
    """
    Function that returns an array with energy values for every pixel_id.

    Parameters
    ----------
    array_temporaneo : numpy.array
                       array containing information about timestamped events
    pixel_ids : int
                pixel_id for every event
    sel_tx : int
             number of the selected tx in which it calculates the energy value of the detected event
    sel_ic : int
             number of the selected asic in which it calculates the energy of the detected event
    Returns
    -------
    arr : numpy.array
          array with energy values for every pixel
    """
    arr = np.zeros((N_TX,N_ASIC,T_ASIC_CHANNELS+1,N_BINS))
    found_pixel = list(map(int,np.unique(pixel_ids)))

    for sel_pixel_i, sel_pixel in enumerate(found_pixel):
        mask_pixel = array_temporaneo['pixel_id'] == sel_pixel
        n,b = np.histogram(array_temporaneo['energy'][mask_pixel],bins=np.linspace(ENERGY_RANGE[0],ENERGY_RANGE[1],N_BINS+1))

        arr[sel_tx,sel_ic,sel_pixel,:] = n
    return arr

@profile
def compute_events(infile,found_tx,found_ic,pedestal,tdc_calibration,crystals,CW):
    """
    Function that computes the events and returns coincidences and energy array.

    Parameters
    ----------
    infile: .dat.dec
           name of the acquisition file
    found_tx : int
               tx found in the acquisition file
    found_ic : int
               asic found in the acquisition file
    pedestal: .json
             json file containing pedestals values
    tdc_calibration: .json
                    json file containing calibrated fine time values
    crystals: .zjson
             compressed json file containing the map of the events in the asic (made by 'find_LUT' function)
    CW: int
        Coincidence Window in which to search for coincidences
    Returns
    -------
    coincidences : numpy.array
                   output of the 'get_coincidences_all' function
    arr : numpy.array
          output of the 'hist_energy' function applied to the timestamped events
    """
    arr = np.zeros((N_TX,N_ASIC,T_ASIC_CHANNELS+1,N_BINS))
    with open ('temp.dat', 'wb+') as f:
        for sel_tx_i, sel_tx in enumerate(found_tx):
            tx_ids = infile['tx_id']==sel_tx
            for sel_ic_i, sel_ic in enumerate(found_ic):
                tx_ic_ids = np.logical_and(tx_ids,infile['asic_id']==sel_ic)
                array_t = calculate_timestamps(infile[tx_ic_ids],sel_tx,sel_ic, tdc_calibration)
                array_e = (infile[tx_ic_ids]['energy']  - pedestal['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)])
                array_2t = choose_timestamp(False, array_t, array_e)
                lut = np.array(crystals['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]['lut'])
                pixel_ids = get_pixel(array_e,lut)
                array_charge = np.ma.masked_less_equal(array_e,0).sum(axis=1)
                array_temporaneo = np.zeros(dtype = T_SINGLE_EVENT_PIXELATED, shape =array_charge.size)
                array_temporaneo['tx_id'] = sel_tx
                array_temporaneo['asic_id'] = sel_ic
                array_temporaneo['timestamp'] = array_2t
                array_temporaneo['energy'] = array_charge
                array_temporaneo['pixel_id'] = pixel_ids
                array_temporaneo.tofile(f)
                arr += hist_energy(array_temporaneo,pixel_ids,sel_tx,sel_ic)


    fn = np.fromfile('temp.dat',dtype=T_SINGLE_EVENT_PIXELATED)
    fn.sort(order = 'timestamp')
    #plt.figure()
    #plt.title('Events in TX {} ASIC {}'.format(sel_tx,sel_ic))
    #plt.xlabel('time [s]')
    #occ, edges,_= plt.hist((fn['timestamp'])*(25*(10**(-9))/16384), bins=np.arange(0,20,1))
    #print(np.arange(0,int(acquisition_time),1))
    #plt.show()
    #print('-------------')
    #print('                        ',fn['timestamp'][0],'  ',fn['timestamp'][-1])
    #print('-------------')
    coincidences = get_coincidences_all(fn,CW)
    coinc_diff = coincidences['timestamp'][::2]-coincidences['timestamp'][1::2]# coincidences['timestamp'][1,:] - coincidences['timestamp'][0,:]
    #ydata_hist_temp,_ = np.histogram(coinc_diff * TDC_UNIT_NS, bins = 120)


    #print(coincidences[0]) #sarebbe al prima colonna di eventi in coincidenza che hanno preso asic 12
    #print('shape coinc che hanno preso 12',len(coincidences[0]), 'dsgfsd',len(coincidences[1]))
    #print(coincidences[:,0]) #prendo la prima coincidenza
    del fn
    return coincidences, arr

@profile
def find_peak(n,window_size=19,order=7,threshold=0.5):
    """
    Function that uses the 'savgol_filter' form scipy.signal to find peaks in the energy histogram.
    Parameters
    ----------
    n : numpy.array
        data to be filtered
    Returns
    -------
    filtered : numpy.array
               filtered data
    rightmost_peak : numpy.array
                     rightmost filtered data
    """
    filtered = signal.savgol_filter(n, window_size, order) #trova picchi
    filtered = filtered / filtered.mean() * n.mean() #normalizzali
    filtered[filtered < threshold*n.max()] = 0 #elimina i picchi bassi
    rightmost_peak = (np.nonzero(np.diff(-np.sign(np.diff(filtered))).clip(0))[0]+2)[-1]
    return filtered, rightmost_peak # take the rightmost peak

@profile
def find_energy_calibration(found_tx, found_ic,arr_calibrazione_energetica,efficiencies_filename):
    """
    This function calculates energy calibration coefficients (or efficiencies) for every pixel and write them in a json file.

    Parameters
    ----------
    found_tx : int
               tx found in the acquisition file
    found_ic : int
               asic found in the acquisition file
    arr_calibrazione_energetica : numpy.array
                                  array containing energy values used to find the peak for the calibration
    efficiencies_filename : .json
                            json file containing energy calibration coefficients (or efficiencies)
    Returns
    -------
    None : None
    """
    # arr_risoluzione_energetica = np.zeros((N_TX,12,113,200))
    efficiencies_dict = {}
    for sel_tx_i, sel_tx in enumerate(found_tx):
        efficiencies_dict['TX_{}'.format(sel_tx)] = {}

        found_pixel = np.linspace(0,T_ASIC_CHANNELS-1,T_ASIC_CHANNELS)

        for sel_ic_i, sel_ic in enumerate(found_ic):
            eff = []
            e = np.linspace(ENERGY_RANGE[0],ENERGY_RANGE[1],N_BINS+1)

            for sel_pixel_i, sel_pixel in enumerate (found_pixel):
                #print(np.shape(arr_calibrazione_energetica))

                filtered, peak = find_peak(arr_calibrazione_energetica[sel_tx,sel_ic,int(sel_pixel),:])
                #print(np.shape(arr_calibrazione_energetica[sel_tx,sel_ic,int(sel_pixel),:]))
                ec = 0.5*(e[1:]+e[:-1])
                eff.append(511./ec[peak-1]) #coefficiente di calibrazione
            efficiencies_dict['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)] = eff
    json.dump(efficiencies_dict,open(efficiencies_filename,'w'),indent=4)

@profile
def apply_energy_calibration(infile,energy_cal, found_tx, found_ic):
    """
    This function applies energy calibration to all the coincidences found previously.

    Parameters
    ----------
    infile : .dat
             input file containing all the coincidences found previously
    energy_cal : .json
                 json file containing energy calibration coefficients
    found_tx : int
               tx found in the input file
    found_ic : int
               asic found in the input file
    Returns
    -------
    arr : numpy.array
          array containing all the calibrated energy values for every pixel
    """
    arr = np.zeros((N_TX,N_ASIC,T_ASIC_CHANNELS+1,N_BINS))
    for sel_tx_i, sel_tx in enumerate(found_tx):
            tx_ids = infile['tx_id']==sel_tx
            for sel_ic_i, sel_ic in enumerate(found_ic):
                tx_ic_ids = np.logical_and(tx_ids,infile['asic_id']==sel_ic)
                #print(len(tx_ic_ids))
                for pixid, efficiency in enumerate(energy_cal['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]):
                    sel = np.logical_and(tx_ic_ids,infile['pixel_id']==pixid)
                    infile['energy'][sel] = infile['energy'][sel] * efficiency
                    n,b = np.histogram(infile['energy'][sel],bins=np.linspace(ENERGY_RANGE_CALIBRATED[0],ENERGY_RANGE_CALIBRATED[1],N_BINS+1))
                    arr[sel_tx,sel_ic,pixid,:] += n
    return arr

@profile
def f(x, C, mu, sigma):
    """
    f is the Gaussian function
    """
    return C*norm.pdf(x, mu, sigma)

@profile
def fit_function(spectrum, xdata): #senza Klein-Nishina (il fit lo fa con 'f')
    """
    This function is a Gaussian fit and it calculates the energy resolution and its uncertainty.
    Parameters
    ----------
    spectrum : numpy.array
               array containing the energy spectrum
    xdata : numpy.array
            array containing the x values of the energy spectrum
    Returns
    -------
    risoluzione_en : float
                     energy resolution
    u_R : float
          energy resolution's uncertainty
    """
    xdata_mask = np.logical_and(xdata<ENERGY_WINDOW[1] , xdata>ENERGY_WINDOW[0])
    popt, pcov = curve_fit(f, xdata[xdata_mask], spectrum[xdata_mask], p0=[150000,400,30])
    """
    plt.plot(xdata, f(xdata, *popt), label='fit')
    plt.show()
    """


    risoluzione_en = 2.35*(popt[2])/popt[1]

    u_R= 2.35*np.sqrt(((np.sqrt(np.diag(pcov)[2]))/popt[1])**2 + (popt[2]*(np.sqrt(np.diag(pcov)[1]))/(popt[1]**2))**2)

    return risoluzione_en, u_R

@profile
def find_CTR(file_coincidenze,count,lista_energia):
    """
    This function finds the Coincidence Time Resolution of the detector.

    Parameters
    ----------
    file_coincidenze :
                      file containing coincidences
    count : int
            size of subfile to process
    lista_energia : list
                    energy window: list where the first element is True/False and the remains stand for [energy_min, energy_max]
    Returns
    -------
    None : None
    """
    i, j = 0, 314
    array_CTR=np.zeros((N_BINS_CTR))
    while(i>=0 and j>0):
        infile = np.fromfile(file_coincidenze,dtype=T_SINGLE_EVENT_PIXELATED,count=int(count),offset=((T_SINGLE_EVENT_PIXELATED.itemsize)*i*int(count)))
        if lista_energia[0]:
            mask1 = np.logical_and(infile['energy'][::2]>lista_energia[1], infile['energy'][::2]<lista_energia[2])
            mask2 =np.logical_and(infile['energy'][1::2]>lista_energia[1], infile['energy'][1::2]<lista_energia[2])
            mask3 = np.logical_or(mask1,mask2)
            mask4 = np.zeros(mask1.shape[0] + mask2.shape[0])
            mask4[::2] = mask3
            mask4[1::2] = mask3
            mask5 = mask4>0
            infile = infile[mask5]

        coinc_diff = infile['timestamp'][::2]-infile['timestamp'][1::2]# coincidences['timestamp'][1,:] - coincidences['timestamp'][0,:]
        n,b = np.histogram(coinc_diff * TDC_UNIT_NS, bins = N_BINS_CTR)
        array_CTR += n
        i+=1
        if (infile.size != int(count)):
            j = -200

    mask = np.logical_and(np.linspace(-10,10,len(array_CTR)) < 5, np.linspace(-10,10,len(array_CTR))> -5)
    popt, pcov = curve_fit(f,np.linspace(-10,10,len(array_CTR))[mask], array_CTR[mask], p0=[15000,0,1])
    plt.figure()
    plt.title('CTR = ({:.3f} +/- {:.3f}) ns'.format(2.35 * popt[2], 2.35*np.sqrt(np.diag(pcov)[2])))
    plt.xlabel('differenza temporale [ns]')
    plt.plot(np.linspace(-10,10,len(array_CTR)), array_CTR)
    plt.show()

@profile
def plot_spectrum(array_energia,tx,asic):
    """
    Function that plots the energy spectrum for every pixel.

    Parameters
    ----------
    array_energia : numpy.array
                    array with the energy values
    tx : int
         tx identification number
    asic : int
           asic identification number
    Returns
    -------
    None : None
    """
    x =np.linspace(ENERGY_RANGE_CALIBRATED[0],ENERGY_RANGE_CALIBRATED[1],N_BINS)
    for sel_tx in tx:
        for sel_ic in asic:
            for pixel in range(0,T_ASIC_CHANNELS):
                plt.figure()
                plt.title('Spettro energetico TX = {}, ASIC = {}, Pixel = {}'.format(sel_tx,sel_ic,pixel))
                plt.xlabel('energy [keV]')
                plt.plot(x,array_energia[sel_tx,sel_ic,pixel,:])
                plt.show()

@profile
def find_energy_resolution(arr_risoluzione_energetica):
    """
    This function plots the total energy spectrum and calculates the energy resolution of the detector.

    Parameters
    ----------
    arr_risoluzione_energetica : numpy.array
                                 array containing all the energy values
    Returns
    -------
    None : None
    """
    x = np.linspace(ENERGY_RANGE_CALIBRATED[0],ENERGY_RANGE_CALIBRATED[1],N_BINS)

    y = arr_risoluzione_energetica.sum(axis = 0)
    y1 = y.sum(axis = 0)
    y2 = y1.sum(axis = 0)
    en, uen = fit_function(y2,x) #o fit_function_spectrum se voglio mettere la sezione d'urto di Klein-Nishina

    plt.figure()
    plt.title('total energy spectrum \n energy resolution = {:.3f} +/- {:.3f}'.format(en,uen))
    plt.plot(x,y2)
    #plt.xlabel('energy [keV]')
    plt.show()
