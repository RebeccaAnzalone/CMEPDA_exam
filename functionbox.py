import matplotlib.pyplot as plt
import os, sys, json, zlib
import numpy as np
from pylab import subplots, hist, plot, Circle, cm
from skimage.feature import blob_log
from skimage.morphology import watershed
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import norm

from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

T_ASIC_CHANNELS = 64
TDC_CALIBRATED_BITS = 14
TOP_BITS = 16
COARSE_BITS = 10
TDC_UNIT_NS = 25 / 16384 #ns

T_ASIC_TEMP_EVENT = np.dtype([
    ('tx_id'     , np.uint8),
    ('asic_id'   , np.uint8),
    ('extra_bit' , np.uint8), #dice se top cresce o decresce
    ('top'       , np.uint16), #contatore tempi lunghi fa 0-n n-0
    ('global'    , np.uint32), #contatore tempi ancola + lunghi, +1 a reset top
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

def find_tx_asic(infile):
    """
    This function find tx and asic in the acquisition file

    Parameters:
        - infile = name of the acquisition file
    """
    found_tx = list(map(int,np.unique(infile['tx_id'])))
    found_ic = list(map(int,np.unique(infile['asic_id'])))
    assert(np.in1d(found_tx,np.arange(18)).all())
    assert(np.in1d(found_ic,np.arange(12)).all())
    print('eventi trovati : {}'.format(infile.size))
    print('TX trovate     : {}'.format(found_tx))
    print('ASIC trovati   : {}'.format(found_ic))
    return found_tx, found_ic

def hist_pedestals(infile,found_tx,found_ic):
    arr=np.zeros((18,12,64,99)) #tx,asic,channels,bins
    for sel_tx_i,sel_tx in enumerate(found_tx):
        tx_ids=infile['tx_id']==sel_tx
        for sel_ic_i,sel_ic in enumerate(found_ic):
            a = infile[np.logical_and(tx_ids,infile['asic_id']==sel_ic)]
            charge=a['energy']
            for i in range(0,64): #loop sui canali
                n,b=np.histogram(charge[:,i],bins=np.linspace(0,100,100))
                arr[sel_tx,sel_ic,i,:]=n
    return arr

def hist_tdc(infile,found_tx,found_ic):
    arr=np.zeros((18,12,64,1024)) #tx,asic,channels,bins
    for sel_tx_i,sel_tx in enumerate(found_tx):
        tx_ids=infile['tx_id']==sel_tx
        for sel_ic_i,sel_ic in enumerate(found_ic):
            a = infile[np.logical_and(tx_ids,infile['asic_id']==sel_ic)]
            fine=a['fine']
            for i in range(0,64): #loop sui canali
                n,b=np.histogram(fine[:,i],bins=np.arange(4.5,1024.5))
                arr[sel_tx,sel_ic,i,5:]=n
    return arr

def find_pedestals(arr,found_tx,found_ic,pedestals_filename):
    pedestals_dict={}
    #found_tx=np.nonzero(np.unique(arr[:,0]))[0]
    #found_ic=np.nonzero(np.unique(arr[:,1]))[0]
    for sel_tx_i,sel_tx in enumerate(found_tx):
        pedestals_dict['TX_{}'.format(sel_tx)]={}
        for sel_ic_i,sel_ic in enumerate(found_ic):
            pedestals_dict['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]=pedestals = [0]*(64)
            for i in range(0,64):
                max=np.where(arr[sel_tx,sel_ic,i,:]==arr[sel_tx,sel_ic,i,:].max())
                pedestals[i]=int(max[0][0]) +1

    json.dump(pedestals_dict,open(pedestals_filename,'w'),indent=4)

def find_tdc_range_to_correct(c,xdata,i):
    DIFF = False
    GRAD = False
    f= np.diff(c)
    if DIFF:
        plt.plot(f)
    g = abs(np.gradient(c))
    if GRAD:
        plt.plot(g)
    peaks = xdata[g>10*g.mean()]
    if len(peaks) == 0:
        range_correct = [0]
        mean = 0
    else:
        values = xdata[int(peaks[0])-30 : int(peaks[0])+30]
        mean = (c[int(peaks[0])-30] + c[int(peaks[0])+30])/2
        range_correct = [values[0],values[-1]]
    return range_correct, mean

def find_tdc(arr,found_tx,found_ic,tdc_filename):
    tdc_dict={}
    #found_tx=np.nonzero(np.unique(arr[:,0]))[0]
    #found_ic=np.nonzero(np.unique(arr[:,1]))[0]
    for sel_tx_i,sel_tx in enumerate(found_tx):
        tdc_dict['TX_{}'.format(sel_tx)]={}
        for sel_ic_i,sel_ic in enumerate(found_ic):
            t_cal = np.zeros((64,1024),dtype=np.int16)-1
            count = 0
            for i in range(0,64):
                xdata = np.arange(4.5,1024.5)
                xdata = 0.5*(xdata[1:]-xdata[:-1])
                try:
                    range_corrected, mean = find_tdc_range_to_correct(arr[sel_tx,sel_ic,i,:],xdata,i)
                    if mean != 0:
                        h[range_corrected[0]:range_corrected[1]] = mean
                except:
                    pass
                events = arr[sel_tx,sel_ic,i,990:]
                if events.all() > 0.00025:
                    count +=1

                N_ideal_bin= sum(arr[sel_tx,sel_ic,i,:])/1019
                calibration = ( 2**(TDC_CALIBRATED_BITS-10) * arr[sel_tx,sel_ic,i,5:]/N_ideal_bin)
                t_cal[i,5:] = ( calibration.cumsum() * (1024/1019) *  ( (2**TDC_CALIBRATED_BITS-1) /2**TDC_CALIBRATED_BITS) ).astype(np.int16)
                tdc_dict['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)] = t_cal.tolist()
            print('tx = {}, asic = {}, wrong channels = {} !'.format(sel_tx,sel_ic,count))
            if count >5:
                print('---Error!---')
    json.dump(tdc_dict,open(tdc_filename,'w'),indent=4)

def floodmap(infile, found_tx, found_ic, data0):
    floodmap_array = np.zeros((18,12,200,200))
    for sel_tx_i,sel_tx in enumerate(found_tx):
        tx_ids=infile['tx_id']==sel_tx
        for sel_ic_i,sel_ic in enumerate(found_ic):
            charge = infile['energy'][np.logical_and(tx_ids,infile['asic_id']==sel_ic)] -  data0['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]
            charge = np.ma.masked_less_equal(charge,0).reshape(-1,8,8)
            charge_sums = charge.sum(axis=1).sum(axis=1)
            event_x = charge.sum(axis=1).dot(np.arange(charge[0,0].size))/charge_sums
            event_y = charge.sum(axis=2).dot(np.arange(charge[0,:,0].size))/charge_sums
            fmap_val, _, _ = np.histogram2d(event_y, event_x,bins=(np.linspace(0,7,201),np.linspace(0,7,201)))
            floodmap_array[sel_tx,sel_ic,:,:] = fmap_val
    return floodmap_array,event_x,event_y
"""
def generate_maps(fmap_val,lista_cry):
    #fmap_val, _, _ = np.histogram2d(event_y, event_x,bins=(np.linspace(0,7,201),np.linspace(0,7,201)))
    blobs_log = blob_log(fmap_val, min_sigma=5, max_sigma=6)
    blobs_log = blobs_log[blobs_log[:,0].argsort()]
    bot_w, bot_h, top_w, top_h = lista_cry #[CRY_BOT_W, CRY_BOT_H, CRY_TOP_W, CRY_TOP_H]
    n_crystals = bot_w*bot_h+top_w*top_h
    n_blobs = blobs_log[:,0].size
    assert(blobs_log[:,0].size==(n_crystals)), f'Dovrebbero essere {n_crystals} blobs invece di {n_blobs}!'
    for i in range(bot_h+top_h):
        row_bot_beg = i*(bot_w+top_w)
        row_top_beg = i*(bot_w+top_w)+bot_w
        row_bot = blobs_log[row_bot_beg:row_top_beg]
        row_top = blobs_log[row_top_beg:row_top_beg+top_w]
        blobs_log[row_bot_beg:row_top_beg] = row_bot[row_bot[:,1].argsort()]
        blobs_log[row_top_beg:row_top_beg+top_w] = row_top[row_top[:,1].argsort()]
    b = blobs_log[(np.arange(bot_w*bot_h).reshape(bot_w,bot_h).T + np.arange(bot_h)*top_w).T,:]
    t = blobs_log[(np.arange(top_w*top_h).reshape(top_w,top_h).T + np.arange(1,top_h+1)*bot_w).T,:]
    blobs_log = np.concatenate([b.reshape(-1,3),t.reshape(-1,3)])
    lut = np.zeros_like(fmap_val,dtype=np.int8)
    markers = lut.copy()
    markers[blobs_log[:,0].astype(np.uint8),blobs_log[:,1].astype(np.uint8)] = np.arange(blobs_log[:,0].size) + 1
    lut = watershed(lut,markers) - 1
    return fmap_val, blobs_log, lut
"""
#NOTA: image sarebbe fmap_val, img_max sarebbe blobs_log e labels sarebbe lut
def generate_maps(fmap_val,lista_cry):
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
    lut = watershed(-blobs_log, 64,watershed_line=False)
    #print(np.unique(lut,return_counts=True))
    #print(len(labels))
    #print(lut)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True, squeeze = False)
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
    return fmap_val, blobs_log, lut


def find_LUT(floodmap_array,found_tx, found_ic, showmaps, crystals_filename):
    crystalmap_dict = {}
    for sel_tx_i,sel_tx in enumerate(found_tx):
        crystalmap_dict['TX_{}'.format(sel_tx)] = {}
        for sel_ic_i, sel_ic in enumerate(found_ic):
            #if sel_tx == 12 :
            CRY_BOT_W, CRY_BOT_H, CRY_TOP_W, CRY_TOP_H = [8,8,0,0]
            #else:
            #CRY_BOT_W, CRY_BOT_H, CRY_TOP_W, CRY_TOP_H = [8,8,7,7]
            #if ((sel_tx == 12 and sel_ic == 11)):
            #    continue
            #else:
            print(np.shape(floodmap_array[sel_tx, sel_ic,:,:]))
            fmap_val, blobs_log, lut = generate_maps(floodmap_array[sel_tx, sel_ic,:,:],[CRY_BOT_W, CRY_BOT_H, CRY_TOP_W, CRY_TOP_H])
            if showmaps:
                plot_maps(fmap_val,blobs_log,lut,sel_tx,sel_ic)
            crystalmap_dict['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)] = {
            'fmap_val': fmap_val.tolist(),
            'blobs_log': blobs_log.tolist(),
            'lut': lut.tolist()
            }
    open(crystals_filename,'wb').write(zlib.compress(json.dumps(crystalmap_dict,indent=4).encode('ascii')))

def plot_maps(fmap_val,blobs_log,lut,sel_tx,sel_ic):
    sel_tx = sel_tx if 'TX' in str(sel_tx) else f'TX{sel_tx:02d}'
    sel_ic = sel_ic if 'ASIC' in str(sel_ic) else f'ASIC{sel_ic:02d}'
    fig, cols = subplots(1,3,figsize=(20,5))
    fig.suptitle(f'{sel_tx} - {sel_ic}')
    if fmap_val is not None:
        cols[0].set_title('Flood map')
        cols[0].imshow(fmap_val, interpolation='nearest', origin='lower',vmax=np.percentile(fmap_val,99.9))
    if blobs_log is not None:
        cols[1].imshow(fmap_val, interpolation='nearest', origin='lower',vmax=np.percentile(fmap_val,99.9))
        cols[1].set_title('Blobs')
        for y, x, s in blobs_log:
            cols[1].add_patch(Circle((x, y), s * np.sqrt(2), color='y', lw=1, fill=False))
    if lut is not None:
        cols[2].set_title('Look-up table')
        cols[2].imshow(lut, interpolation='nearest', origin='lower', cmap=cm.Paired)
        for i, (y, x, s) in enumerate(blobs_log):
            cols[2].add_patch(Circle((x, y), s * np.sqrt(2), color='k', lw=1, fill=False))
            cols[2].text(x,y,i,ha='center',va='center',fontsize=8)

def get_pixel(charge,lut):
    charge = np.ma.masked_less_equal(charge,0).reshape(-1,8,8)
    charge_sums = charge.sum(axis=1).sum(axis=1)
    event_x = charge.sum(axis=1).dot(np.arange(charge[0,0].size))/charge_sums
    event_y = charge.sum(axis=2).dot(np.arange(charge[0,:,0].size))/charge_sums
    pixel_ids = lut[np.clip((event_y/(7)*lut[1].size).astype('i2'),0,198),np.clip((event_x/(7)*lut[0].size).astype('i2'),0,198)]
    bot_idx = pixel_ids<64
    bot_idx, top_idx = np.nonzero(bot_idx)[0], np.nonzero(np.logical_not(bot_idx))[0]
    return pixel_ids, bot_idx, top_idx

def calculate_timestamps(a,sel_tx,sel_ic, tdc_calibration):
    tdc_calibration['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)] = np.array(tdc_calibration['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)])
    t_cal = tdc_calibration['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]
    tfine_raw = np.ma.masked_less_equal(a['fine'],0)
    tfine = np.ma.masked_where(tfine_raw.mask,np.take_along_axis(t_cal.T,tfine_raw,axis=0)) # apply calibration... non dovrebbe gia essere in ps?
    tglobal = a['global'].astype(np.int64)
    ttop    = a['top'].astype(np.int64)
    tcoarse = a['coarse'].astype(np.int64)

    # numpy.where(tcoarse>900,axis=1)
    tbc = np.logical_and((ttop & 1) != a['extra_bit'],tcoarse.max(axis=1) > 900)
    ttop[tbc] -= 1
    tfine  = tfine.astype(np.int64)
    channel_timestamps = (
            np.broadcast_to((tglobal << (TDC_CALIBRATED_BITS + COARSE_BITS + TOP_BITS)),    (64,tglobal.size)).T +
            np.broadcast_to((ttop    << (TDC_CALIBRATED_BITS + COARSE_BITS)),                  (64,ttop.size)).T +
            (tcoarse << TDC_CALIBRATED_BITS) -
            tfine
        )
    channel_timestamps[tbc] -= 2 << (COARSE_BITS + TDC_CALIBRATED_BITS)
    #print(np.shape(channel_timestamps))
    #print ((channel_timestamps[0,:])* (25*(10**(-9))/16384))
    return channel_timestamps

def choose_timestamp(AVERAGE, timestamp, charge):
    th = 0 #somma*0.40
    cc = np.ma.masked_less_equal(charge,th)
    #faccio due funzioni per gestire 0 e 4.
    #se incontro 1,2 e 3 butto l'evento e lo segnalo
    if AVERAGE:
        arr = np.average(timestamp, weights = cc, axis = 1)
    else:
        arr = np.take_along_axis(timestamp,np.ma.masked_less_equal(charge,0).argmax(axis=1).reshape(-1,1),axis=1)[:,0]
        print(np.shape(arr))
        print ((arr)* (25*(10**(-9))/16384))
        #plt.figure()
        #plt.title('Events in TX {} ASIC {}'.format(sel_tx,sel_ic))
        #plt.xlabel('time [s]')
        #occ, edges,_= plt.hist((arr)*(25*(10**(-9))/16384), bins=np.arange(0,20,1))
        #print(np.arange(0,int(acquisition_time),1))
        #plt.show()
    return arr

def get_coincidences_all(infile,CW):
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
    aa = np.dstack([coincidences[0], coincidences[1]]).flatten().ravel()

    return aa

def hist_energy(array_temporaneo,pixel_ids,sel_tx,sel_ic):
    arr = np.zeros((18,12,113,200))
    found_pixel = list(map(int,np.unique(pixel_ids)))

    for sel_pixel_i, sel_pixel in enumerate(found_pixel):
        mask_pixel = array_temporaneo['pixel_id'] == sel_pixel
        n,b = np.histogram(array_temporaneo['energy'][mask_pixel],bins=np.linspace(0,2000,201))

        arr[sel_tx,sel_ic,sel_pixel,:] = n

        """
        plt.figure()
        plt.title('Spettro del pixel {}, tx {}, asic {}'.format(sel_pixel, sel_tx, sel_ic))
        plt.plot(np.linspace(0,2000,200),arr[sel_tx,sel_ic,sel_pixel,:]) #spettro di un pixel calibrato
        plt.show()
        """
    return arr

def compute_events(infile,found_tx,found_ic,pedestal,tdc_calibration, crystals,CW,i):
    arr = np.zeros((18,12,113,200))
    #with open ('temp.dat', 'wb+') as f:
    for sel_tx_i, sel_tx in enumerate(found_tx):
        tx_ids = infile['tx_id']==sel_tx
        for sel_ic_i, sel_ic in enumerate(found_ic):
                with open ('temp_{}_{}_{}.dat'.format(sel_tx,sel_ic,i), 'wb+') as f:
                    tx_ic_ids = np.logical_and(tx_ids,infile['asic_id']==sel_ic)
                    array_t = calculate_timestamps(infile[tx_ic_ids],sel_tx,sel_ic, tdc_calibration)
                    array_e = (infile[tx_ic_ids]['energy']  - pedestal['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)])
                    array_2t = choose_timestamp(False, array_t, array_e)
                    lut = np.array(crystals['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]['lut'])
                    pixel_ids, bot_idx, top_idx = get_pixel(array_e,lut)
                    array_charge = np.ma.masked_less_equal(array_e,0).sum(axis=1)
                    array_temporaneo = np.zeros(dtype = T_SINGLE_EVENT_PIXELATED, shape =array_charge.size)
                    array_temporaneo['tx_id'] = sel_tx
                    array_temporaneo['asic_id'] = sel_ic
                    array_temporaneo['timestamp'] = array_2t
                    array_temporaneo['energy'] = array_charge
                    array_temporaneo['pixel_id'] = pixel_ids
                    array_temporaneo.tofile(f)
                    arr += hist_energy(array_temporaneo,pixel_ids,sel_tx,sel_ic)


    fn = np.fromfile('temp_{}_{}_{}.dat'.format(sel_tx,sel_ic,i),dtype=T_SINGLE_EVENT_PIXELATED)
    fn.sort(order = 'timestamp')
    plt.figure()
    #plt.title('Events in TX {} ASIC {}'.format(sel_tx,sel_ic))
    plt.xlabel('time [s]')
    occ, edges,_= plt.hist((fn['timestamp'])*(25*(10**(-9))/16384), bins=np.arange(0,20,1))
    #print(np.arange(0,int(acquisition_time),1))
    plt.show()
    print('-------------')
    print('                        ',fn['timestamp'][0],'  ',fn['timestamp'][-1])
    print('-------------')
    coincidences = get_coincidences_all(fn,CW)
    coinc_diff = coincidences['timestamp'][::2]-coincidences['timestamp'][1::2]# coincidences['timestamp'][1,:] - coincidences['timestamp'][0,:]
    #ydata_hist_temp,_ = np.histogram(coinc_diff * TDC_UNIT_NS, bins = 120)


    #print(coincidences[0]) #sarebbe al prima colonna di eventi in coincidenza che hanno preso asic 12
    #print('shape coinc che hanno preso 12',len(coincidences[0]), 'dsgfsd',len(coincidences[1]))
    #print(coincidences[:,0]) #prendo la prima coincidenza
    #del fn
    return coincidences, arr

def find_peak(n,window_size=19,order=7,threshold=0.5):
    filtered = signal.savgol_filter(n, window_size, order) #trova picchi
    filtered = filtered / filtered.mean() * n.mean() #normalizzali
    filtered[filtered < threshold*n.max()] = 0 #elimina i picchi bassi
    return filtered, (np.nonzero(np.diff(-np.sign(np.diff(filtered))).clip(0))[0]+2)[-1] # take the rightmost peak

def find_energy_calibration(found_tx, found_ic,arr_calibrazione_energetica,efficiencies_filename):
    # arr_risoluzione_energetica = np.zeros((18,12,113,200))
    efficiencies_dict = {}
    for sel_tx_i, sel_tx in enumerate(found_tx):
        efficiencies_dict['TX_{}'.format(sel_tx)] = {}
        #if (sel_tx == 12):
            #found_pixel = np.linspace(0,112,113)
        #else:
        found_pixel = np.linspace(0,63,64)

        for sel_ic_i, sel_ic in enumerate(found_ic):
            eff = []
            e = np.linspace(0,2000,210)

            for sel_pixel_i, sel_pixel in enumerate (found_pixel):
                print(np.shape(arr_calibrazione_energetica))

                filtered, peak = find_peak(arr_calibrazione_energetica[sel_tx,sel_ic,int(sel_pixel),:])
                #print(np.shape(arr_calibrazione_energetica[sel_tx,sel_ic,int(sel_pixel),:]))
                ec = 0.5*(e[1:]+e[:-1])
                eff.append(511./ec[peak-1]) #coefficiente di calibrazione
            efficiencies_dict['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)] = eff
    json.dump(efficiencies_dict,open(efficiencies_filename,'w'),indent=4)

def apply_energy_calibration(infile,energy_cal, found_tx, found_ic):
    arr = np.zeros((18,12,113,200))
    for sel_tx_i, sel_tx in enumerate(found_tx):
            tx_ids = infile['tx_id']==sel_tx
            for sel_ic_i, sel_ic in enumerate(found_ic):
                if (sel_tx == 12 and sel_ic == 11):
                    continue
                else:
                    tx_ic_ids = np.logical_and(tx_ids,infile['asic_id']==sel_ic)
                    #print(len(tx_ic_ids))
                    for pixid, efficiency in enumerate(energy_cal['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)]):
                        sel = np.logical_and(tx_ic_ids,infile['pixel_id']==pixid)
                        infile['energy'][sel] = infile['energy'][sel] * efficiency
                        n,b = np.histogram(infile['energy'][sel],bins=np.linspace(0,1000,201))
                        arr[sel_tx,sel_ic,pixid,:] += n
    return arr

def f(x, C, mu, sigma):
    return C*norm.pdf(x, mu, sigma)

def find_energy_resolution_perpixel(array_risoluzione_energetica, found_tx, found_ic, found_pixel, resolution_filename):
    resolution_dict = {}
    #arr = np.zeros((18,12,113,1))
    for sel_tx_i, sel_tx in enumerate(found_tx):
        resolution_dict['TX_{}'.format(sel_tx)] = {}
        for sel_ic_i, sel_ic in enumerate(found_ic):
            if ((sel_tx == 12 and sel_ic == 11)):
                continue
            else:
                ris_list = []
                for pixel_id_i, sel_pixel in enumerate(found_pixel):
                    spectrum = array_risoluzione_energetica[sel_tx,sel_ic, sel_pixel,:]
                    xdata = np.linspace(0,1000,200)
                    """plt.figure()
                    plt.title('spettro')
                    plt.plot(xdata,spectrum)
                    plt.show()"""
                    ris, uris = fit_function(spectrum, xdata)
                    ris_list.append(ris)
                    #arr[sel_tx,sel_ic, sel_pixel,:] = ris
                resolution_dict['TX_{}'.format(sel_tx)]['ASIC_{}'.format(sel_ic)] = ris_list
    json.dump(resolution_dict,open (resolution_filename,'w'),indent=4)

def kn(x):
    E=511   #energia del gamma incidente
    ct = 1 - (511 / E) * (E/x -1)
    r = 2.81*10**(-13)    #cm
    a =r*r*((1 + ct**2) / 2)
    b = 1/((1 + E *(1 - ct))**2)
    c =1 + ((E**2*(1-ct)**2) / ( (1 + ct**2)*(1 + E*(1-ct)) ) )
    sigma=a*b+c
    return sigma

def fitfinale(x, C, mu, sigma, A):
    return C * norm.pdf(x, mu, sigma) + A*kn(x)

def fit_function(spectrum, xdata):
    xdata_mask = np.logical_and(xdata<650 , xdata>350)
    popt, pcov = curve_fit(f, xdata[xdata_mask], spectrum[xdata_mask], p0=[150000,400,30])
    plt.plot(xdata, f(xdata, *popt), label='fit')
    plt.show()


    risoluzione_en = 2.35*(popt[2])/popt[1]

    u_R= 2.35*np.sqrt(((np.sqrt(np.diag(pcov)[2]))/popt[1])**2 + (popt[2]*(np.sqrt(np.diag(pcov)[1]))/(popt[1]**2))**2)

    return risoluzione_en, u_R

def fit_function_spectrum(spectrum, xdata):
    xdata_mask = np.logical_and(xdata<650 , xdata>350)
    popt, pcov = curve_fit(fitfinale, xdata[xdata_mask], spectrum[xdata_mask], p0=[15000,511,50, 100])
    risoluzione_en = 2.35*(popt[2])/popt[1]

    u_R= 2.35*np.sqrt(((np.sqrt(np.diag(pcov)[2]))/popt[1])**2 + (popt[2]*(np.sqrt(np.diag(pcov)[1]))/(popt[1]**2))**2)

    return risoluzione_en, u_R

def find_CTR(file_coincidenze, count,lista_energia):
    i, j = 0, 314
    array_CTR=np.zeros((120))
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
        n,b = np.histogram(coinc_diff * TDC_UNIT_NS, bins = 120)
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

def find_CTR_perASIC(file_coincidenze, count,lista_energia, settori):
    i, j = 0, 314
    array_CTR=np.zeros((120))
    while(i>=0 and j>0):
        infile = np.fromfile(file_coincidenze,dtype=T_SINGLE_EVENT_PIXELATED,count=int(count),offset=((T_SINGLE_EVENT_PIXELATED.itemsize)*i*int(count)))
        mask1 = np.logical_and(infile['tx_id'] == settori[1], infile['asic_id'] ==  settori[2])
        mask2 = np.logical_and(infile['tx_id'] == settori[3], infile['asic_id'] == settori[4])
        mask3 = np.logical_and(mask1[::2],mask2[1::2])
        mask4 = np.logical_and(mask2[::2],mask1[1::2])
        mask5 = np.logical_or(mask3,mask4)
        mask6 = np.zeros(mask3.shape[0] + mask4.shape[0])
        mask6[::2] = mask5
        mask6[1::2] = mask5
        mask7 = mask6 > 0
        infile = infile[mask7]

        coinc_diff = infile['timestamp'][::2]-infile['timestamp'][1::2]# coincidences['timestamp'][1,:] - coincidences['timestamp'][0,:]
        n,b = np.histogram(coinc_diff * TDC_UNIT_NS, bins = 120)
        array_CTR += n
        i+=1
        if (infile.size != int(count)):
            j = -200

    mask = np.logical_and(np.linspace(-10,10,len(array_CTR)) < 5, np.linspace(-10,10,len(array_CTR))> -5)
    popt, pcov = curve_fit(f,np.linspace(-10,10,len(array_CTR))[mask], array_CTR[mask], p0=[15000,0,1])
    plt.figure()
    plt.title('TX = ({},{}), ASIC = ({},{}).\n CTR  = ({:.3f} +/- {:.3f}) ns'.format(settori[1],settori[3],settori[2],settori[4],2.35 * popt[2], 2.35*np.sqrt(np.diag(pcov)[2])))
    plt.xlabel('differenza temporale [ns]')
    plt.plot(np.linspace(-10,10,len(array_CTR)), array_CTR)
    plt.show()

def find_CTR_perASIC_ALL(file_coincidenze,count,lista_energia, tx, asic):
    a = 0
    for sel_tx in tx:
        for sel_ic in asic:
            for sel_tx2 in tx:
                if (sel_tx == sel_tx2):
                    continue
                for sel_ic2 in asic:
                    find_CTR_perASIC(file_coincidenze, int(count),lista_energia, [a,sel_tx,sel_ic,sel_tx2,sel_ic2])

def plot_spectrum(array_energia,tx,asic):
    x =np.linspace(0,1000,200)
    for sel_tx in tx:
        for sel_ic in asic:
            #if sel_ic == 12:
            n_pixel = 64
            #else:
            #    n_pixel = 113
            for pixel in range(0,n_pixel):
                plt.figure()
                plt.title('Spettro energetico TX = {}, ASIC = {}, Pixel = {}'.format(sel_tx,sel_ic,pixel))
                plt.xlabel('energy [keV]')
                plt.plot(x,array_energia[sel_tx,sel_ic,pixel,:])
                plt.show()

def find_energy_resolution(arr_risoluzione_energetica, TOTAL, TRIMAGE):
        x = np.linspace(0,1000,200)
        if TOTAL:
            y = arr_risoluzione_energetica.sum(axis = 0)
            y1 = y.sum(axis = 0)
            y2 = y1.sum(axis = 0)
            en, uen = fit_function(y2,x)

            plt.figure()
            plt.title('total energy spectrum \n energy resolution = {:.3f} +/- {:.3f}'.format(en,uen))
            plt.plot(x,y2)
            #plt.xlabel('energy [keV]')
            plt.show()


        else:
            if TRIMAGE:
                y = arr_risoluzione_energetica[13,:,:,:]
                y1 = y.sum(axis = 0)
                y2 = y1.sum(axis=0)
                en, uen = fit_function_spectrum(y2,x)
                plt.figure()
                plt.title('total energy spectrum in TRIMAGE \n energy resolution = {:.3f} +/- {:.3f}'.format(en,uen))
                plt.plot(x,y2)
                plt.xlabel('energy [keV]')
                plt.show()
            else:
                y = arr_risoluzione_energetica[12,:,:,:]
                y1 = y.sum(axis = 0)
                y2 = y1.sum(axis=0)
                en, uen = fit_function_spectrum(y2,x)
                plt.figure()
                plt.title('total energy spectrum in SPECTRON \n energy resolution = {:.3f} +/- {:.3f}'.format(en,uen))
                plt.plot(x,y2)
                plt.xlabel('energy [keV]')
                plt.show()
