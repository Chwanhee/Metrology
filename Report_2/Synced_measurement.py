import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from tqdm import tqdm
import pickle
import time

import scipy.fftpack as fft

from swimAD2 import swimAD2 as ad2
import dwfconstants as dwfc

import threading

########## Begin function defining ##########

def find_times(ch1,ch2):
# Test by first bringing backwards. If fails, test forwards. 
    ''' **Warning**
    Ch1s seem to have an offset. Ch2s will thus have a better match.
    '''
    
    backward,forward = [],[]
    for i in range(1,150):
        score = abs((ch1[i:]-ch2[:-i])/(ch1[i:]+ch2[:-i]))
        backward.append(np.mean(score))

        score = abs((ch1[:-i]-ch2[i:])/(ch1[:-i]+ch2[i:]))
        forward.append(np.mean(score))

    score_back = np.min(backward)
    score_forw = np.min(forward)

    if score_back > score_forw:
        i = np.argmin(forward)
        direction = "f"
    else: 
        i = np.argmin(backward)
        direction = "b"
    return i, direction

def sync_series(t0,ch1,ch2,ch3,ref):
    '''
    "ref" is the duplicate channel used for reference
    '''
    i,direction = find_times(ch2,ref)
    i = i+1 if i==0 else i
    if direction=="b":
        return t0[:-i],ch1[:-i],ch2[:-i],ch3[i:]
    else: 
        return t0[i:],ch1[i:],ch2[i:],ch3[:-i]

def sampling(time):
    # Priority on oscilloscope detection rata
    size = 8192
    rate = size/time
    return rate

########## End function defining ##########

ad2.disconnect()            
zoroku = ad2.connect(0)
alice = ad2.connect(1)

# freq = np.linspace(1e0,5e2,250) # Low freq
freq = np.linspace(5e2,7.5e3,300) # High freq
amp = np.linspace(5e-2,2,100)

data = {}
for F in amp:
    data[F] = {}
    for f in freq:
        data[F][f] = []

def measure(device):
    for F in tqdm(amp,desc="amp",position=0,leave=False):
        for f in tqdm(freq,desc="freq",position=1,leave=False):
            # Sometimes, the z channel does not read. 
            if data[F][f]=="NA":
                continue

            duration = 10/f if 1/f<10 else 10
            rate = sampling(duration)   
            range = 20 if F>4.5 else 3

            ad2.config_oscilloscope(zoroku
                                    , range0=range, range1=range
                                    , sample_rate=rate)
            ad2.config_oscilloscope(alice
                                    , range0=range, range1=range
                                    , sample_rate=rate)

            ad2.config_wavegen(device, frequency=f, amplitude=F, signal_shape=dwfc.funcSine)
            ad2.start_wavegen(zoroku, channel=0)
            
            time.sleep(0.01)
            t0, ch1, ch2 = ad2.measure_oscilloscope(device)
            
            if len(ch2)==0:
                print("Failed measurement.")
                data[F][f] = "NA"
                continue

            ad2.stop_wavegen(zoroku, channel=0)
            ad2.reset_wavegen(zoroku, channel=0)
            
            # First list is zoroku, second is alice.
            data[F][f].append([t0, ch1, ch2]) 

def main():
    threads = []
    for devices in [zoroku, alice]:
        thread = threading.Thread(name=devices, target=measure, args=(devices,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

main()
name = "low " if max(freq)<500 else "high "
handle = open(f"measurement_{name}freq.pkl", 'wb')
pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

ad2.disconnect()