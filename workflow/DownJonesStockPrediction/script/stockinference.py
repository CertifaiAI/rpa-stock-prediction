'''
Created on Nov 27, 2019

@author: TUF
'''
import warnings
import copy
import os
import sys
import logging
logging.disable(logging.WARNING)
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import numpy as np
    import keras
    from keras.models import load_model
from datetime import datetime, timedelta
from _overlapped import NULL

# datein = "2019-11-26,2019-11-25,2019-11-22,2019-11-21,2019-11-20,2019-11-19,2019-11-18,2019-11-15,2019-11-14,2019-11-13,2019-11-12,2019-11-11,2019-11-08,2019-11-07,2019-11-06,2019-11-05,2019-11-04,2019-11-01,2019-10-31,2019-10-30,2019-10-29,2019-10-28,2019-10-25,2019-10-24,2019-10-23,2019-10-22,2019-10-21,2019-10-18,2019-10-17,2019-10-16,2019-10-15,2019-10-14,2019-10-11,2019-10-10,2019-10-09,2019-10-08,2019-10-07,2019-10-04,2019-10-03,2019-10-02,2019-10-01,2019-09-30,2019-09-27,2019-09-26,2019-09-25,2019-09-24,2019-09-23,2019-09-20,2019-09-19,2019-09-18"
# openin = "28080.75,27917.7695,27831.2305,27820.2793,27879.5508,28079.7598,27993.2207,27843.5391,27757.1992,27622.0391,27701.5898,27580.6602,27686.1992,27590.1602,27502.7402,27500.2305,27402.0605,27142.9492,27188.3691,27110.7109,27061.0703,27040.3301,26789.6094,26893.9297,26835.2402,26850.4297,26852.6699,27004.4902,27032.3809,26972.3105,26811.1992,26766.4297,26694.1992,26317.3496,26308.2305,26276.5898,26502.3301,26271.6992,26039.0195,26425.8594,26962.5391,26852.3301,26987.2598,27004.1094,26866.7109,27034.0703,26851.4492,27102.1797,27186.0508,27075.3906"
# highin = "28146.0195,28068.6895,27898.4609,27828.3301,27897.2793,28090.2109,28040.9707,28004.8906,27800.7109,27806.4004,27770.8594,27714.3906,27694.9492,27774.6699,27526.0508,27560.3594,27517.5801,27347.4297,27188.3691,27204.3594,27165.9395,27167.8809,27015.3691,26931.7793,26896.8906,26946.6406,26852.6699,27018.25,27112.1602,27058.3398,27120.1094,26874.3301,27013.9707,26603.3105,26424.3105,26421.8105,26655.8398,26590.7402,26205.1992,26438.0391,27046.2109,26998.8594,27012.5391,27015.0703,27016.5605,27079.6797,27011.0703,27194.75,27272.1699,27161.9297"
# lowin = "28042.2109,27917.7695,27773.9805,27708.3398,27675.2793,27894.5195,27969.2402,27843.5391,27676.9707,27587.1992,27635.3203,27517.6699,27578.9707,27590.1602,27407.8105,27453.5508,27402.0605,27142.9492,26918.2891,26999.6406,27039.7598,27028.7109,26765.6797,26714.3398,26745,26782.6094,26747.6191,26770.1309,26970.2891,26943.2891,26811.1992,26749.1797,26694.1992,26314.5098,26249.75,26139.8008,26424.5391,26271.6992,25743.4609,25974.1191,26562.2207,26852.3301,26715.8203,26803.8398,26755.8594,26704.9609,26831.3398,26926.6797,27064.2109,26899.1504"
# closein = "28121.6797,28066.4707,27875.6191,27766.2891,27821.0898,27934.0195,28036.2207,28004.8906,27781.9609,27783.5898,27691.4902,27691.4902,27681.2402,27674.8008,27492.5605,27492.6309,27462.1094,27347.3594,27046.2305,27186.6895,27071.4609,27090.7207,26958.0605,26805.5293,26833.9492,26788.0996,26827.6406,26770.1992,27025.8809,27001.9805,27024.8008,26787.3594,26816.5898,26496.6699,26346.0098,26164.0391,26478.0195,26573.7207,26201.0391,26078.6191,26573.0391,26916.8301,26820.25,26891.1191,26970.7109,26807.7695,26949.9902,26935.0703,27094.7891,27147.0801"
# adjin = "28121.6797,28066.4707,27875.6191,27766.2891,27821.0898,27934.0195,28036.2207,28004.8906,27781.9609,27783.5898,27691.4902,27691.4902,27681.2402,27674.8008,27492.5605,27492.6309,27462.1094,27347.3594,27046.2305,27186.6895,27071.4609,27090.7207,26958.0605,26805.5293,26833.9492,26788.0996,26827.6406,26770.1992,27025.8809,27001.9805,27024.8008,26787.3594,26816.5898,26496.6699,26346.0098,26164.0391,26478.0195,26573.7207,26201.0391,26078.6191,26573.0391,26916.8301,26820.25,26891.1191,26970.7109,26807.7695,26949.9902,26935.0703,27094.7891,27147.0801"
# volumein = "324050000,248420000,214780000,232020000,258140000,245890000,252320000,283720000,303970000,278390000,213670000,202350000,221440000,259020000,237910000,286350000,273030000,270870000,270910000,231750000,269610000,290770000,274610000,253590000,247680000,265510000,241030000,288970000,222540000,214660000,245510000,178620000,282080000,217680000,190060000,244590000,195200000,221310000,241610000,309640000,260110000,222680000,217780000,229180000,237220000,301750000,204240000,497640000,212360000,212860000"
# averagein = "28094.1152,27993.2295,27836.2207,27768.33495,27786.2793,27992.3652,28005.10545,27924.21485,27738.8408,27696.7998,27703.08985,27616.03025,27636.95995,27682.41505,27466.93065,27506.9551,27459.8203,27245.18945,27053.3291,27102,27102.84965,27098.2959,26890.5244,26823.05955,26820.9453,26864.625,26800.1445,26894.19045,27041.22465,27000.81445,26965.6543,26811.7549,26854.08495,26458.91015,26337.03025,26280.80565,26540.18945,26431.2197,25974.33005,26206.0791,26804.2158,26925.59475,26864.1797,26909.45505,26886.20995,26892.3203,26921.20505,27060.71485,27168.1904,27030.54005"

datetemp = list(datein.split(sep=","))
opentemp = list(openin.split(sep=","))
hightemp = list(highin.split(sep=","))
lowtemp = list(lowin.split(sep=","))
closetemp = list(closein.split(sep=","))
adjtemp = list(adjin.split(sep=","))
volumetemp = list(volumein.split(sep=","))
averagetemp = list(averagein.split(sep=","))

open_minmax = [1243.709961, 26457.879882999998]
high_minmax = [1251.209961, 26523.459961000004]
low_minmax = [1235.530029, 26399.790284]
close_minmax = [1242.050049, 26449.440185]
adjclose_minmax =[1242.050049, 26449.440185]
volume_minmax = [2530000, 2188280000]
average_minmax = [1243.369995, 26459.719848999997]

for idx, val in enumerate(opentemp):
    datetemp[idx] = datetime.strptime(datetemp[idx], "%Y-%m-%d").date()
    opentemp[idx] = (float(opentemp[idx]) - open_minmax[0]) / (open_minmax[1] - open_minmax[0])
    hightemp[idx] = (float(hightemp[idx]) - high_minmax[0]) / (high_minmax[1] - high_minmax[0])
    lowtemp[idx] = (float(lowtemp[idx]) - low_minmax[0]) / (low_minmax[1] - low_minmax[0])
    closetemp[idx] = (float(closetemp[idx]) - close_minmax[0]) / (close_minmax[1] - close_minmax[0])
    adjtemp[idx] = (float(adjtemp[idx]) - adjclose_minmax[0]) / (adjclose_minmax[1] - adjclose_minmax[0])
    volumetemp[idx] = (float(volumetemp[idx]) - volume_minmax[0]) / (volume_minmax[1] - volume_minmax[0])
    averagetemp[idx] = (float(averagetemp[idx]) - average_minmax[0]) / (average_minmax[1] - average_minmax[0])

for j in range(3):
    tmr = NULL
    tmr = datetemp[0] + timedelta(days=1)
    print("original tmr: ", tmr)
    if tmr.weekday() < 5:
        datetemp.insert(0, tmr)
        print("Is Weekday: ", tmr)
        print("tmr.Weekday(): ", tmr.weekday())
    elif tmr.weekday() == 5:
        tmr = tmr + timedelta(days=2)
        datetemp.insert(0, tmr)
        print("Is Saturday: ", tmr)
        print("tmr.Weekday(): ", tmr.weekday())
    elif tmr.weekday() == 6:
        tmr = tmr + timedelta(days=1)
        datetemp.insert(0, tmr)
        datetemp.insert(0, tmr)
        print("Is Sunday: ", tmr)
        print("tmr.Weekday(): ", tmr.weekday())
print(len(datetemp))

for k in range(len(datetemp)):
    datetemp[k] = str(datetemp[k])
    
np_open = np.array(opentemp, dtype=float).reshape(50,1)
np_high = np.array(hightemp, dtype=float).reshape(50,1)
np_low = np.array(lowtemp, dtype=float).reshape(50,1)
np_close = np.array(closetemp, dtype=float).reshape(50,1)
np_adj = np.array(adjtemp, dtype=float).reshape(50,1)
np_volume = np.array(volumetemp, dtype=float).reshape(50,1)
np_average = np.array(averagetemp, dtype=float).reshape(50,1)

inference_np = np.concatenate((np_open, np_high, np_low, np_close, np_adj, np_volume, np_average), axis=1)
result_np = copy.copy(inference_np)

model = load_model(model_path)
 
for ii in range(3):
    output_ = NULL
    newout = NULL
    output_ = model.predict(inference_np.reshape(1, 50 , 7))
    avg = (float(output_[0][1]) + float(output_[0][2])) / 2.0
    newout = copy.copy(output_)
    newout = np.append(newout, avg).reshape(1,7)
    inference_np = np.delete(inference_np, -1, axis=0)
    inference_np = np.insert(inference_np, 0, newout, axis=0)
    result_np = np.insert(result_np, 0, newout, axis=0)

return_date = datetemp[::-1]    
return_open = []
return_high = []
return_low = []
return_close = []
return_adj = []
return_volume = []
return_avg = []

for kk in range(len(result_np)):
    return_open.append(float(result_np[kk][0]))
    return_high.append(float(result_np[kk][1]))
    return_low.append(float(result_np[kk][2]))
    return_close.append(float(result_np[kk][3]))
    return_adj.append(float(result_np[kk][4]))
    return_volume.append(float(result_np[kk][5]))
    return_avg.append(float(result_np[kk][6]))

return_open = return_open[::-1]
return_high = return_high[::-1]
return_low = return_low[::-1]
return_close = return_close[::-1]
return_adj = return_adj[::-1]
return_volume = return_volume[::-1]
return_avg = return_avg[::-1]  

pdate = ""
popen = ""
phigh = ""
plow = ""
pclose = ""
padj = ""
pvolume = ""
pavg = ""

for each in range(len(return_open)):
    return_open[each] =   (return_open[each] * (open_minmax[1] - open_minmax[0])) + open_minmax[0]
    return_high[each] =   (return_high[each] * (high_minmax[1] - high_minmax[0])) + high_minmax[0]
    return_low[each] =   (return_low[each] * (low_minmax[1] - low_minmax[0])) + low_minmax[0]
    return_close[each] =   (return_close[each] * (close_minmax[1] - close_minmax[0])) + close_minmax[0]
    return_adj[each] =   (return_adj[each] * (adjclose_minmax[1] - adjclose_minmax[0])) + adjclose_minmax[0]
    return_volume[each] =   (return_volume[each] * (volume_minmax[1] - volume_minmax[0])) + volume_minmax[0]
    return_avg[each] = (return_avg[each] * (average_minmax[1] - average_minmax[0])) + average_minmax[0]
    if each == len(return_open) -1 :
        pdate = pdate + str(return_date[each])
        popen = popen + str(return_open[each])
        phigh = phigh + str(return_high[each])
        plow = plow + str(return_low[each])
        pclose = pclose + str(return_close[each])
        padj = padj + str(return_adj[each])
        pvolume = pvolume + str(return_volume[each])
        pavg = pavg + str(return_avg[each])
    else:
        pdate = pdate + str(return_date[each]) + ","
        popen = popen + str(return_open[each]) + ","
        phigh = phigh + str(return_high[each]) + ","
        plow = plow + str(return_low[each]) + ","
        pclose = pclose + str(return_close[each]) + ","
        padj = padj + str(return_adj[each]) + ","
        pvolume = pvolume + str(return_volume[each]) + ","
        pavg = pavg + str(return_avg[each]) + ","
        
dates = pdate
opens = popen
highs = phigh
lows = plow
closes = pclose
adjs = padj
volumes = pvolume
avgs = pavg

