#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 09:25:17 2020

@author: patrickmayerhofer

This library was created for the use in the open source Wearables Course BPK409, Lab2 - PPG
For more information: 
    https://docs.google.com/document/d/e/2PACX-1vTr1zOyrUedA1yx76olfDe5jn88miCNb3EJcC3INmy8nDmbJ8N5Y0B30EBoOunsWbA2DGOVWpgJzIs9/pub
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps

"""Step 1: This function finds the peaks of the derivative of the PPG signal
Input: ppg signal, time
Output: ppg derivative, position of peaks of d_ppg"""
def dppg_peaks(ppg, time):
    """Step 1: Find the peaks of the derivative of the PPG signal"""
    d_ppg = np.diff(ppg) #find derivative of ppg signal
    peaks_d_ppg,_ = sps.find_peaks(d_ppg) #peaks of d_ppg
     
    # plot step 1
    # plt.figure()
    # plt.plot(time[0:len(time)-1], d_ppg, color = 'red')
    # plt.plot(time[peaks_d_ppg], d_ppg[peaks_d_ppg], "x", color = 'g')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Derivative of activation []')
    # plt.title('R-wave peaks step 1: peaks of derivative of PPG')
    # plt.show()
    return d_ppg, peaks_d_ppg
    
"""Step 2: This function filters out all peaks that are under the height threshold
    and not over a minimum distance from each other. \
     Input: d_ppg signal, position of peaks from dppg_peaks(), time, 
         height threshold percentage in decimal, distance threshold in decimal
     Output: Rwave peaks of d_ppg"""    
def d_ppg_peaks(d_ppg, peaks_d_ppg, std_threshold, time, heightper, distanceper):
    meanpeaks_d_ppg = np.mean(d_ppg[peaks_d_ppg]) # find the mean of the peaks
    stdpeaks_d_ppg = np.std(d_ppg[peaks_d_ppg])
    threshold = (meanpeaks_d_ppg+std_threshold*stdpeaks_d_ppg)*heightper #IMPROVE use mean + n standard deviations for finding peaks. it filters out all the peaks from the bottom and those too short to be an R peak
    newpeaks_d_ppg,_ = sps.find_peaks(d_ppg, height = threshold) # find the new peaks
    # newpeaks_d_ppg_t = time[newpeaks_d_ppg]
    # newpeaks_d_ppg_t = newpeaks_d_ppg_t.reset_index(drop = True)
    meandistance = np.mean(np.diff(newpeaks_d_ppg))
    Rwave_peaks_d_ppg,_ = sps.find_peaks(d_ppg,height = threshold, distance = meandistance*distanceper) # 
    
      #plot step 2
    # plt.figure()  
    # plt.plot(time[0:len(time)-1], d_ppg, color = 'red') 
    # plt.plot(time[Rwave_peaks_d_ppg], d_ppg[Rwave_peaks_d_ppg], "x", color = 'g')
    # #plt.axhline(meanpeaks_d_ppg, color = 'b')
    # #plt.axhline(max_d_ppg, color = 'b')
    # thres = plt.axhline(threshold, color = 'black', label = 'threshold')
    # plt.title('R-wave peaks step 2: d_PPG peaks')
    # plt.ylabel('Derivative of activation []')
    # plt.xlabel('Time [s]')
    # plt.legend()
    # plt.show()
    return Rwave_peaks_d_ppg, threshold
    

    
"""Step 3: this function finds the Rwave peaks at the original ppg signal
with the before defined peaks of the d_ppg signal
Input: ppg signal,derivative of ppg signal,
    Rwave peaks of d_ppg from height_distance_threshold_peaks
Output: Rwave peaks"""
def Rwave_peaks(ppg, d_ppg, Rwave_peaks_d_ppg, time):   
    if len(Rwave_peaks_d_ppg) > 1:
        Rwave = np.empty([len(Rwave_peaks_d_ppg)], np.int64)
    else:
        # Handle the case where the length is not sufficient (e.g., print an error message)
        print("Not enough peaks to create Rwave array.")
        Rwave_t = np.empty(1)
        return Rwave_t

    for i in range(0, len(Rwave)): # for all peaks
        start = Rwave_peaks_d_ppg[i-1]
        if Rwave_peaks_d_ppg[i-1] > Rwave_peaks_d_ppg[i]:
            start = 0
        # print(start)
        # print(Rwave_peaks_d_ppg[i])
        end = Rwave_peaks_d_ppg[i]
        if i < len(Rwave)-1:
            end = Rwave_peaks_d_ppg[i+1]
        # print(end)

        ppglowrange = ppg[start:Rwave_peaks_d_ppg[i]] # create array that contains of the ppg within the d_ppg_peaks
        ppghighrange = ppg[Rwave_peaks_d_ppg[i]:end] # create array that contains of the ppg within the d_ppg_peaks
        mx = np.max(ppg[start:end])
        # print(mx)
        percentage = np.round((len(ppglowrange)+len(ppghighrange))*0.2)
        
        ppglowrange = ppglowrange[-int(percentage):]
        ppghighrange = ppghighrange[:int(percentage)]
        # print(ppglowrange, ppghighrange)
        
        if len(ppglowrange)>0 and len(ppghighrange)>0:
            ppgmax = max(np.max(ppglowrange), np.max(ppghighrange))
        elif len(ppglowrange)>0:
            ppgmax = np.max(ppglowrange)
        elif len(ppghighrange)>0:
            ppgmax = np.max(ppghighrange)

        lowmaxpos = np.array(list(np.where(ppglowrange == ppgmax))) - (len(ppglowrange)) # find the index of the max value of ppg
        highmaxpos = np.array(list(np.where(ppghighrange == ppgmax))) # find the index of the max value of ppg
        maxpos = lowmaxpos
        if not lowmaxpos:
            maxpos = highmaxpos

        Rwave[i] = Rwave_peaks_d_ppg[i] + maxpos[0,0]  # save this index
        # print(Rwave[i])
    
    # Rwave = Rwave.astype(np.int64)
    # Rwave_t = time[Rwave]
    # Rwave_t = Rwave_t.reset_index(drop = True)
    # Rwave_t = Rwave_t.drop(columns = ['index'])
    
    # plot step 3
    # fig, ax1 = plt.subplots()
    # ax1.plot(time[0:len(time)-1], d_ppg, color = 'r', label = 'Derivative of PPG')
    # ax1.set_ylabel('Activation Derivative []')
    # plt.xlabel('Time [s]') 
    # plt.title('R-wave peaks step 3: R-wave peaks')
    # ax2 = ax1.twinx()
    # ax2.plot(time, ppg, color = 'b', label = 'PPG')
    # ax2.plot(time[Rwave], ppg[Rwave], "x", color = 'g')
    # ax2.set_ylabel('Activation []')
    # ax1.legend()
    # ax2.legend()
    # plt.show()
    return Rwave

def Rwave_t_peaks(time, Rpeaks):
    Rpeaks = Rpeaks.astype(np.int64)
    Rwave_t = time[Rpeaks]
    Rwave_t = Rwave_t.reset_index(drop = True)
    Rwave_t = Rwave_t.drop(columns = ['index'])
    return Rwave_t