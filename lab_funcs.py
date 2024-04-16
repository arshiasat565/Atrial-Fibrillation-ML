#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 09:25:17 2020

@author: patrickmayerhofer

This library was created for the use in the open source Wearables Course BPK409, Lab2 - ECG
For more information: 
    https://docs.google.com/document/d/e/2PACX-1vTr1zOyrUedA1yx76olfDe5jn88miCNb3EJcC3INmy8nDmbJ8N5Y0B30EBoOunsWbA2DGOVWpgJzIs9/pub
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sps

"""Step 1: This function finds the peaks of the derivative of the ECG signal
Input: ecg signal, time
Output: ecg derivative, position of peaks of d_ecg"""
def decg_peaks(ecg, time):
    """Step 1: Find the peaks of the derivative of the ECG signal"""
    d_ecg = np.diff(ecg) #find derivative of ecg signal
    peaks_d_ecg,_ = sps.find_peaks(d_ecg) #peaks of d_ecg
     
    # plot step 1
    # plt.figure()
    # plt.plot(time[0:len(time)-1], d_ecg, color = 'red')
    # plt.plot(time[peaks_d_ecg], d_ecg[peaks_d_ecg], "x", color = 'g')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Derivative of activation []')
    # plt.title('R-wave peaks step 1: peaks of derivative of ECG')
    # plt.show()
    return d_ecg, peaks_d_ecg
    
"""Step 2: This function filters out all peaks that are under the height threshold
    and not over a minimum distance from each other. \
     Input: d_ecg signal, position of peaks from decg_peaks(), time, 
         height threshold percentage in decimal, distance threshold in decimal
     Output: Rwave peaks of d_ecg"""    
def d_ecg_peaks(d_ecg, peaks_d_ecg, time, heightper, distanceper):
    meanpeaks_d_ecg = np.mean(d_ecg[peaks_d_ecg]) # find the mean of the peaks
    stdpeaks_d_ecg = np.std(d_ecg[peaks_d_ecg])
    threshold = (meanpeaks_d_ecg+2*stdpeaks_d_ecg)*heightper #IMPROVE use mean + 2 standard deviations for finding peaks. it filters out all the peaks from the bottom and those too short to be an R peak
    newpeaks_d_ecg,_ = sps.find_peaks(d_ecg, height = threshold) # find the new peaks
    # newpeaks_d_ecg_t = time[newpeaks_d_ecg]
    # newpeaks_d_ecg_t = newpeaks_d_ecg_t.reset_index(drop = True)
    meandistance = np.mean(np.diff(newpeaks_d_ecg))
    Rwave_peaks_d_ecg,_ = sps.find_peaks(d_ecg,height = threshold, distance = meandistance*distanceper) # 
    
      #plot step 2
    # plt.figure()  
    # plt.plot(time[0:len(time)-1], d_ecg, color = 'red') 
    # plt.plot(time[Rwave_peaks_d_ecg], d_ecg[Rwave_peaks_d_ecg], "x", color = 'g')
    # #plt.axhline(meanpeaks_d_ecg, color = 'b')
    # #plt.axhline(max_d_ecg, color = 'b')
    # thres = plt.axhline(threshold, color = 'black', label = 'threshold')
    # plt.title('R-wave peaks step 2: d_ECG peaks')
    # plt.ylabel('Derivative of activation []')
    # plt.xlabel('Time [s]')
    # plt.legend()
    # plt.show()
    return Rwave_peaks_d_ecg
    

    
"""Step 3: this function finds the Rwave peaks at the original ecg signal
with the before defined peaks of the d_ecg signal
Input: ecg signal,derivative of ecg signal,
    Rwave peaks of d_ecg from height_distance_threshold_peaks
Output: Rwave peaks"""
def Rwave_peaks(ecg, d_ecg, Rwave_peaks_d_ecg, time):   
    if len(Rwave_peaks_d_ecg) > 1:
        Rwave = np.empty([len(Rwave_peaks_d_ecg)], np.int64)
    else:
        # Handle the case where the length is not sufficient (e.g., print an error message)
        print("Not enough peaks to create Rwave array.")
        Rwave_t = np.empty(1)
        return Rwave_t

    for i in range(0, len(Rwave)): # for all peaks
        start = Rwave_peaks_d_ecg[i-1]
        if Rwave_peaks_d_ecg[i-1] > Rwave_peaks_d_ecg[i]:
            start = 0
        # print(start)
        # print(Rwave_peaks_d_ecg[i])
        end = Rwave_peaks_d_ecg[i]
        if i < len(Rwave)-1:
            end = Rwave_peaks_d_ecg[i+1]
        # print(end)

        ecglowrange = ecg[start:Rwave_peaks_d_ecg[i]] # create array that contains of the ecg within the d_ecg_peaks
        ecghighrange = ecg[Rwave_peaks_d_ecg[i]:end] # create array that contains of the ecg within the d_ecg_peaks
        mx = np.max(ecg[start:end])
        # print(mx)
        percentage = np.round((len(ecglowrange)+len(ecghighrange))*0.2)
        
        ecglowrange = ecglowrange[-int(percentage):]
        ecghighrange = ecghighrange[:int(percentage)]
        # print(ecglowrange, ecghighrange)
        
        if len(ecglowrange)>0 and len(ecghighrange)>0:
            ecgmax = max(np.max(ecglowrange), np.max(ecghighrange))
        elif len(ecglowrange)>0:
            ecgmax = np.max(ecglowrange)
        elif len(ecghighrange)>0:
            ecgmax = np.max(ecghighrange)

        lowmaxpos = np.array(list(np.where(ecglowrange == ecgmax))) - (len(ecglowrange)) # find the index of the max value of ecg
        highmaxpos = np.array(list(np.where(ecghighrange == ecgmax))) # find the index of the max value of ecg
        maxpos = lowmaxpos
        if not lowmaxpos:
            maxpos = highmaxpos

        Rwave[i] = Rwave_peaks_d_ecg[i] + maxpos[0,0]  # save this index
        # print(Rwave[i])
    
    # Rwave = Rwave.astype(np.int64)
    # Rwave_t = time[Rwave]
    # Rwave_t = Rwave_t.reset_index(drop = True)
    # Rwave_t = Rwave_t.drop(columns = ['index'])
    
    # plot step 3
    # fig, ax1 = plt.subplots()
    # ax1.plot(time[0:len(time)-1], d_ecg, color = 'r', label = 'Derivative of ECG')
    # ax1.set_ylabel('Activation Derivative []')
    # plt.xlabel('Time [s]') 
    # plt.title('R-wave peaks step 3: R-wave peaks')
    # ax2 = ax1.twinx()
    # ax2.plot(time, ecg, color = 'b', label = 'ECG')
    # ax2.plot(time[Rwave], ecg[Rwave], "x", color = 'g')
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