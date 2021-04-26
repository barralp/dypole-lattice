#!/usr/bin/python
# -*- coding: utf-8 -*-

from databaseCommunication import createDataFrame, createDataFrame_2, createDataFrame_images, createDataFrame_list
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import numpy as np

def createLabeledDataFrame(labels):
    imageIDList = []
    for label in labels:
        imageIDList += list(range(label[1][0], label[1][1]+1))
    detuning = []
    field_direction = []
    TOF = []
    for imageID in imageIDList:
        for label in labels:
            if imageID in range(label[1][0], label[1][1]+1):
                detuning += [label[0][0]]
                field_direction += [label[0][1]]
                TOF += [label[0][2]]
    df = createDataFrame(imageIDList)
    df['detuning'] = detuning
    df['field_direction'] = field_direction
    df['TOF'] = TOF
    return df

def createLabeledDataFrame_2(labels):
    imageIDList = []
    for label in labels:
        imageIDList += list(range(label[1][0], label[1][1]+1))
    lattice_power = []
    field_direction = []
    TOF = []
    for imageID in imageIDList:
        for label in labels:
            if imageID in range(label[1][0], label[1][1]+1):
                lattice_power += [label[0][0]]
    df = createDataFrame(imageIDList)
    df['lattice_power'] = lattice_power
    return df

def createLabeledDataFrame_3(labels, label_head):
    imageIDList = []
    for label in labels:
        imageIDList += list(range(label[1][0], label[1][1]+1))
    labels_array = []
    lattice_power = []
    field_direction = []
    TOF = []
    for imageID in imageIDList:
        for label in labels:
            
            lattice_power += [label[0][0]]**len(label[0])
            if imageID in range(label[1][0], label[1][1]+1):
                lattice_power += [label[0][0]]
    df = createDataFrame(imageIDList)
    df['lattice_power'] = lattice_power
    return df

def createLabeledDataFrame_4(labels):
    imageIDList = []
    latticeDetuning = []
    for label in labels:
        imageIDList += list(range(label[1][0], label[1][1]+1))
    for imageID in imageIDList:
        for label in labels:
            if imageID in range(label[1][0], label[1][1]+1):
                latticeDetuning += [label[0][0]]
    df = createDataFrame_2(imageIDList)
    df['latticeDetuning'] = latticeDetuning
    return df

def createLabeledDataFrame_list(labels, label_heads, ciceroVariables, fitVariables = ['nCount', 'xWidth', 'yWidth']):
    imageIDList = []
    latticeDetuning = []
    labels_list = []
    for label in labels:
        imageIDList += list(range(label[1][0], label[1][1]+1))
    for imageID in imageIDList:
        for label in labels:
            if imageID in range(label[1][0], label[1][1]+1):
                labels_list += [label[0]]
    labels_list_array = np.array(labels_list)
    df = createDataFrame_list(imageIDList, ciceroVariables, fitVariables)
    i = 0
    for label_head in label_heads:
        df[label_head] = labels_list_array[i,:]
    return df

def createLabeledDataFrame_images(labels):
    imageIDList = []
    latticeDetuning = []
    for label in labels:
        imageIDList += list(range(label[1][0], label[1][1]+1))
    for imageID in imageIDList:
        for label in labels:
            if imageID in range(label[1][0], label[1][1]+1):
                latticeDetuning += [label[0][0]]
    df = createDataFrame_images(imageIDList)
    df['latticeDetuning'] = latticeDetuning
    return df

def N_polarized(t, N0, alpha, gamma):
    return np.sqrt(alpha / (np.exp(2*alpha*t)*(alpha/(N0**2)+gamma)-gamma))

def fitRun_1and3b(df, detuning = '1GHz', field_direction = 'Bz', TOF = 3):
    df_reduced1 = df[df['detuning'] == detuning]
    df_reduced2 = df_reduced1[df_reduced1['TOF'] == TOF]
    df_reduced = df_reduced2[df_reduced2['field_direction'] == field_direction]
    N0_guess = max(df_reduced['nCount'])
    time_guess = max(df_reduced['BECHoldTime'])
    popt, pcov = curve_fit(N_polarized, df_reduced['BECHoldTime'], df_reduced['nCount'],
                       p0 = [N0_guess, 1/time_guess, 1/(N0_guess**2*time_guess)],
                       method = 'lm'
                      )
    return popt, pcov

def fitRun_1and3b_2(df, lattice_power):
    df_reduced1 = df[df['lattice_power'] == lattice_power]
    N0_guess = max(df_reduced1['nCount'])
    time_guess = max(df_reduced1['BECHoldTime'])
    popt, pcov = curve_fit(N_polarized, df_reduced1['BECHoldTime'], df_reduced1['nCount'],
                       p0 = [N0_guess, 1/time_guess, 1/(N0_guess**2*time_guess)],
                       method = 'lm'
                      )
    return popt, pcov
    
def N_polarized_pure3b(t, N0, gamma):
    return N0 / np.sqrt(1+N0**2*gamma*t)

def getReducedDataframe(df, detuning = None, field_direction = None, TOF = None):
    df_temporary = df
    if detuning:
        df_temporary = df_temporary[df_temporary['detuning'] == detuning]
    if field_direction:
        df_temporary = df_temporary[df_temporary['field_direction'] == field_direction]
    if TOF:
        df_temporary = df_temporary[df_temporary['TOF'] == TOF]
    return df_temporary

def getReducedResults(results, detuning = None, field_direction = None, TOF = None):
    df_temporary = results
    if detuning:
        df_temporary = df_temporary[df_temporary['detuning'] == detuning]
    if field_direction:
        df_temporary = df_temporary[df_temporary['field_direction'] == field_direction]
    if TOF:
        df_temporary = df_temporary[df_temporary['TOF'] == TOF]
    return df_temporary

def fitRun_3b(df, detuning = '1GHz', field_direction = 'Bz', TOF = 3):
    df_reduced = getReducedDataframe(df, detuning, field_direction, TOF)
    N0_guess = max(df_reduced['nCount'])
    time_guess = max(df_reduced['BECHoldTime'])
    popt, pcov = curve_fit(N_polarized_pure3b, df_reduced['BECHoldTime'], df_reduced['nCount'],
                       p0 = [N0_guess, 1/(N0_guess**2*time_guess)],
                       method = 'lm'
                      )
    return popt, pcov

def fitRun_3b_linear(df, lattice_power, time_max):
    df_reduced = df[df['lattice_power'] == lattice_power]
    df_reduced1 = df_reduced[df_reduced['BECHoldTime'] <= time_max]
    N0_guess = max(df_reduced1['nCount'])
    time_guess = max(df_reduced1['BECHoldTime'])
    popt, pcov = curve_fit(N_linear, df_reduced1['BECHoldTime'], df_reduced1['nCount'],
                       p0 = [N0_guess, time_guess],
                       method = 'lm'
                      )
    return popt, pcov

def N_linear(t, N0, gamma):
    return N0 - gamma*t

def fitRun_3b_2(df, lattice_power):
    df_reduced = df[df['lattice_power'] == lattice_power]
    N0_guess = max(df_reduced['nCount'])
    time_guess = max(df_reduced['BECHoldTime'])
    popt, pcov = curve_fit(N_polarized_pure3b, df_reduced['BECHoldTime'], df_reduced['nCount'],
                       p0 = [N0_guess, 1/(N0_guess**2*time_guess)],
                       method = 'lm'
                      )
    return popt, pcov

def fitRun_3b_3(df_run):
    N0_guess = max(df_run['nCount'])
    time_guess = max(df_run['BECHoldTime'])
    popt, pcov = curve_fit(N_polarized_pure3b, df_run['BECHoldTime'], df_run['nCount'],
                       p0 = [N0_guess, 1/(N0_guess**2*time_guess)],
                       method = 'lm'
                      )
    return popt, pcov

def fitRun_3b_4(df_run):
    N0_guess = max(df_run['nCount'])
    time_guess = max(df_run['BECHoldTime'])
    popt, pcov = curve_fit(N_polarized, df_run['BECHoldTime'], df_run['nCount'],
                       p0 = [N0_guess, 1/time_guess, 1/(N0_guess**2*time_guess)],
                       method = 'lm'
                      )
    return popt, pcov

def deltaTime(results, detuning, TOF):
    holdTime = np.linspace(0,20, 10000)
    results_Bz = getReducedResults(results, detuning, 'Bz', TOF)
    results_By = getReducedResults(results, detuning, 'By', TOF)
    N0_z, gamma_z = results_Bz['N0'].iloc[0], results_Bz['gamma'].iloc[0]
    N0_y, gamma_y = results_By['N0'].iloc[0], results_By['gamma'].iloc[0]
    index = np.argmin(np.abs(N_polarized_pure3b(holdTime, N0_z, gamma_z)-N_polarized_pure3b(0, N0_y, gamma_y)))
    return holdTime[index]


def powerlaw(t, N0, alpha, beta):
    return (N0**(-(beta-1))+(beta-1)*alpha*t)**(-1/(beta-1))

def fit_powerlaw(df, detuning = '1GHz', field_direction = 'Bz', TOF = 3):
    df_reduced = getReducedDataframe(df, detuning, field_direction, TOF)
    N0_guess = max(df_reduced['nCount'])
    time_guess = max(df_reduced['BECHoldTime'])
    beta_guess = 3
    popt, pcov = curve_fit(powerlaw, df_reduced['BECHoldTime'], df_reduced['nCount'],
                       p0 = [N0_guess, 1/(N0_guess**2*time_guess), beta_guess],
                       method = 'lm'
                      )
    return popt, pcov
