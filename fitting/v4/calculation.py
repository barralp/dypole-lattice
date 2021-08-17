#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from parameters import *

def polarizability(detuning):
    polarizability_prefactor = 3*np.pi*kappa_red*epsilon_0*c**3*Gamma_741/(omega_741**3)
    polarizability = 220*polarizability_0-polarizability_prefactor/detuning
    return polarizability

def trapFrequency(detuning, power):
    polarizability_here = polarizability(detuning)
    omega_z = np.sqrt(np.abs(32*np.pi*polarizability_here*power/((741*10**(-9))**2*beamwaist**2*epsilon_0*c*m)))
    return omega_z
    

def add_aoh(detuning, power):
    omega_z = trapFrequency(detuning, power)*(2*np.pi*10**3)
    oscillator_length = np.sqrt(hbar/(m/2*omega_z))
    add_aoh = add/oscillator_length
    return add_aoh
    
"""
def updateTrapDepth(self):
    omega_z = self.trapFrequency*(2*np.pi*10**3)
    self.trapDepth = self.m**2*omega_z**2/(self.hbar**2*(self.omega_741/self.c)**4)
    
def updateScattering(self):
    gamma_blue = self.gamma_blue_prefactor*self.power/self.beamwaist**2
    gamma_red = self.gamma_red_prefactor/(self.detuning**2)*self.power/self.beamwaist**2
    self.scattering = gamma_blue + gamma_red
"""

def getSubDF(df, variableList, variableValues):
    df_temp = df
    for variable, variableValue in zip(variableList, variableValues):
        df_temp = df_temp[df_temp[variable] == variableValue]
    return df_temp.reset_index(drop = True)

def noiseFilter(df, nMax, nMin, xWidthMax, xWidthMin, yWidthMax, yWidthMin):
    df_copy = df.copy()
    indexNames = pd.Index(np.where(df_copy['nCount'] > nMax)[0])
    df_copy.drop(indexNames, inplace = True)
    df_copy.reset_index(inplace = True, drop = True)
    indexNames = pd.Index(np.where(df_copy['nCount'] < nMin)[0])
    df_copy.drop(indexNames, inplace = True)
    df_copy.reset_index(inplace = True, drop = True)
    indexNames = pd.Index(np.where(df_copy['xWidth'] > xWidthMax)[0])
    df_copy.drop(indexNames, inplace = True)
    df_copy.reset_index(inplace = True, drop = True)
    indexNames = pd.Index(np.where(df_copy['xWidth'] < xWidthMin)[0])
    df_copy.drop(indexNames, inplace = True)
    df_copy.reset_index(inplace = True, drop = True)
    indexNames = pd.Index(np.where(df_copy['yWidth'] > yWidthMax)[0])
    df_copy.drop(indexNames, inplace = True)
    df_copy.reset_index(inplace = True, drop = True)
    indexNames = pd.Index(np.where(df_copy['yWidth'] < yWidthMin)[0])
    df_copy.drop(indexNames, inplace = True)
    df_copy.reset_index(inplace = True, drop = True)
    return df_copy

def renameUnits(df, magnification, cameraPixelSize = 6.5, axis = 'vertical', experiment = 'TOF'):
                # , xWithInSitu = 0, yWidthInSitu = 0, zWidthInSitu = 0):
    df.rename(columns = {'latticeDepth_final' : 'latticeDepth_mW'}, inplace = True)
    df.rename(columns = {'compz_rotation' : 'compz'}, inplace = True)
    df.rename(columns = {'latticeDetuning' : 'latticeDetuning_GHz'}, inplace = True)
    # inSituTrue = (df['TOF'] == 0) | (df['experiment'] == 'inSitu') # True if inSitu imaging, False if TOF
        
    if experiment == 'TOF':
        if axis == 'vertical':
            print("Fix the code here, unknown yet")
            df['xWidth_TOF_v_um'] = df['yWidth']*cameraPixelSize/magnification
            df['yWidth_TOF_v_um'] = df['xWidth']*cameraPixelSize/magnification
            # df['zWidth_TOF_v_um'] = 0
        elif axis == 'horizontal':
            df['xWidth_TOF_h_um'] = df['xWidth']*cameraPixelSize/magnification
            # df['yWidth_TOF_h_um'] = 0
            df['zWidth_TOF_h_um'] = df['yWidth']*cameraPixelSize/magnification
        else:
            print('Camera axis not defined')
    elif experiment == 'inSitu':
        # note that there is a problem as I don't know yet how to merge different inSitu dataframe to have one unique size
        # but maybe useless anyway (and partially solved with _v _h)
        if axis == 'vertical':
            df['xWidth_inSitu_v_um'] = df['yWidth']*cameraPixelSize/magnification
            df['yWidth_inSitu_v_um'] = df['xWidth']*cameraPixelSize/magnification
            # df['zWidth_inSitu_v_um'] = 0
        elif axis == 'horizontal':
            df['xWidth_inSitu_h_um'] = df['xWidth']*cameraPixelSize/magnification
            # df['yWidth_inSitu_h_um'] = 0
            df['zWidth_inSitu_h_um'] = df['yWidth']*cameraPixelSize/magnification
        else:
            print('Camera axis not defined')
    else:
        print('experiment not set')

    # cols = ["xWidth_inSitu_um", "yWidth_inSitu_um", "zWidth_inSitu_um", "xWidth_TOF_um", "yWidth_TOF_um", 'zWidth_TOF_um']
    # df[cols] = df[cols].replace([0], np.nan) # replaces the 0 by np.nan (impossible before because of the addition
    # df.dropna(axis=1, how='all')     # removes columns filled with np.nan
        
def dropExcluded(df, excluded):
    indexNames = pd.Index(np.where(df['imageID'].isin(excluded))[0])
    df.drop(indexNames, inplace = True)
    
def computeBareTrapFrequencies(df, trapFrequency_x = 2*np.pi*330, trapFrequency_y = 2*np.pi*208, trapFrequency_z = 2*np.pi*235):
    df['trapFrequencyZ_kHz'] = trapFrequency(df['latticeDetuning_GHz']*GHz, df['latticeDepth_mW']*mW)/kHz
    df['deconfiningTrapFrequency_Hz'] = np.sqrt(hbar*df['trapFrequencyZ_kHz']*kHz/(m*w_741**2))/Hz
    df['bareTrapFrequencyX_Hz'] = trapFrequency_x/Hz
    df['bareTrapFrequencyY_Hz'] = trapFrequency_y/Hz
    df['bareTrapFrequencyZ_Hz'] = trapFrequency_z/Hz

def computeTrapFrequencies(df):
    df['trapFrequencyX_Hz'] = np.sqrt(
        (df['latticeDetuning_GHz'] > 0)*
            ((df['bareTrapFrequencyX_Hz']*Hz)**2 - (df['deconfiningTrapFrequency_Hz']*Hz)**2)
        + (df['latticeDetuning_GHz'] < 0)*
            ((df['bareTrapFrequencyX_Hz']*Hz)**2 + (df['trapFrequencyZ_kHz']*kHz/aspectRatio)**2)
                                        )/Hz

    df['trapFrequencyY_Hz'] = np.sqrt(
        (df['latticeDetuning_GHz'] > 0)*
            ((df['bareTrapFrequencyY_Hz']*Hz)**2 - (df['deconfiningTrapFrequency_Hz']*Hz)**2)
        + (df['latticeDetuning_GHz'] < 0)*
            ((df['bareTrapFrequencyY_Hz']*Hz)**2 + (df['trapFrequencyZ_kHz']*kHz/aspectRatio)**2)
                                        )/Hz
    
"""def computeBareTrapFrequencies(df, trapFrequency_x, trapFrequency_y, trapFrequency_z, trapFrequency_x_blue, trapFrequency_y_blue):
    df['trapFrequencyZ_kHz'] = trapFrequency(df['latticeDetuning_GHz']*GHz, df['latticeDepth_mW']*mW)/kHz
    df['deconfiningTrapFrequency_Hz'] = np.sqrt(hbar*df['trapFrequencyZ_kHz']*kHz/(m*w_741**2))/Hz
    # I could modify this line to take into account the change in ODT 1 and 2 also
    df['bareTrapFrequencyX_Hz'] = (
                        (df['latticeDetuning_GHz'] > 0)*np.sqrt(
                            (trapFrequency_x_blue*np.sqrt(df['ODT3_Comp_final']/0.105))**2 + df['ODTFactor']*trapFrequency_x**2
                                                                )
                      + (df['latticeDetuning_GHz'] < 0)*trapFrequency_x*np.sqrt(df['ODTFactor'])
                                     )/Hz
    df['bareTrapFrequencyY_Hz'] = (
                        (df['latticeDetuning_GHz'] > 0)*np.sqrt(
                            (trapFrequency_y_blue*np.sqrt(df['ODT3_Comp_final']/0.105))**2 + df['ODTFactor']*trapFrequency_y**2
                                                                )
                      + (df['latticeDetuning_GHz'] < 0)*trapFrequency_y*np.sqrt(df['ODTFactor'])
                                     )/Hz
"""