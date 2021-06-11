#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
from calculation import getSubDF
import pandas as pd

## Import those function in the file

def N_polarized_pure3b(t, N0, gamma):
    return N0 / np.sqrt(1+2*N0**2*gamma*t)

def N_polarized(t, N0, alpha, gamma):
    return np.sqrt(alpha / (np.exp(2*alpha*t)*(alpha/(N0**2)+gamma)-gamma))

def N_linear(t, N0, b):
    return N0-b*t



def fitRun_pure(df_run):
    N0_guess = max(df_run['nCount'])
    time_guess = max(df_run['BECHoldTime'])
    popt, pcov = curve_fit(N_polarized_pure3b, df_run['BECHoldTime'], df_run['nCount'],
                       p0 = [N0_guess, 1/(N0_guess**2*time_guess)],
                          )#bounds = ((0, 0), (np.inf, np.inf)),
                           #method = 'trf'
                      #)
    return popt, pcov

def fitRun_mix(df_run):
    N0_guess = max(df_run['nCount'])
    time_guess = max(df_run['BECHoldTime'])
    alpha_guess = 1/time_guess
    gamma_guess = 1/(N0_guess**2*time_guess)
    popt, pcov = curve_fit(N_polarized, df_run['BECHoldTime'], df_run['nCount'],
                       p0 = [N0_guess, alpha_guess, gamma_guess],
                          )#bounds = ((N0_guess/10, alpha_guess/10, gamma_guess/100), (N0_guess*10, alpha_guess*10, gamma_guess*100)),
                           #method = 'trf'
                      #)
    return popt, pcov

def fitRun_lin(df_run, tmax_ms):
    N0_guess = max(df_run['nCount'])
    time_guess = max(df_run['BECHoldTime'])
    df_run_cut = df_run[df_run['BECHoldTime'] < tmax_ms]
    popt, pcov = curve_fit(N_linear, df_run_cut['BECHoldTime'], df_run_cut['nCount'],
                       p0 = [N0_guess, N0_guess/time_guess],
                          )#bounds = ((0, 0), (np.inf, np.inf)),
                           #method = 'trf'
                      #)
    return popt, pcov

def buildResultIndex(df, parametersList):
    # parametersList = ['latticeDetuning', 'latticeDepth', 'ODTFactor', 'isotope']
    parametersValuesList = []
    for parameter in parametersList:
        parametersValuesList += [list(np.sort(df[parameter].unique()))]
    return np.array(np.meshgrid(*parametersValuesList)).T.reshape(-1, len(parametersList))

def fitDF(df, parametersList, fitType = 'pure', tmax_ms = 20):
    resultsIndex = buildResultIndex(df, parametersList)
    results = []
    for runParameters in resultsIndex:
        df_run = getSubDF(df, parametersList, runParameters)
        if not df_run.empty:
            # for debugging only, helps to check that the std is 0 on the values that should be the same run
            # runParametersMean = df_run.mean().add_suffix('_mean').to_dict()
            # runParametersStd = df_run.std().add_suffix('_std').to_dict()
            runParameters = df_run.mean().to_dict()
            if fitType == 'pure':
                popt, pcov = fitRun_pure(df_run)
                results += [{'N0' : popt[0],
                             'alpha' : np.nan,
                             'gamma' : popt[1],
                             'b' : np.nan
                                    }]
            elif fitType == 'mix':
                popt, pcov = fitRun_mix(df_run)
                results += [{'N0' : popt[0],
                             'alpha' : popt[1],
                             'gamma' : popt[2],
                             'b' : np.nan
                                    }]
            elif fitType == 'lin':
                popt, pcov = fitRun_lin(df_run, tmax_ms)
                results += [{'N0' : popt[0],
                             'alpha' : np.nan,
                             'gamma' : 2*popt[1]/popt[0]**3,  # gamm = 2b/N0**3, which links the linear model to the rest
                             'b' : popt[1]
                                    }]
            # results[-1].update(runParametersMean)
            # results[-1].update(runParametersStd)
            results[-1].update(runParameters)
    return pd.DataFrame(results)
                                #.drop(columns = ['level_0']) # just to remove the index that went through getSubDF
"""
            popt, pcov = fitRun(df_run)

    resultsDF = pd.DataFrame(results)
    for latticeDetuning in latticeDetuningList:
        for latticeDepth in latticeDepthList:
            if latticeDetuning > 0:
                mean_size_line = getSubDF(df_inSitu_mean, ['latticeDepth', 'latticeDetuning'], [latticeDepth, latticeDetuning])
                mean_size = ((mean_size_line['xWidth'] + mean_size_line['yWidth'])/2).mean()
            else:
                mean_size = np.nan
            for compz in compzList:
                for ODTFactor in ODTFactorList:
                    df_run = getSubDF(df, ['latticeDepth', 'latticeDetuning', 'compz', 'ODTFactor'], [latticeDepth, latticeDetuning, compz, ODTFactor])
                    if not df_run.empty:
                        popt, pcov = fitRun(df_run)
                        results += [{'latticeDetuning' : latticeDetuning,
                                         'latticeDepth' : latticeDepth,
                                        'N0' : popt[0],
                                        'alpha' : min(-10**(-10), popt[1]),
                                         'gamma' : popt[2],
                                     'trapFrequency' : df_run['trapFrequency'].mean(),
                                     'trapFrequencyPerp1' : df_run['trapFrequencyPerp1'].mean(),
                                     'trapFrequencyPerp2' : df_run['trapFrequencyPerp2'].mean(),
                                     'compz' : compz,
                                     'ODTFactor' : ODTFactor,
                                    'mean_size' : mean_size
                                    }]
    resultsDF = pd.DataFrame(results)

for latticeDetuning in latticeDetuningList:
    for latticeDepth in latticeDepthList:
        if latticeDetuning > 0:
            mean_size_line = getSubDF(df_inSitu_mean, ['latticeDepth', 'latticeDetuning'], [latticeDepth, latticeDetuning])
            mean_size = ((mean_size_line['xWidth'] + mean_size_line['yWidth'])/2).mean()
        else:
            mean_size = np.nan
        for compz in compzList:
            for ODTFactor in ODTFactorList:
                df_run = getSubDF(df, ['latticeDepth', 'latticeDetuning', 'compz', 'ODTFactor'], [latticeDepth, latticeDetuning, compz, ODTFactor])
                if not df_run.empty:
                    popt, pcov = fitRun_pure(df_run)
                    results_pure += [{'latticeDetuning' : latticeDetuning,
                                     'latticeDepth' : latticeDepth,
                                    'N0' : popt[0],
                                    'alpha' : 0,
                                     'gamma' : popt[1],
                                      'trapFrequency' : df_run['trapFrequency'].mean(),
                                     'trapFrequencyPerp1' : df_run['trapFrequencyPerp1'].mean(),
                                     'trapFrequencyPerp2' : df_run['trapFrequencyPerp2'].mean(),
                                      'compz' : compz,
                                      'ODTFactor' : ODTFactor,
                                      'mean_size' : mean_size
                                }]
resultsDF_pure = pd.DataFrame(results_pure)

for latticeDetuning in latticeDetuningList:
    for latticeDepth in latticeDepthList:
        if latticeDetuning > 0:
            mean_size_line = getSubDF(df_inSitu_mean, ['latticeDepth', 'latticeDetuning'], [latticeDepth, latticeDetuning])
            mean_size = ((mean_size_line['xWidth'] + mean_size_line['yWidth'])/2).mean()
        else:
            mean_size = np.nan
        for compz in compzList:
            for ODTFactor in ODTFactorList:
                df_run = getSubDF(df, ['latticeDepth', 'latticeDetuning', 'compz', 'ODTFactor'], [latticeDepth, latticeDetuning, compz, ODTFactor])
                if not df_run.empty:
                    popt, pcov = fitRun_lin(df_run[df_run['BECHoldTime']<25])
                    results_lin += [{'latticeDetuning' : latticeDetuning,
                                     'latticeDepth' : latticeDepth,
                                    'N0' : popt[0],
                                     'b' : popt[1],
                                     'trapFrequency' : df_run['trapFrequency'].mean(),
                                     'trapFrequencyPerp1' : df_run['trapFrequencyPerp1'].mean(),
                                     'trapFrequencyPerp2' : df_run['trapFrequencyPerp2'].mean(),
                                     'compz' : compz,
                                     'ODTFactor' : ODTFactor,
                                      'mean_size' : mean_size
                                }]
resultsDF_lin = pd.DataFrame(results_lin)"""