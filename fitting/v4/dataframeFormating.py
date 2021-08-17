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
    labels_list_array = np.array(labels_list).T
    df = createDataFrame_list(imageIDList, ciceroVariables, fitVariables)
    i = 0
    for label_head in label_heads:
        df[label_head] = labels_list_array[i,:]
        i += 1
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

class Model():
    def __init__(self):
        self.power = 0 # in W
        self.beamwaist = 0 # in um
        self.detuning = 0 # in GHz
        
        self.trapFrequency = 0  # in kHz
        self.trapDepth = 0  # in kHz
        self.add_aoh = 0    # no units
        self.scattering = 0 # in s-1
        
        self.initConstants()
    
kappa_red = (1/2)*(1+1/153)
hbar = 1.0545718*10**-34
c = 2.99792458*10**8
polarizability_0 = 1.64877727*10**(-41)
m = 163.929*1.66053906660*10**(-27)
epsilon_0 = 8.85418782*10**(-12)
k_B = 1.38064852*10**(-23)
a_0 = 5.29177210903*10**(-11)
mu_B = 9.274009994*10**(-24)
Gamma_741 = 2*np.pi*1.78*10**3   # in Hz
omega_741 = 2*np.pi*c/(741*10**(-9))
Gamma_421 = 2*np.pi*32.2*10**6
omega_421 = 2*np.pi*c/(421.172*10**(-9))
delta_421 = omega_421 - omega_741
omega_405 = 2*np.pi*c/(404.597*10**(-9))
omega_419 = 2*np.pi*c/(418.682*10**(-9))
omega_419_bis = 2*np.pi*c/(419.484*10**(-9))
Gamma_405 = 1.92*10**8
Gamma_419 = 1.26*10**8
Gamma_419_bis = 8.8*10**7
kappa_blue = (1/2)*((1+1/153)+(1/9)*(Gamma_419/Gamma_421)**2
                                 *(omega_421/omega_419)**6
                                 +(1/9)*(Gamma_419_bis/Gamma_421)**2
                                 *(omega_421/omega_419_bis)**6
                                 +(15/17)*(Gamma_405/Gamma_421)**2
                                 *(omega_421/omega_405)**6)
polarizability_prefactor = 3*np.pi*kappa_red*epsilon_0*c**3*Gamma_741/(omega_741**3)
add =m/2 * (1/(epsilon_0*c**2))/(4*np.pi)*(10*mu_B/hbar)**2
gamma_blue_prefactor = 3*kappa_blue*c**2*Gamma_421**2/(hbar*delta_421**2*omega_421**3)*(421/741)**3
gamma_red_prefactor = 3*kappa_red*c**2*Gamma_741**2/(hbar*omega_741**3)

def polarizability(detuning):
    polarizability_prefactor = 3*np.pi*kappa_red*epsilon_0*c**3*Gamma_741/(omega_741**3)
    polarizability = 220*polarizability_0-polarizability_prefactor/detuning
    return polarizability

def trapFrequency(detuning, power):
    polarizability = polarizability(detuning)
    omega_z = np.sqrt(np.abs(32*np.pi*polarizability*power/((741*10**(-9))**2*beamwaist**2*epsilon_0*c*m)))
    trapFrequency = omega_z/(2*np.pi*10**3)
    return trapFrequency
    
def updateTrapDepth(self):
    omega_z = self.trapFrequency*(2*np.pi*10**3)
    self.trapDepth = self.m**2*omega_z**2/(self.hbar**2*(self.omega_741/self.c)**4)
    
def add_aoh(detuning, power):
    omega_z = trapFrequency(detuning, power)*(2*np.pi*10**3)
    oscillator_length = np.sqrt(hbar/(m/2*omega_z))
    add_aoh = add/oscillator_length
    return add_aoh
    
def updateScattering(self):
    gamma_blue = self.gamma_blue_prefactor*self.power/self.beamwaist**2
    gamma_red = self.gamma_red_prefactor/(self.detuning**2)*self.power/self.beamwaist**2
    self.scattering = gamma_blue + gamma_red

