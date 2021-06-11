#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

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

beamwaist = 27*10**(-6)

def polarizability(detuning):
    polarizability_prefactor = 3*np.pi*kappa_red*epsilon_0*c**3*Gamma_741/(omega_741**3)
    polarizability = 220*polarizability_0-polarizability_prefactor/detuning
    return polarizability

def trapFrequency(detuning, power):
    polarizability_here = polarizability(detuning)
    omega_z = np.sqrt(np.abs(32*np.pi*polarizability_here*power/((741*10**(-9))**2*beamwaist**2*epsilon_0*c*m)))
    trapFrequency = omega_z/(2*np.pi*10**3)
    return trapFrequency
    

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
    return df_temp.reset_index()
