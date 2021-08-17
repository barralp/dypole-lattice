#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

trapFrequency_x = 2*np.pi*70 # corrected with parametric heating measurement. Correspond to end of evap ODT1 = 0.38 / ODT2 = 0.36 / ODT3 = 0.105
trapFrequency_y = 2*np.pi*28
trapFrequency_z = 2*np.pi*50

trapFrequency_x_blue = 2*np.pi*28 #*np.sqrt(1/0.105) # corrected with parametric heating measurement. with 2pi
trapFrequency_y_blue = 2*np.pi*28 #*np.sqrt(1/0.105)
trapFrequency_z_blue = 0

aspectRatio = 162
w_741 = 27*10**(-6)
#beamwaist = 27*10**(-6)
beamwaist = 37*10**(-6)
#m = 164*1.67*10**(-27)
m = 162*1.67*10**(-27)
k_B = 1.38*10**(-23)
hbar = 6.626*10**(-34)/(2*np.pi)

# units
# recreate and analysis file where all the constants are defined in one file
Hz = 2*np.pi
kHz = 2*np.pi*10**3
GHz = 2*np.pi*10**9
nK = 10**(-9)
ms = 10**(-3)
mW = 10**(-3)
um = 10**(-6)
nm = 10**(-9)
cm3 = 10**(-6)


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


