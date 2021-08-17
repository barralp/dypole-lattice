
import numpy as np 

time_without_gradient = [1.8, 4.3] 

time_with_gradient = [2.0, 6.5] 

timeconstant_without_gradient = {1: {'tau1': 1, 'tau2': 1.2}, 2: {'tau1': 1, 'tau2': 1.2}, 3: {'tau1': 100000, 'tau2': 1.2}} 

voltage_without_gradient = {1: {'init': 6, 'final1': 2.1, 'final2': 0.34}, 2: {'init': 5.4, 'final1': 2.1, 'final2': 0.35}, 3: {'init': 5.4, 'final1': 5.4, 'final2': 0.095}} 

timeconstant_with_gradient = {1: {'tau1': 1, 'tau2': 0.9}, 2: {'tau1': 1, 'tau2': 0.9}, 3: {'tau1': 100000, 'tau2': 0.9}} 

voltage_with_gradient = {1: {'init': 6, 'final1': 1.1, 'final2': 0.09}, 2: {'init': 5.5, 'final1': 1.1, 'final2': 0.11}, 3: {'init': 5.4, 'final1': 5.4, 'final2': 0.05}} 

def exponential(V_init, V_final, tau, t_f, t):
    return V_init+(V_final - V_init)*(np.exp(-t/tau)-1)/(np.exp(-t_f/tau)-1)

def two_exponentials(V_init, V_final1, V_final2, tau1, tau2, t_f1, t_f2, t):
    result1 = (t <= t_f1)*exponential(V_init, V_final1, tau1, t_f1, t)
    result2 = (t > t_f1)*(t <= t_f2)*exponential(V_final1, V_final2, tau2, t_f2 - t_f1, t - t_f1)
    return result1 + result2 

def ODT_voltage_to_power(ODT_number, V):
    if ODT_number == 1:
        return 1.5247161536596812*V + -0.05154300512506249
    elif ODT_number == 2:
        return 1.3098684011439605*V + -0.010412796842118596
    elif ODT_number == 3:
        return 0.4733333343134272*V + -0.0028333339102657273 

def ODT_voltage(ODT_number, t, with_gradient = False):
    if with_gradient:
        voltage = voltage_with_gradient[ODT_number]
        timeconstant = timeconstant_with_gradient[ODT_number]
        t_f1, t_f2 = time_with_gradient
    else:
        voltage = voltage_without_gradient[ODT_number]
        timeconstant = timeconstant_without_gradient[ODT_number]
        t_f1, t_f2 = time_without_gradient
    return two_exponentials(voltage["init"], voltage["final1"], voltage["final2"], timeconstant["tau1"], timeconstant["tau2"], t_f1, t_f2, t) 

def ODT_power(ODT_number, t, with_gradient = False):
    return ODT_voltage_to_power(ODT_number, ODT_voltage(ODT_number, t, with_gradient)) 

