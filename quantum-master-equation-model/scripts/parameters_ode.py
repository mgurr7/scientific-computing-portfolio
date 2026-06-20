# Parameters in SI units

import numpy as np


class material_parameters:
    
    def __init__(self):
        
        
        self.electron_mass = 9.1093837e-31
        self.material_velocity = 6.7e5
        self.electron_charge = 1.60217663e-19
        self.polarization_normalized = 1
        self.polarization = (self.polarization_normalized*self.electron_mass*self.material_velocity)**2
        self.epsilon_0 = 8.8541878128e-12
        self.Lz = (1e-5)
        #self.Lz = (1e-3)
        self.L = 20e-9
        
        #self.surface = (100e-9)**2
        self.surface = (15*1e-6)**2 
        self.eV = 1.602176634e-19
        self.h_bar = 1.054*10**(-34)
        
        self.h = 6.62607015*10**(-34)
        self.mode_resonance = 2.01*self.eV/self.h_bar
        self.exciton_resonance = 2.01*self.eV/self.h_bar
        #self.n_x = 10**(14)
        self.e_b =  8.33132e-20
        self.a_B = 1.95*10**(-9)
        #self.e_0 = - 3.204353e-22
        self.k_B = 1.380649e-23
        self.T_l = 8
        
        self.mass = 0.67*9.11e-31
        
        #self.e_LP = -0.76e-3*self.eV
        #self.beta_l = (self.k_B*self.T_l)**(-1)
        self.rho_x = self.mass/(self.h_bar)**2
        self.SW = (self.e_b*2.07*(self.a_B**2)/self.h_bar)
        self.G_0 = np.sqrt((self.h_bar*self.polarization*(self.electron_charge)**2)/(np.pi*self.epsilon_0*(self.electron_mass*self.a_B)**2*self.Lz*self.mode_resonance))
        
        self.e_lower = self.h_bar*(self.mode_resonance+self.exciton_resonance)/2-np.sqrt((self.h_bar**2)*(self.mode_resonance+self.exciton_resonance)**2+4*(self.G_0**2-((self.h_bar**2)*(self.mode_resonance*self.exciton_resonance))))/2
        self.e_upper = self.h_bar*(self.mode_resonance+self.exciton_resonance)/2+np.sqrt((self.h_bar**2)*(self.mode_resonance+self.exciton_resonance)**2+4*(self.G_0**2-((self.h_bar**2)*(self.mode_resonance*self.exciton_resonance))))/2

        
        self.X_LP = np.sqrt(((self.h_bar*self.exciton_resonance*self.e_upper-self.h_bar*self.mode_resonance*self.e_lower)/(self.e_upper**2-self.e_lower**2)))
        self.C_LP = np.sqrt(((self.h_bar*self.mode_resonance*self.e_upper-self.h_bar*self.exciton_resonance*self.e_lower)/(self.e_upper**2-self.e_lower**2)))
        
        self.W_0 = self.SW*(self.X_LP**4)*self.polarization_normalized*self.h_bar/(2*np.pi*self.L**2) #has the dimensions of an energy
        self.gamma_LP = 5e11 #(self.C_LP**2)/2e-11 

        self.e_LP = self.e_lower-self.h_bar*self.exciton_resonance  
      
        self.joule = 6.242e+18
        self.px =  3.7824899063893974e+26#4.05e26#3.782e26#3.214183794244142e+26#4.17e26#3.7824899063893974e+26#2.3209007328226485e+26#3.214e26#1.6759e26#4.17e26# cccc
        self.gamma_thermalization= 1e12
        self.n_max = 20#50
        
