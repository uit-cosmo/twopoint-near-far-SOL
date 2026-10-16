from math import pi
me = 9.109e-31 # Electron mass in kg
Da = 1.661e-27 # Unified atomic mass unit in kg
mD = 2.014*Da # Deuterium mass in kg
eps0 = 55.26e6 # Vacuum permittivity in units of e^2 eV^{-1} m^{-1} where e is the elementary charge
e = 1.602e-19 #Elemetary chage in C
lnLambdaC = 16 # Coulomb logarithm [Hazeltine, Meiss, Chen]
Z = 1 # Charge number

# Mass of species in kg
mass = {'protium': 1.008*Da,
        'deuterium': 2.014*Da,
        }

def xi_factor(tau,ion_species):
    return 7*(1+tau)**1.5*(me/mass[ion_species])**0.5/(3.2*2)

# This is the constant 3.2/(\kappa_{0,e}\sqrt{me}) in units of eV^2 m^2
# It is multiplied by n_e L/T_e^2 with T_e in eV to get the dimensionless collisonality.
coll_const = lnLambdaC*Z**2/(6*2**0.5*pi**1.5*eps0**2)
