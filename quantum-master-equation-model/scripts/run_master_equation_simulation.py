"""
Self-consistent quantum master-equation simulation for a reaction-coordinate
polariton condensate.

The model couples the polariton density matrix to reservoir density and energy
dynamics. It includes interpolated scattering rates, GPU-accelerated Liouville-
space evolution, comparison with a rate-equation reference model, and
calculation of correlation functions and emission spectra.

Developed during PhD in Electrical and Photonics Engeneering.
'''


'''
Master equation for reaction coordinate polariton condensate. 
QuTiP is used for setting up the Hilbert space and operators. The standard functionality for solving the master equation with
QuTiP cannot be used, because the reservoir density and temperature/energy must be solved self-consistently.
Emission spectrum from the two-time average <p^\dagger(t+tau) p(t)>.
'''

import numpy as np
from scipy.integrate import solve_ivp
import qutip as qt
import parameters_ode
import pickle
from scipy.interpolate import PchipInterpolator
import torch
import matplotlib.pyplot as plt
################################################################################
# CONSTANTS AND PARAMETERS
# all units are in SI
device = 'cpu'
parameter_obj = parameters_ode.material_parameters()
hbar= parameter_obj.h_bar
kB = parameter_obj.k_B
e_LP = parameter_obj.e_LP
rho_x = parameter_obj.rho_x
SW = parameter_obj.SW
W_LP = parameter_obj.W_0

TL = parameter_obj.T_l
S = parameter_obj.surface
gamma_LP = parameter_obj.gamma_LP
gamma_x = parameter_obj.gamma_x

################################################################################
# SCATTERING RATES AND INTERPOLATION
#
# load temperature range and pre-calculated W_in and W_out ranges.

f_in = open('W_sweep_20nm/temperature_W_in.p', 'rb')
W_in = pickle.load(f_in)
f_in.close()


f_out_1 = open('W_sweep_20nm/temperature_W_out_1.p', 'rb')
W_out_1 = pickle.load(f_out_1)
f_out_1.close()


f_out_2 = open('W_sweep_20nm/temperature_W_out_2.p', 'rb')
W_out_2 = pickle.load(f_out_2)
f_out_2.close()

f_in_correction = open('W_sweep_20nm/temperature_W_in_correction.p', 'rb')
W_in_correction = pickle.load(f_in_correction)
f_in_correction.close()

f_out_correction = open('W_sweep_20nm/temperature_W_out_correction.p', 'rb')
W_out_correction = pickle.load(f_out_correction)
f_out_correction.close()


Tx_data = np.linspace(1,80,18)


# make interpolating functions
W_in_data = np.array(W_in[1])
W_out_1_data = np.array(W_out_1[1])
W_out_2_data = np.array(W_out_2[1])
W_out_correction_data = np.array(W_out_correction[1])
W_in_correction_data = np.array(W_in_correction[1])

W_in_interp_func = PchipInterpolator(Tx_data, W_in_data)
W_out_1_interp_func = PchipInterpolator(Tx_data, W_out_1_data)
W_out_2_interp_func = PchipInterpolator(Tx_data, W_out_2_data)
W_out_correction_interp_func = PchipInterpolator(Tx_data, W_out_correction_data)
W_in_correction_interp_func = PchipInterpolator(Tx_data, W_in_correction_data)

################################################################################
# HILBERT SPACE AND LIOUVILLE SPACE OPERATORS
#
# set up operators for master equation

n_max = parameter_obj.n_max     # maximum number of particles in the polariton state is n_max-1
p = qt.destroy(n_max)           # polariton annihilation operator
psi_init = qt.basis(n_max,0)    # creates the vacuum state
rho_init = qt.ket2dm(psi_init)  # make corresponding initial density operator


# The master equation can be written as
# d(rho)/dt = L0[rho] + G_in L_in[rho] + (G_out + gamma_LP) L_out[rho],
# where L0, L_in and L_out are superoperators and G_in = nx^2 * W_in,G_in_c = nx^3 * W_in_c, 
# G_out_1 = nx * W_out_1,G_out_2 = nx^2 * W_out_2,G_out_c = nx^2 * W_out_c
# The superoperators are be defined here
# the function .full() returns the underlying array data from a qutip Qobj object.

L0 = 1j/hbar*W_LP*(qt.spost(p.dag()*p.dag()*p*p) - qt.spre(p.dag()*p.dag()*p*p)).full()
L_in = qt.superoperator.lindblad_dissipator(p.dag()).full()
L_out = qt.superoperator.lindblad_dissipator(p).full()

#convert to pythorch
L0_gpu = torch.from_numpy(L0).to(device)
L_in_gpu = torch.from_numpy(L_in).to(device)
L_out_gpu = torch.from_numpy(L_out).to(device)


# when solving the master equation and the differential equations for nx and ex,
# it is practical to represent nx, ex and rho as one combined vector, x.
# The density operator is vectorized. f the density operator dimensions
# are n x n, the the vector will be of length n^2.
# All superoperators that work on the density operator can then be represented
# as n^2 x n^2 matrices in Liouville space. The way that L0, L_in and L_out are
# created above is actually already in the correct Liouville space structure, so
# here, we just need to vectorize the initial density operator
rho_init = qt.operator_to_vector(rho_init).full()

# in addition, the operators for calculating expectation values such as
# <p> = Tr[p rho] and so on are:
p_pre = qt.spre(p).full() # pre-multiplication by p
p_dag_pre = qt.spre(p.dag()).full()
p_dag_post = qt.spost(p.dag()).full()
n0_pre = qt.spre(p.dag()*p).full()
trace_vec = qt.operator_to_vector(qt.qeye(n_max)).full()

# Tr[p rho] can now be evaluated in Liouville space like
Tr_P_rho_init = trace_vec.T@p_pre@rho_init # @ denotes is standard matrix multiplication

# and Tr[p p^\dagger rho] can now be evaluated in Liouville space like
Tr_PPdag_rho_init = trace_vec.T@p_pre@p_dag_pre@rho_init

################################################################################
# SELF-CONSISTENT MASTER EQUATION
#
# Set up the self-consistent master equation
# The total state vector x has 2+n_max^2 elements.
# The first element x[0] is nx, the second element x[1] is ex and the remaining
# n_max^2 elements is the vectorized density operator. x0 that defines the initial cond. is than converted to pythorch
x0 = np.zeros((2+n_max**2),dtype='complex64')
x0_gpu = torch.zeros(2 + n_max**2, dtype=torch.complex64)
output_gpu =  torch.zeros_like(x0_gpu)
x0_gpu = None

def dxdt(t,x,px):
    # unpack x
    nx = np.real(x[0])
    ex = np.real(x[1])
    rho = x[2:]
    N0 = np.real(trace_vec.T@n0_pre@rho)[0]
    rho_gpu = torch.from_numpy(rho).to(device)
    
    if nx==0:
        Tx = TL
    else:
        Tx = ex/(kB*nx)
    
    if Tx < np.min(Tx_data):
        
        W_in = W_in_interp_func(np.min(Tx_data))
        W_out_1 = W_out_1_interp_func(np.min(Tx_data))
        W_out_2 = W_out_2_interp_func(np.min(Tx_data))
        #W_out_3 = W_out_3_interp_func(np.min(Tx_data))
        W_in_correction = W_in_correction_interp_func(np.min(Tx_data))
        W_out_correction = W_out_correction_interp_func(np.min(Tx_data))

    elif Tx > np.max(Tx_data):
        #print(Tx)
        W_in = W_in_interp_func(np.max(Tx_data))
        W_out_1 = W_out_1_interp_func(np.max(Tx_data))
        W_out_2 = W_out_2_interp_func(np.max(Tx_data))
        #W_out_3 = W_out_3_interp_func(np.max(Tx_data))
        W_in_correction = W_in_correction_interp_func(np.max(Tx_data))
        W_out_correction = W_out_correction_interp_func(np.max(Tx_data))

    else:
        W_in = W_in_interp_func(Tx)
        W_out_1 = W_out_1_interp_func(Tx)
        W_out_2 = W_out_2_interp_func(Tx)
       # W_out_3 = W_out_2_interp_func(Tx)
        W_in_correction = W_in_correction_interp_func(Tx)
        W_out_correction = W_out_correction_interp_func(Tx)


    
    drhodt =  (W_in*nx**2+W_in_correction*nx**3)*torch.matmul(L_in_gpu,rho_gpu) + (W_out_1*nx +W_in_correction*nx**3+W_out_2*nx**2+W_out_correction*nx**2+gamma_LP)*torch.matmul(L_out_gpu,rho_gpu)+ torch.matmul(L0_gpu,rho_gpu) 
    dnxdt = - 1/S*(W_in*nx**2*(1+N0) +W_in_correction*nx**3- ((W_out_1*nx+W_out_2*nx**2+W_out_correction*nx**2)*N0)) - gamma_x*nx + px
    dexdt = - e_LP/S*(W_in*nx**2*(1+N0) +W_in_correction*nx**3- ((W_out_1*nx+W_out_2*nx**2+W_out_correction*nx**2)*N0)) + px*kB*TL - gamma_x*nx*kB*Tx #- gamma_thermalization*(ex-kB*TL*nx)

    output_gpu[0] = dnxdt
    output_gpu[1] = dexdt
    output_gpu[2:] = drhodt
    return output_gpu.cpu().numpy()

################################################################################
# RATE EQUATION FOR REFERENCE CALCULATION
#
# for testing, reference calculation with the rate-equation ODE

def dxdt_RATE_EQ(t,x,px):
    N0 = x[0]
    nx = x[1]
    ex = x[2]

    if nx==0:
        Tx = TL
    else:
        Tx = ex/(kB*nx)

    W_in = W_in_interp_func(Tx)
    W_out_1 = W_out_1_interp_func(Tx)
    W_out_2 = W_out_2_interp_func(Tx)
    W_in_correction = W_in_correction_interp_func(Tx)
    W_out_correction = W_out_correction_interp_func(Tx)

    dN0dt = W_in*nx**2*(1+N0) +W_in_correction*nx**3- ((W_out_1*nx+W_out_2*nx**2+W_out_correction*nx**2)*N0)- gamma_LP*N0
    dnxdt = - 1/S*(W_in*nx**2*(1+N0) +W_in_correction*nx**3- ((W_out_1*nx+W_out_2*nx**2+W_out_correction*nx**2)*N0)) - gamma_x*nx + px
    dexdt = - e_LP/S*(W_in*nx**2*(1+N0) +W_in_correction*nx**3- ((W_out_1*nx+W_out_2*nx**2+W_out_correction*nx**2)*N0)) + px*kB*TL - gamma_x*nx*kB*Tx #- gamma_thermalization*(ex-kB*TL*nx)

    return np.array([dN0dt, dnxdt, dexdt])



################################################################################
# SOLVING THE SELF-CONSISTENT MASTER EQUATION

x0[0] = 0
x0[1] = kB*TL/S
x0[2:] = rho_init[:,0]  # qt.operator_to_vector creates an array with shape [n_max^2, 1]
t_begin = 0             # initial time
t_end = 1e-9            # end time
time_points = np.linspace(t_begin,t_end, 1000)
px = parameter_obj.px # pumping rate density (m^-2s^-1)

# Run ode-solver
out = solve_ivp(dxdt, [t_begin, t_end], x0, args=(px,), method='RK45', t_eval=time_points, rtol=1e-10, atol=1e-8)

time = out.t
x_steady = out.y[:,-1]
nx_out = np.real(out.y[0,:])
ex_out = np.real(out.y[1,:])
rho_out = out.y[2:,:]

# Precompute operator products
N0_op = trace_vec.T @ n0_pre
m0_op = trace_vec.T @ p_dag_pre @ p_dag_pre @ p_pre @ p_pre

N0_out = np.real((N0_op @ rho_out)[0])
m0_out = np.real((m0_op @ rho_out)[0])
Tx_out = ex_out / (kB * nx_out)


# Staedy-state values for each pumping px 
nx_ME_ss = nx_out[-1]
ex_ME_ss = ex_out[-1]
rho_out_ss = out.y[2:,-1]
N0_ME_ss = N0_out[-1]
m0_out_ss = m0_out[-1]
Tx_ME_ss = ex_ME_ss/(kB*nx_ME_ss)

Gamma_in_ME_ss = W_in_interp_func(Tx_ME_ss)*nx_ME_ss**2
Gamma_out_1_ME_ss = W_out_1_interp_func(Tx_ME_ss)*nx_ME_ss
Gamma_out_2_ME_ss = W_out_2_interp_func(Tx_ME_ss)*nx_ME_ss**2
Gamma_in_corr_ME_ss = W_in_correction_interp_func(Tx_ME_ss)*nx_ME_ss**3
Gamma_out_corr_ME_ss = W_out_correction_interp_func(Tx_ME_ss)*nx_ME_ss**2
W_in = Gamma_in_ME_ss+Gamma_in_corr_ME_ss
W_out = Gamma_out_1_ME_ss+Gamma_out_2_ME_ss+Gamma_in_corr_ME_ss+Gamma_out_corr_ME_ss+ gamma_LP

# Data to save
output_dir = "output_file_ode_ME"

data_to_save = {
    "N0.p": [time, N0_out],
    "m0.p": [time, m0_out],
    "nx.p": [time, nx_out],
    "ex.p": [time, ex_out],
    "W_in.p": [time, W_in],
    "W_out.p": [time, W_out],
}

for filename, data in data_to_save.items():
    with open(f"{output_dir}/{filename}", "wb") as f:
        pickle.dump(data, f)


# Run reference calculation with rate equation
x0 = np.array([0, 0, kB*TL]) # initial value
out_RATE_EQ = solve_ivp(dxdt_RATE_EQ, [t_begin, t_end], x0, args=[px], method='RK45', dense_output=True, rtol=1e-12, atol=1e-12)

time_RATE_EQ = out_RATE_EQ.t
N0_RATE_EQ = out_RATE_EQ.y[0,:]
nx_RATE_EQ = out_RATE_EQ.y[1,:]
ex_RATE_EQ = out_RATE_EQ.y[2,:]


# Plot results and compare rate equation and master equation.
# deviation in N0 between the two must stem from having a too low n_max!!
fig, ax = plt.subplots(1,3,figsize=(12,4))
fig.suptitle('Master Equation vs Rate Equation')
ax[0].plot(time, N0_out, label='master eq')
ax[0].plot(time_RATE_EQ, N0_RATE_EQ, '--', label='rate eq')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('N0')
ax[0].legend()

ax[1].plot(time, nx_out)
ax[1].plot(time_RATE_EQ, nx_RATE_EQ, '--')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel(r'nx [1/m^2]')

ax[2].plot(time, ex_out)
ax[2].plot(time_RATE_EQ, ex_RATE_EQ, '--')
ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel(r'ex [J/m^2]')

plt.tight_layout()
plt.show()


################################################################################
# MULTI-TIME AVERAGES AND THE QUANTUM REGRESSION THEOREM
#
# The self-consistent master equation to evaluate two-time averages.
# Specifically, steady state first-order correlation function (g1), defined as
# g1(tau) = lim(t -> infinity) <p^\dagger(t+tau) p(t)>.
# This can be used to calculate the steady-state emission spectrum through the Wiener-Khinchin theorem.
#
# A general two-time average <A(t+tau)B(t)> can be calculated (under the Markov approximation) as
# Tr[A Lambda(tau,t)], where Lambda(tau,t) is the solution of the master equation
# d[Lambda(tau,t)]/dtau, with the initial condition Lambda(0,t) = B rho(t).
# In the case of a steady-state correlation function, we have
# Lambda(0, t->infinity) = B*rho_ss, where rho_ss is the steady state
# density matrix.



# Here we calculate the steady state by explicitly time-evolving the state vector
# until the change over time is negligible
x0 = np.zeros((2+n_max**2), dtype='complex')
x0[0] = 0
x0[1] = kB*TL
x0[2:] = rho_init[:,0] # for some reason, qt.operator_to_vector creates an array with shape [n_max^2, 1]
t_begin = 0            # initial time
t_end = 2e-9           # end time. from the calculation above, we know that this is sufficient to reach steady state
px = 7e24              

out = solve_ivp(dxdt, [t_begin, t_end], x0, args=[px], method='RK45', dense_output=True, rtol=1e-12, atol=1e-12)
x_steady = out.y[:,-1]

        

# NB I define a new function because this time nx_ss and ex_ss are not evolving over time
def dxdt_ss(t,x,Tx_ss, nx_ss):
    # unpack x
    
    rho = x[:]
    rho_gpu = torch.from_numpy(rho).to(device)
    
    W_in = W_in_interp_func(Tx_ss)
    W_out_1 = W_out_1_interp_func(Tx_ss)
    W_out_2 = W_out_2_interp_func(Tx_ss)
    W_in_correction = W_in_correction_interp_func(Tx_ss)
    W_out_correction = W_out_correction_interp_func(Tx_ss)
    
    drhodt =  (W_in*nx_ss**2+W_in_correction*nx_ss**3)*torch.matmul(L_in_gpu,rho_gpu) + (W_out_1*nx_ss +W_in_correction*nx_ss**3+W_out_2*nx_ss**2+W_out_correction*nx_ss**2+gamma_LP)*torch.matmul(L_out_gpu,rho_gpu)+ torch.matmul(L0_gpu,rho_gpu) 
    return drhodt.cpu().numpy()

# Following the quantum regression theorem, we now create a new initial state
# vector, x2, where x2[:] = p*rho_steady (corresponding to Lambda(0,t))
x2 = np.zeros(x_steady.shape, dtype=complex)
x2[0] = x_steady[0]
x2[1] = x_steady[1]
x2[2:] = p_pre@x_steady[2:]

# set initial and final values for tau
tau_begin = 0
tau_end = 1e-10 # end time
time_points = np.linspace(t_begin,t_end, 10000)
# run solver with x2 as initial condition
out2 = solve_ivp(dxdt_ss, [tau_begin, tau_end], x2, args=[Tx_ME_ss, nx_ME_ss], method='RK45', t_eval=time_points, rtol=1e-12, atol=1e-12)



# unpack output
tau = out2.t
Lambda = out2.y


# calculate correlation function by following the quantum regression theorem, 
# #we include also higher order terms, 4 and 6-operator exp. values
# Calculate correlation functions following the quantum regression theorem

tr_pdag = trace_vec.T @ p_dag_pre

g1 = (tr_pdag @ Lambda)[0, :]

op = tr_pdag @ p_dag_pre @ p_pre
m = (op @ Lambda)[0, :]

op = op @ p_dag_pre @ p_pre
l = (op @ Lambda)[0, :]

op = op @ p_dag_pre @ p_pre
r = (op @ Lambda)[0, :]

op = op @ p_dag_pre @ p_pre
s = (op @ Lambda)[0, :]

op = op @ p_dag_pre @ p_pre
z = (op @ Lambda)[0, :]

op = op @ p_dag_pre @ p_pre
T = (op @ Lambda)[0, :]

op = op @ p_dag_pre @ p_pre
h = (op @ Lambda)[0, :]

# Save results
output_dir = "output_file_ode_ME"

data = {
    "g1.p": [tau, g1],
    "m.p":  [tau, m],
    "l.p":  [tau, l],
    "r.p":  [tau, r],
    "s.p":  [tau, s],
    "z.p":  [tau, z],
    "T.p":  [tau, T],
    "h.p":  [tau, h],
}

for filename, values in data.items():
    with open(f"{output_dir}/{filename}", "wb") as f:
        pickle.dump(values, f)


################################################################################
# Fourier transform correlation function to get emission spectrum
# specify frequency range for evaluation of the spectrum (NB: the frequency is
# relative to the lower polariton frequency e_LP/hbar, because the entire calculation
# with the master equation is performed in a rotating frame with respect to the frequency e_LP/hbar.)
#data_g1 = None

wrng = np.linspace(-1e14, 1e14, 201)

# perform Fourier transform by integrating over tau. 
spec = np.real(np.trapz(g1[:,None]*np.exp(-1j*wrng[None,:]*tau[:,None]), x=tau, axis=0))
spec_RATE_EQ = np.real(np.trapz(N_RATE_EQ[:,None]*np.exp(-1j*wrng[None,:]*tau[:,None]), x=tau, axis=0))

# plot correlation function and emission spectrum.
fig, ax = plt.subplots(1,2,figsize=(12,4))
plt.suptitle('Correlation function and emission spectrum')
ax[0].plot(tau, np.real(g1), label='real part')
ax[0].plot(tau, np.imag(g1), label='imag part')
ax[0].set_xlabel(r'Delay time $\tau$ [s]')
ax[0].set_ylabel(r'First-order corr. func., $g^{(1)}(\tau)$')
ax[0].legend()

ax[1].plot(wrng, spec, label='MASTER EQ')
ax[1].plot(wrng, spec_RATE_EQ, label='RATE EQ')
ax[1].legend()
ax[1].set_xlabel(r'Frequency $\omega - \omega_{LP} \:\mathrm{[rad\:s^{-1}]}$')
ax[1].set_ylabel(r'Emission spectrum, $S(\omega)$')

plt.tight_layout()
plt.show()


################################################################################
# SECOND-ORDER CORRELATION FUNCTION
#
# Here it is calculated the second-order correlation function (G2), which is defined as
# G2(tau) = lim(t->infinity) <p^\dagger(t)p^\dagger(t+tau) p(t+tau)p(t)>.
# 
# This can also be done using the quantum regression theorem. Now we have
# Lambda(0,t->infinity) = p rho_steady p^dagger, and
# g2(tau) = Tr[p^\dagger p Lambda(tau, t->infinity)].
# Following the quantum regression theorem, we now create a new initial state
# vector, x2, where x2[0] = nx_steady, and x2[1] = ex_steady
# and x2[2:] = p*rho_steady (corresponding to Lambda(0,t))
x2 = np.zeros(x_steady.shape, dtype=complex)
x2[0] = x_steady[0]
x2[1] = x_steady[1]
x2[2:] = p_dag_post@p_pre@x_steady[2:] 

# set initial and final values for tau
tau_begin = 0
tau_end = 1e-10

# run solver with x2 as initial condition
out2 = solve_ivp(dxdt, [tau_begin, tau_end], x2, args=[px], method='RK45', dense_output=True, rtol=1e-12, atol=1e-12)

# unpack output
tau = out2.t
Lambda = out2.y[2:]

# calculate correlation function by following the quantum regression theorem
G2 = (trace_vec.T@n0_pre@Lambda)[0,:]

# the quantity of interest is often the normalised G2-function (g2):
N0 = np.real(trace_vec.T@n0_pre@x_steady[2:])[0]
g2 = G2/N0**2
#data_g2 = []
#data_g2 = [tau,g2]
#f_N0 = open('output_file/g2_'+str(px)+'.p', 'wb')
#pickle.dump(data_g2, f_N0)
#f_N0.close()

plt.figure()
plt.plot(tau, g2)
plt.xlabel(r'$\tau$ [s]')
plt.ylabel(r'Second-order corr. func. $g^{(2)}(\tau)$')
plt.tight_layout()
plt.show()

################################################################################
# PARTICLE NUMBER DISTRIBUTION
#
# steady-state probability distribution of the number of particles in the lower polariton
# In the standard density operator formalism, the probability of having n particles
# in the lower polariton state is p_n = <n|rho|n>. In the vectorized Liouville-space
# formalism, we need to vectorize the projection operator |n><n|, multiply it onto the vectorized
# steady-state density operator and perform the trace operation:

n_projector_list = [qt.spre(qt.basis(n_max,n).proj()).full() for n in range(n_max)]
p_n = np.array([(trace_vec.T@n_projector@x_steady[2:])[0] for n_projector in n_projector_list])
plt.figure()
#data_bar = []
#data_bar = [n_max,p_n]
#f_N0 = open('output_file/distribution_'+str(px)+'.p', 'wb')
#pickle.dump(data_bar, f_N0)
#f_N0.close()

plt.bar(np.arange(n_max),p_n)
plt.xlabel(r'Particle number, $n$')
plt.ylabel(r'Probability, $p_n$')
plt.gca().set_yscale('log')

plt.show()


