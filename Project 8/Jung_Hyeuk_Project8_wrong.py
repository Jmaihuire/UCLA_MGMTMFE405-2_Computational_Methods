#%%
#
# Jung_Hyeuk_Project8.py
# MGMTMFE405-2 Computational Methods in Finance - Project 8
# Hyeuk Jung (005259560)
#

import random
import time
import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

random.seed(int(time.time()))
start = time.time()

##### Q1. Vasicek Model -------------------------------------------------------
def vasicek_r(r0, sd, T, N, r_bar, kappa, dt, steps):
    rates = zeros((N, steps+1))
    rates[:, 0] = r0
    z = random.normal(0, 1, N*steps).reshape(N, steps)
    dW = sqrt(dt)*z
    # r_tk = k*(r_bar - r_tk-1)*dt + sd*sqrt(r_tk-1)*dW(sqrt(dt)*z)
    for i in range(1, steps+1, 1):
        rates[:, i] = rates[:, i-1] + kappa*(r_bar - rates[:, i-1])*dt + sd*dW[:, i-1]

    return rates #np.mean(rates, axis = 0) # not sure 1 path or the entire paths?

def pureBondPrice(rates, t, T, L, dt, N): # t: in case we are not discounting to 0
    T_index = int(T/dt) # ex. T = 1, t = 0.3 discount time = 0.7 (T/dt- t/dt)
    t_index = int(t/dt)
    #bonds = mean(L * exp(-dt * sum(rates[:, t_index:T_index], axis = 1)))
    bonds = L * exp(-dt * sum(rates[:, t_index:T_index], axis = 1))

    return bonds

def couponBondPrice(rates, t, T, L, dt, N, coupon, num): 
    # num = number of coupon payments in a year

    # initialize the payments, including the par value
    coupon = [coupon]*(T*num -1) + [coupon + L] #print(coupon)
    #time_interval = 1/num
    coupon_time = arange(1/num, T + 1/num, 1/num) #print(coupon_time)

    # Assume each coupon is a pure bond
    couponbond = [ mean(pureBondPrice(rates, t, time, c, dt, N)) for time, c in zip(coupon_time, coupon)]

    return sum(couponbond)


def callOption_explicit(r0, sd, kappa, r_bar, t, T, L, r_func, N, steps, K, S):
    dt = T/steps
    call_steps = int(S*365)

    # Get rates between 0 and the call's maturity
    rates = r_func(r0, sd, S, N, r_bar, kappa, dt, call_steps) 
    
    # Calculate components of the explicit formula
    B = 1/kappa*(1 - exp(-kappa*(T-S)))
    A = exp( (r_bar - sd*sd/(2*kappa*kappa)) * (B - (T-S)) - sd*sd/(4*kappa)*B*B )
    purebond = L*A*exp(-B*rates[:, -1])

    # Calculate call option payoff
    payoff = maximum(0, purebond - K)
    print('1.c Average payoff:', mean(payoff))
    call = mean(payoff * exp(-dt * sum(rates, axis = 1)))
    
    return call



def callOption_bs(r0, sd, kappa, r_bar, t, T, L, r_func, b_func, N, steps, coupon, num, K, S):
    dt = T/steps
    call_steps = int(S*365)

    rates_bond = r_func(r0, sd, T, N, r_bar, kappa, dt, steps)
    rates_call = r_func(r0, sd, T, N, r_bar, kappa, dt, call_steps)
    couponbond = b_func(rates_bond, S, T, L, dt, N, coupon, num)

    payoff = maximum(0, couponbond - K)
    call = mean(payoff * exp(-dt * sum(rates_call, axis = 1)))

    return call


# Bond and Option Price
def Price_vasicek(r0, sd, kappa, r_bar, t, T, L, r_func, b_func, N, steps, coupon, num, K, S):
    # N = number of paths
    # L = par value of the bond
    # T = maturity of the bond
    dt = T/steps
    rates = r_func(r0, sd, T, N, r_bar, kappa, dt, steps)

    # Bonds only
    if coupon == 0:
        bond = mean(b_func(rates, t, T, L, dt, N))
    else:
        bond = b_func(rates, t, T, L, dt, N, coupon, num)
    
    return bond


r0 = 0.05; sd = 0.18; kappa = 0.82; r_bar = 0.05; L = 1000; t = 0; T = 0.5; N = 50000; steps = int(365*T)

# 1.(a) Pure discount bond
purebond = Price_vasicek(r0, sd, kappa, r_bar, t, T, L, vasicek_r, pureBondPrice, N, steps, 0, 0, 0, 0)
print("1.(a). Pure Discount Bond Value =", purebond)

# 1.(b) Coupon paying bond
T = 4; coupon = 30; num = 2; steps = int(365*T)
couponbond = Price_vasicek(r0, sd, kappa, r_bar, t, T, L, vasicek_r, couponBondPrice, N, steps, coupon, num, 0, 0)
print("1.(b). Coupon Paying Bond Value =", couponbond)

# 1.(c) Call option on pure discount bond, using explicit formula for the underlying
r0 = 0.05; sd = 0.18; kappa = 0.82; r_bar = 0.05; L = 1000; t = 0; T = 0.5; N = 50000; steps = int(365*T)
K = 980; S = 3/12
purebond, purebond_call = Price_vasicek(r0, sd, kappa, r_bar, t, T, L, vasicek_r, pureBondPrice, N, steps, 0, 0, K, S)
print("1.(c). Call on Pure Discount Bond Value =", purebond_call)
#print("cf. Explicit Pure bond value =", purebond)

r0 = 0.05; sd = 0.18; kappa = 0.82; r_bar = 0.05; L = 1000; t = 0; T = 4; N = 50000; steps = int(365*T)
T = 4; coupon = 30; num = 2
K = 980; S = 3/12
couponbond, couponbond_call = Price_vasicek(r0, sd, kappa, r_bar, t, T, L, vasicek_r, couponBondPrice, N, steps, coupon, num, K, S)
print("1.(d). Call on Coupon-Paying Discount Bond Value =", couponbond_call)


#%%
##### Q2. CIR Model -----------------------------------------------------------
def cir_r(r0, sd, T, N, r_bar, kappa, dt, steps):
    rates = zeros((N, steps+1))
    rates[:, 0] = r0
    z = random.normal(0, 1, N*steps).reshape(N, steps)
    dW = sqrt(dt)*z
    for i in range(1, steps+1, 1):
        rates[:, i] = rates[:, i-1] + kappa*(r_bar - rates[:, i-1])*dt + sd*sqrt(rates[:, i-1])*dW[:, i-1]

    return rates 

# Bond and Option Price
def cirPrice_mc(r0, sd, kappa, r_bar, t, S, L, r_func, b_func, N, steps, coupon, num, K, T):
    # N = number of paths
    # L = par value of the bond
    # T = maturity of the option
    # S = maturity of the bond
    dt = T/steps
    rates = r_func(r0, sd, S, N, r_bar, kappa, dt, steps)

    # Bonds only
    if coupon == 0:
        bond = b_func(rates, t, S, L, dt, N)
    else:
        bond = b_func(rates, t, S, L, dt, N, coupon, num)

    # European call on the pure bond
    if coupon == 0 and K != 0: 
        # Get rates between 0 and the call's maturity
        steps = int(T*365)
        rates_call = r_func(r0, sd, T, N, r_bar, kappa, dt, steps)

        payoff = maximum(0, bond - K)
        call = mean(payoff * exp(-dt * sum(rates_call, axis = 1)))

        return(mean(bond), call)
    
    # European call on the coupon paying bond
    elif coupon != 0 and K != 0: 
        # Get rates between 0 and the call's maturity
        steps = int(T*365)
        rates_call = r_func(r0, sd, T, N, r_bar, kappa, dt, steps)
        
        # Calculate call option payoff
        payoff = maximum(0, bond - K)
        call = mean(payoff * exp(-dt * sum(rates_call, axis = 1)))

        return(bond, call)
    
    return bond

def cirPrice_explicit(r0, sd, kappa, r_bar, t, S, L, r_func, N, steps, K, T):
    # S = bond maturity
    # T = option maturity
    # t = 0 
    dt = T/steps

    # rates for discounting call price to time 0; 0 ~ T
    rates_call = r_func(r0, sd, T, N, r_bar, kappa, dt, steps)

    # Calculate components of the explicit formula
    h1 = sqrt(kappa*kappa + 2*sd*sd)
    h2 = (kappa + h1) / 2
    h3 = (2*kappa*r_bar) / (sd*sd)

    # A(T, S) and B(T, S)
    A = ( h1*exp(h2*(S - T)) / (h2*(exp(h1*(S-T)) - 1) + h1) )**h3 # S = bond maturity, T = option expiration time
    B = (exp(h1*(S-T)) - 1) / (h2*( exp(h1*(S-T)) -1 ) + h1)

    purebond = L*A*exp(-B*rates_call[:, -1])

    # Calculate call option payoff
    payoff = maximum(0, purebond - K)
    call = mean(payoff * exp(-dt * sum(rates_call, axis = 1)))

    return(mean(purebond), call)


r0 = 0.05; sd = 0.18; kappa = 0.92; r_bar = 0.055; t = 0; steps = int(365*S); N = 50000
K = 980; T = 0.5 # T = call option expiration
L = 1000; S = 1 # S = bond maturity

bond_1, call_1 = cirPrice_mc(r0, sd, kappa, r_bar, t, S, L, cir_r, pureBondPrice, N, steps, 0, 0, K, T)
print('2.(a). European call value using MC (under CIR)', bond_1, call_1)

bond_2, call_2 = cirPrice_explicit(r0, sd, kappa, r_bar, t, S, L, cir_r, N, steps, K, T)
print('2.(b). European call value using explicit formula (under CIR)', bond_2, call_2)

total_execution = time.time() - start
print('Total execution time:', total_execution)




#%%
    # European call on the pure bond
    if coupon == 0 and K != 0: 
        # Get rates between 0 and the call's maturity
        steps = int(S*365)
        rates = r_func(r0, sd, S, N, r_bar, kappa, dt, steps)
        
        # Calculate components of the explicit formula
        B = 1/kappa*(1 - exp(-kappa*(T-S)))
        A = exp( (r_bar - sd*sd/(2*kappa*kappa)) * (B - (T-S)) - sd*sd/(4*kappa)*B*B )
        purebond = L*A*exp(-B*rates[:, -1])

        # Calculate call option payoff
        payoff = maximum(0, purebond - K)
        print('1.c Average payoff:', mean(payoff))
        call = mean(payoff * exp(-dt * sum(rates, axis = 1)))

        return(mean(purebond), call)
    
    # European call on the coupon paying bond
    elif coupon != 0 and K != 0: 
        # Get rates between 0 and the call's maturity
        steps = int(S*365)
        rates = r_func(r0, sd, S, N, r_bar, kappa, dt, steps)
        couponbond = b_func(rates, t, S,)

        # Calculate call option payoff
        payoff = maximum(0, bond - K)
        call = mean(payoff * exp(-dt * sum(rates, axis = 1)))

        return(bond, call)