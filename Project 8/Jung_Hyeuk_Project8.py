#%%
#
# Jung_Hyeuk_Project8.py
# MGMTMFE405-2 Computational Methods in Finance - Project 8
# Hyeuk Jung (005259560)
#

import random
import time
from numpy import *

#random.seed(int(time.time()))
random.seed(12345)
start = time.time()


#%%
##### Q1. Vasicek Model -------------------------------------------------------
def vasicek_r(r0, sd, t, T, N, r_bar, kappa, dt, steps):
    # set the step size
    T_index = int(T/dt)
    t_index = int(t/dt)
    steps = T_index - t_index

    # Using antithetic variates method
    rates = zeros((N, steps+1))
    rates[:, 0] = r0
    z = random.normal(0, 1, int(N/2)*steps).reshape(int(N/2), steps)
    w = sqrt(dt)*z
    w_anti = sqrt(dt)*(-1)*z
    dW = concatenate((w, w_anti), axis=0)
    for i in range(1, steps+1, 1):
        rates[:, i] = rates[:, i-1] + kappa*(r_bar - rates[:, i-1])*dt + sd*dW[:, i-1]

    return rates

def pureBondPrice(r0, sd, kappa, r_bar, t, T, L, r_func, N, steps): # t: in case we are not discounting to 0
    dt = T/steps
    rates = r_func(r0, sd, t, T, N, r_bar, kappa, dt, steps)
    bonds = L * exp(-dt * sum(rates, axis = 1))

    return bonds

def couponBondPrice(r0, sd, kappa, r_bar, t, T, L, r_func, N, steps, coupon, num): 
    dt = T/steps
    # num = number of coupon payments in a year

    # initialize the payments, including the par value
    coupon = [coupon]*(T*num -1) + [coupon + L] #time_interval = 1/num
    coupon_time = arange(1/num, T + 1/num, 1/num) #print(coupon_time)

    # Assume each coupon is a pure bond
    couponbond = [ mean(pureBondPrice(r0, sd, kappa, r_bar, t, time, c, r_func, N, steps)) for time, c in zip(coupon_time, coupon)]

    return sum(couponbond)

def call_vasicek_explicit(r0, sd, kappa, r_bar, t, T, L, r_func, N, steps, K, S):
    dt = T/steps
    call_steps = int(S*365)

    # Get rates between 0 and the call's maturity
    rates = r_func(r0, sd, t, S, N, r_bar, kappa, dt, call_steps) 
    
    # Calculate components of the explicit formula
    B = 1/kappa*(1 - exp(-kappa*(T-S)))
    A = exp( (r_bar - sd*sd/(2*kappa*kappa)) * (B - (T-S)) - sd*sd/(4*kappa)*B*B )
    purebond = L*A*exp(-B*rates[:, -1])

    # Calculate call option payoff
    payoff = maximum(0, purebond - K)
    call = mean(payoff * exp(-dt * sum(rates, axis = 1)))
    
    return call

def call_vasicek_mc(r0, sd, kappa, r_bar, t, T, L, r_func, b_func, N, steps, coupon, num, K, S):
    dt = T/steps
    call_steps = int(S*365)

    # Make interest rate paths from 0 to T
    rates_call = r_func(r0, sd, t, T, N, r_bar, kappa, dt, call_steps)
    
    # Using the last path of the interest rate path, extend it to T (paths from T to S)
    couponbond = b_func(rates_call[:, -1], sd, kappa, r_bar, S, T, L, r_func, N, steps, coupon, num)
    payoff = maximum(0, couponbond - K) 
    #couponbond = [ mean(b_func(i, sd, kappa, r_bar, S, T, L, r_func, N, steps, coupon, num)) for i in rates_call[:, -1] ] 
    #payoff = maximum(0, asarray(couponbond) - K) 

    call = mean(payoff * exp(-dt * sum(rates_call, axis = 1)))

    return call


r0 = 0.05; sd = 0.18; kappa = 0.82; r_bar = 0.05; L = 1000; t = 0; T = 0.5; N = 100000; steps = int(365*T)

# 1.(a) Pure discount bond
purebond = mean(pureBondPrice(r0, sd, kappa, r_bar, t, T, L, vasicek_r, N, steps))
print("1.(a). Pure Discount Bond Value =", round(purebond, 5))

# 1.(b) Coupon paying bond
T = 4; coupon = 30; num = 2#; steps = int(365*T)
couponbond = couponBondPrice(r0, sd, kappa, r_bar, t, T, L, vasicek_r, N, steps, coupon, num)
print("1.(b). Coupon Paying Bond Value =", round(couponbond, 5))

# 1.(c) Call option on pure discount bond, using explicit formula for the underlying
r0 = 0.05; sd = 0.18; kappa = 0.82; r_bar = 0.05; L = 1000; t = 0; T = 0.5; N = 100000; steps = int(365*T)
K = 980; S = 3/12
purebond_call = call_vasicek_explicit(r0, sd, kappa, r_bar, t, T, L, vasicek_r, N, steps, K, S)
print("1.(c). European Call on Pure Discount Bond Value:", round(purebond_call, 5))

r0 = 0.05; sd = 0.18; kappa = 0.82; r_bar = 0.05; L = 1000; t = 0; T = 4; N = 1000; steps = int(365*T)
T = 4; coupon = 30; num = 2
K = 980; S = 3/12
couponbond_call = call_vasicek_mc(r0, sd, kappa, r_bar, t, T, L, vasicek_r, couponBondPrice, N, steps, coupon, num, K, S)
print("1.(d). European Call on Coupon-Paying Discount Bond Value:", round(couponbond_call, 5))


#%%
##### Q2. CIR Model -----------------------------------------------------------
def cir_r(r0, sd, t, T, N, r_bar, kappa, dt, steps):
    # set the step size
    T_index = int(T/dt)
    t_index = int(t/dt)
    steps = T_index - t_index

    rates = zeros((N, steps+1))
    rates[:, 0] = r0
    dW = sqrt(dt)*random.normal(0, 1, int(N/2)*steps).reshape(int(N/2), steps) #dW = sqrt(dt)*z
    dW = concatenate((dW, -dW), axis=0)
    
    # using full truncation
    for i in range(1, steps+1, 1):
        rates[:, i] = maximum(rates[:, i-1], 0) + kappa*(r_bar - maximum(rates[:, i-1], 0))*dt + sd*sqrt(maximum(rates[:, i-1], 0))*dW[:, i-1]

    return rates 

def call_cir_mc(r0, sd, kappa, r_bar, t, S, L, r_func, b_func, N, steps, K, T):
    # N = number of paths / L = par value of the bond / T = maturity of the option / S = maturity of the bond    
    dt = S/steps
    call_steps = int(T*365)

    # Make interest rate paths from 0 to T
    rates_call = r_func(r0, sd, t, T, N, r_bar, kappa, dt, call_steps)
    
    # Generate pure bond values from T to S
    purebond = [mean(b_func(i, sd, kappa, r_bar, T, S, L, r_func, N, steps)) for i in rates_call[:, -1]]
    payoff = maximum(0, asarray(purebond) - K)

    call = mean(payoff * exp(-dt * sum(rates_call, axis = 1)))

    return call

# chi-square cdf -> cdf(x, df, loc=0, scale=1)
def call_cir_explicit(r0, sd, kappa, r_bar, t, S, L, r_func, N, steps, K, T):
    # S = bond maturity
    # T = option maturity
    # t = 0 
    dt = S/steps
    call_steps = int(T*365)

    # rates for discounting call price to time 0; 0 ~ T
    rates_call = r_func(r0, sd, t, T, N, r_bar, kappa, dt, call_steps)

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

    return(call)


r0 = 0.05; sd = 0.18; kappa = 0.92; r_bar = 0.055; t = 0; steps = int(365*S); N = 100000
K = 980; T = 0.5 # T = call option expiration
L = 1000; S = 1 # S = bond maturity

N = 1000
purebond_call_cir = call_cir_mc(r0, sd, kappa, r_bar, t, S, L, cir_r, pureBondPrice, N, steps, K, T)
print('2.(a). European call value using MC (under CIR):', round(purebond_call_cir, 5))

N = 100000
call_2 = call_cir_explicit(r0, sd, kappa, r_bar, t, S, L, cir_r, N, steps, K, T)
print('2.(b). European call value using explicit formula (under CIR):', round(call_2, 5))

print('Q2 comment:\n Both Monte Carlo simulation and the explicit formula gives similar call option value.\n')


#%%
##### Q3. G2++ Model ----------------------------------------------------------
def g2plusplus(x0, y0, a, b, sd, eta, rho, phi0, phi_t, dt, N, steps, t, T):
    T_index = int(T/dt)
    t_index = int(t/dt)
    steps = T_index - t_index

    x = zeros((N, steps+1))
    y = zeros((N, steps+1))
    r = zeros((N, steps+1))
    x[:, 0] = x0
    y[:, 0] = y0
    r[:, 0] = x0 + y0 + phi0

    dW_1 = sqrt(dt)*random.normal(0, 1, int(N/2)*steps).reshape(int(N/2), steps)
    dW_1 = concatenate((dW_1, -dW_1), axis = 0)
    dW_2 = sqrt(dt)*random.normal(0, 1, int(N/2)*steps).reshape(int(N/2), steps)
    dW_2 = concatenate((dW_2, -dW_2), axis = 0)

    #dW = sqrt(dt)*z
    for i in range(1, steps+1, 1):
        x[:, i] = x[:, i-1] - a*x[:, i-1]*dt + sd*dW_1[:, i-1]
        y[:, i] = y[:, i-1] - b*y[:, i-1]*dt + eta*(rho*dW_1[:, i-1] + sqrt(1-rho*rho)*dW_2[:, i-1])
        r[:, i] = x[:, i] + y[:, i] + phi_t

    return(r, x, y)

def put_g2plusplus_mc(x0, y0, a, b, sd, eta, rho, phi0, phi_t, N, steps, L, S, K, T, r_func, t, b_func):
    # x0, y0, r0, a, b, sd, eta, rho, phi0, phi_t, dt, N, steps: int rate info
    # L, S: pure discount bond information
    # K, T: put option info

    dt = S/steps

    # Put option discount factor at time 0 ~ T
    put_steps = int(T*365)
    rates_put, x_put, y_put = r_func(x0, y0, a, b, sd, eta, rho, phi0, phi_t, dt, N, put_steps, t, T)

    # Generate pure bond values at time T (put option exercising moment)
    purebond = [mean(L * exp(-dt*sum(r_func(x_put[i, -1], y_put[i, -1], a, b, sd, eta, rho, phi0, phi_t, dt, N, steps, T, S)[0], axis = 1))) for i in arange(0, N, 1)]

    payoff = maximum(0, K - asarray(purebond))
    put = mean(payoff * exp(-dt * sum(rates_put, axis = 1)))

    return put


N = 1000
x0 = y0 = 0; phi0 = r0 = 0.03; rho = 0.7; a = 0.1; b = 0.3; sd = 0.03; eta = 0.08; phi_t = 0.03 
t = 0
K = 985; T = 0.5 # Put option
L = 1000; S = 1 # Pure discount bond
steps = int(S*365)

purebond_put_g2 = put_g2plusplus_mc(x0, y0, a, b, sd, eta, rho, phi0, phi_t, N, steps, L, S, K, T, g2plusplus, t, pureBondPrice)
print('3. European put value using MC (under G2++):', purebond_put_g2)

total_execution = time.time() - start
print('Total execution time:', total_execution)

