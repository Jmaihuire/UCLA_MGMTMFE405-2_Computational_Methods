import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

#1. Expected values and probabilities
X0 = 1; Y0 = 3/4
n = 1000; m = 10000
seed = time.time()
def Y(t, Y0, n, m):
    dt = t / n; Y=[]
    for j in range(m):
        Y_i=[Y0]
        Z = np.random.normal(0, 1, n)
        Y_i = [((2/(1+(i+1)*dt))*Y_i[-1] + (1+((i+1)*dt)**3)/3) * dt + 
                (1+((i+1)*dt)**3)/3 * dt**0.5 * Z[i] for i in range(n)]
        Y.append(sum(Y_i))
    return Y
def X(t, X0, n, m):
    dt = t / n; X=[]
    for j in range(m):
        X_i=[X0]
        W = np.random.normal(0, 1, n)
        X_i = [(1/5 - 1/2 * X_i[-1]) * dt + 2/3 * dt**0.5 * W[i] for i in range(n)]
        X.append(sum(X_i))
    return X
#Pr(Y_2 > 5)
Y_2 = Y(2, Y0, n, m)
Prob = np.mean(np.array(Y_2)>5)

#E[X^(1/3)_2]
X0 = 1; dt = 2 / n; X_2 = []
X_2 = X(2, X0, n, m)
E1 = 0
for x in X_2:
    if x >= 0:
        E1 += x**(1/3) / m
    else:
        E1 += -((-x)**(1/3)) / m

#E[Y_3]
Y_3 = Y(3, Y0, n, m)
E2 = np.mean(Y_3)

#E[X_2*Y_2*1(X_2>1)]
E3 = np.mean(np.array(X_2) * np.array(Y_2) * (np.array(X_2) > 1))

Result_1 = pd.DataFrame({
        "Values" : [Prob, E1, E2, E3]},
    index = ["Prob", "E1", "E2", "E3"])
print("Inputs - seed:", seed)
print("Outputs: \n", Result_1)

#2. Expected values
X0 = 1
t = 3; n = 1000; m = 10000; dt = t / n
seed = time.time()

X = []
for j in range(m):
    X_i=[X0]
    Z = np.random.normal(0, 1, 1)
    W = np.random.normal(0, 1, 1)
    for i in range(n):
        X_i.append(1/4 * X_i[i] * dt + 1/3 * X_i[i] * dt**0.5 * W - 3/4 * X_i[i] * dt**0.5 * Z)
    X.append(sum(X_i[1:len(X_i)]))  
E1 = 0
for x in X:
    if (1+x) >= 0:
        E1 += (1+x)**(1/3) / m
    else:
        E1 += -((-(1+x))**(1/3)) / m

Z = np.random.normal(0, 1, m)
W = np.random.normal(0, 1, m)
E2 = 0
for j in range(m):
    Y = np.exp(-0.08 * t + 1/3 * t**0.5 * W[j] + 3/4 * t**0.5 * Z[j])
    E2 += Y / m

print("Inputs - seed:", seed)
print("Outputs - Values: E1, E2 \n", E1, E2)

#3
#(a) European call option price - Monte Carlo with Antithetic variates
def Call_MC(S0, T, X,  r, sigma):
    n = 1000
    Z = np.random.normal(0, 1, n)
    S_T1 = S0 * np.exp((r - sigma**2 / 2) * T + sigma * T**0.5 * Z)
    S_T2 = S0 * np.exp((r - sigma**2 / 2) * T + sigma * T**0.5 * (-Z))
    S_T = (S_T1 + S_T2) / 2
    C = [max(0, np.exp(-r * T)*(S - X)) for S in S_T]
    c = sum(C) / n
    return c

#(b) European call option price - Black-Scholes
def Ncdf(d):
    if d > 0:
        N = 1 - 1/2 * (1 + 0.0498673470 * d + 0.0211410061 * d**2 + 0.0032776263 * d**3 + 
                       0.0000380036 * d**4 + 0.0000488906 * d**5 + 0.0000053830 * d**6)**(-16)
    else:
        d = -d
        N = 1/2 * (1 + 0.0498673470 * d + 0.0211410061 * d**2 + 0.0032776263 * d**3 + 
                   0.0000380036 * d**4 + 0.0000488906 * d**5 + 0.0000053830 * d**6)**(-16)
    return N
def Call_Theo(S0, T, X,  r, sigma):
    d1 = 1 / (sigma * T**0.5) * (np.log(S0 / X) + (r + sigma**2 / 2) * T)
    d2 = 1 / (sigma * T**0.5) * (np.log(S0 / X) + (r - sigma**2 / 2) * T)
    N_d1 = Ncdf(d1)
    N_d2 = Ncdf(d2)
    c_theo = S0 * N_d1 - np.exp(-r * T) * X * N_d2
    return c_theo
#(c) Greeks
seed = time.time()
S0 = np.arange(15,26); X= 20; sigma = 0.25; r = 0.04; T = 0.5
C0_MC = [Call_MC(S, T, X,  r, sigma) for S in S0]
C0_Theo = [Call_Theo(S, T, X,  r, sigma) for S in S0]
Result_Call = pd.DataFrame({"C1(MC)" : C0_MC, "C2(BS)":C0_Theo}, index = S0)
#Delta
Delta_MC = np.diff(C0_MC) / np.diff(S0)
Delta_Theo = np.diff(C0_Theo) /np.diff(S0)
Result_Delta = pd.DataFrame({"Δ1(MC)" : Delta_MC, "Δ2(BS)":Delta_Theo}, 
                             index = S0[1:len(S0)])
#Gamma
Gamma_MC = np.diff(Delta_MC) / np.diff(S0[1:len(S0)])
Gamma_Theo = np.diff(Delta_Theo) / np.diff(S0[1:len(S0)])
Result_Gamma = pd.DataFrame({"Γ1(MC)" : Gamma_MC, "Γ2(BS)":Gamma_Theo}, 
                             index = S0[2:len(S0)])
#Rho
r0 = np.arange(0.015, 0.065, 0.005)
C0_r_MC = pd.DataFrame()
for S in S0:
    C0_r_MC[S] = [Call_MC(S, T, X,  r, sigma) for r in r0]
C0_r_Theo = pd.DataFrame()
for S in S0:
    C0_r_Theo[S] = [Call_Theo(S, T, X,  r, sigma) for r in r0]
Rho_MC = [np.diff(np.diff(C0_r_MC), axis=0)[:,i] / np.diff(r0) 
            for i in range(C0_r_MC.shape[1]-1)]
Rho_Theo = [np.diff(np.diff(C0_r_Theo), axis=0)[:,i] / np.diff(r0) 
            for i in range(C0_r_Theo.shape[1]-1)]
Rho_MC = pd.DataFrame(Rho_MC, index = S0[1:len(S0)])
Rho_MC.columns = [round(i,3) for i in r0[1:len(r0)]]
Rho_Theo = pd.DataFrame(Rho_Theo, index = S0[1:len(S0)])
Rho_Theo.columns = [round(i,3) for i in r0[1:len(r0)]]
#Vega
sigma0 = np.arange(0.1, 0.37, 0.03)
C0_s_MC = pd.DataFrame()
for S in S0:
    C0_s_MC[S] = [Call_MC(S, T, X,  r, s) for s in sigma0]
C0_s_Theo = pd.DataFrame()
for S in S0:
    C0_s_Theo[S] = [Call_Theo(S, T, X,  r, s) for s in sigma0]
Vega_MC = [np.diff(np.diff(C0_s_MC), axis=0)[:,i] / np.diff(sigma0) 
            for i in range(C0_s_MC.shape[1]-1)]
Vega_Theo = [np.diff(np.diff(C0_s_Theo), axis=0)[:,i] / np.diff(sigma0) 
            for i in range(C0_s_Theo.shape[1]-1)]
Vega_MC = pd.DataFrame(Vega_MC, index = S0[1:len(S0)])
Vega_MC.columns = [round(i,3) for i in sigma0[1:len(r0)]]
Vega_Theo = pd.DataFrame(Vega_Theo, index = S0[1:len(S0)])
Vega_Theo.columns = [round(i,3) for i in sigma0[1:len(r0)]]
#Theta
T0 = np.arange(0.25, 0.75, 0.05)
C0_t_MC = pd.DataFrame()
for S in S0:
    C0_t_MC[S] = [Call_MC(S, t, X,  r, sigma) for t in T0]
C0_t_Theo = pd.DataFrame()
for S in S0:
    C0_t_Theo[S] = [Call_Theo(S, t, X,  r, sigma) for t in T0]
Theta_MC = [np.diff(np.diff(C0_t_MC), axis=0)[:,i] / np.diff(T0) 
            for i in range(C0_t_MC.shape[1]-1)]
Theta_Theo = [np.diff(np.diff(C0_t_Theo), axis=0)[:,i] / np.diff(T0) 
            for i in range(C0_t_Theo.shape[1]-1)]
Theta_MC = pd.DataFrame(Theta_MC, index = S0[1:len(S0)])
Theta_MC.columns = [round(i,3) for i in T0[1:len(r0)]]
Theta_Theo = pd.DataFrame(Theta_Theo, index = S0[1:len(S0)])
Theta_Theo.columns = [round(i,3) for i in T0[1:len(r0)]]

print("Inputs - seed, S_0, T, X, r, Sigma:\n", seed, S0, T, X, r, sigma)
print("Outputs - Values: C1, C2 \n", Result_Call)
print("The graphs for Call option price and 5 Greeks are as follows.")
print("When calculating each values, two different methods(Monte Carlo simulation")
print("and Black-Scholes model) were applied. For Rho, Vega and Theta, different")
print("interest rates, standard deviations and time were applied to figure out")
print("the sensitivity of each variable")
Result_Call.plot(title = 'Call Option Prices')
Result_Delta.plot(title = 'Deltas')
Result_Gamma.plot(title = 'Gammas')
pd.DataFrame(Rho_MC).plot(title = 'Rho(MC)')
pd.DataFrame(Rho_Theo).plot(title = 'Rho(BS)')
pd.DataFrame(Vega_MC).plot(title = 'Vega(MC)')
pd.DataFrame(Vega_Theo).plot(title = 'Vega(BS)')
pd.DataFrame(Theta_MC).plot(title = "Theta(MC)")
pd.DataFrame(Theta_Theo).plot(title = "Theta(BS)")

#4
rho = -0.6; r = 0.03; S0 = 48; V0 = 0.05; sigma = 0.42; alpha = 5.8; beta = 0.0625
K = 50; T = 0.5
n = 1000; m = 100
def Call_MC_V(S0, V0, T, K, r, sigma, alpha, beta, n, Key):
    dt = T / n
    Z1 = np.random.normal(0, 1, n)    
    Z2 = np.random.normal(0, 1, n)
    V = [V0]; S = [S0]
    if Key == "PT":
        for i in range(n):
            V.append(V[i] + alpha * (beta - V[i]) * dt + 
                     sigma * (max(0, V[i]) * dt)**0.5 * Z1[i])
            S.append(S[i] + r * S[i] * dt + 
                 S[i] * (max(0, V[i+1]) * dt)**0.5 * (rho * Z1[i] + (1-rho**2)**0.5 * Z2[i]))
    elif Key == "R":
        for i in range(n):
            V.append(np.abs(V[i]) + alpha * (beta - np.abs(V[i])) * dt + 
                     sigma * (np.abs(V[i]) * dt)**0.5 * Z1[i])
            S.append(S[i] + r * S[i] * dt + 
                 S[i] * (np.abs(V[i+1]) * dt)**0.5 * (rho * Z1[i] + (1-rho**2)**0.5 * Z2[i]))
    else:
        for i in range(n):
            V.append(V[i] + alpha * (beta - max(0, V[i])) * dt + 
                     sigma * (max(0, V[i]) * dt)**0.5 * Z1[i])
            S.append(S[i] + r * S[i] * dt + 
                 S[i] * (max(0, V[i+1]) * dt)**0.5 * (rho * Z1[i] + (1-rho**2)**0.5 * Z2[i]))    
    c = max(0, np.exp(-r * T)*(S[-1] - K))
    return V, S, c

#Full Truncation methods
V_FT = pd.DataFrame(); S_FT = pd.DataFrame(); c_FT = 0
for i in range(m):
    temp = Call_MC_V(S0, V0, T, K, r, sigma, alpha, beta, n, "FT")
    V_FT[i] = temp[0]
    S_FT[i] = temp[1]
    c_FT += temp[2] / m
#Partial Truncation methods
V_PT = pd.DataFrame(); S_PT = pd.DataFrame(); c_PT = 0
for i in range(m):
    temp = Call_MC_V(S0, V0, T, K, r, sigma, alpha, beta, n, "PT")
    V_PT[i] = temp[0]
    S_PT[i] = temp[1]
    c_PT += temp[2] / m
#Reflection methods
V_R = pd.DataFrame(); S_R = pd.DataFrame(); c_R = 0
for i in range(m):
    temp = Call_MC_V(S0, V0, T, K, r, sigma, alpha, beta, n, "R")
    V_R[i] = temp[0]
    S_R[i] = temp[1]
    c_R += temp[2] / m
print("3 price estimates using the tree methods are as belows.")
pd.DataFrame({"Full Truncation" : c_FT,
              "Partial Truncation" : c_PT,
              "Reflection" : c_R,},
                index = ["Call Price"])
print("In addition, to compare the volatility and stock price for 3 methods,")
print("graphs are indicated.")
plt.title("Full Truncation methods - Volatility"); plt.plot(V_FT)
plt.title("Full Truncation methods - Stock price"); plt.plot(S_FT)
plt.title("Partial Truncation methods - Volatility"); plt.plot(V_PT)
plt.title("Partial Truncation methods - Stock price"); plt.plot(S_PT)
plt.title("Reflection methods - Volatility"); plt.plot(V_R)
plt.title("Reflection methods - Stock price"); plt.plot(S_R)

#5
#(a) 100 uniform 2D vectors
def Unif(n, seed):
    a = 7**5; b = 0; m = 2**31 -1
    X = [seed]; U = [X[0] / m]
    for i in range(n-1):
        X.append((a * X[i] + b) % m)
        U.append(X[i+1] / m)
    return U
