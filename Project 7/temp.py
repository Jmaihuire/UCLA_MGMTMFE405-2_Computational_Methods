#%%
def uniformGrid(N, M): # 2N+1 = row
    stock_yy, stock_xx = np.mgrid[ -N:(N+1), 0:(M+1) ]
    option_yy, option_xx = np.mgrid[ -N:(N+1), 0:(M+1) ] #0:(T+dt):d
    #plt.scatter(stock_xx, stock_yy, marker="x")
    #plt.show()
    return

#dt = 0.1
#uniformGrid(5, T/dt)


#np.savetxt("stock.csv", np.exp(logstocks), delimiter=",")
#down_price = logstocks[ range(n, 2*n+1, 1),(logstocks[n:,:]!=0).argmax(axis=1) ] # length = n+1
explicit = dict()
for s in S:
    explicit[(s)] = [ ExplicitFiniteDifference("EP", s, r, sd, T, dt, dx, K, 4, 16) for dx in dX ]

df_explicit = pd.concat([pd.DataFrame(list(explicit.keys())), pd.DataFrame(list(explicit.values())), pd.DataFrame(BS_put)], axis=1)
df_explicit.columns = ['S0', 'sd*sqrt(dt)', 'sd*sqrt(3*dt)', 'sd*sqrt(4*dt)', 'BS_put']
print('Explicit Finite Difference Method:\n', df_explicit)


def logStockPrice(S0, r, sd, n, dX): #T, dt
    # Using the trinomial method
    X0 = math.log(S0)
    #xu = dX; xd = -dX

    # Initialize the log-stock array
    logstocks = np.zeros((2*n+1, n+1))
    logstocks[n, 0] = X0
    # Calculate log-stock values for each node (real nodes)
    # Each row holds the same log-stock price
    for i in range(1, n+1, 1):
        logstocks[n-i, i] = X0 + dX*i
        for j in range(1, i*2+1, 1):
            logstocks[n-i+j, i] = logstocks[n-i, i] - dX*j

    return logstocks


#%%
explicit = [ ExplicitFiniteDifference("EP", S0, r, sd, T, dt, dx, K, 4, 16) for dx in dX ]
explicit_result = pd.concat( [pd.DataFrame(S), pd.DataFrame(np.transpose(explicit)), pd.DataFrame(BS_put)  ] , axis=1)
explicit_result.columns = ['S0', 'sd*sqrt(dt)', 'sd*sqrt(3*dt)', 'sd*sqrt(4*dt)', 'BS_put']
explicit_result['Error_dX1'] = (abs(explicit_result['sd*sqrt(dt)'] - explicit_result['BS_put'])).astype(float)
explicit_result['Error_dX2'] = (abs(explicit_result['sd*sqrt(3*dt)'] - explicit_result['BS_put'])).astype(float)
explicit_result['Error_dX3'] = (abs(explicit_result['sd*sqrt(4*dt)'] - explicit_result['BS_put'])).astype(float)
print('Explicit Finite Difference Method:\n', explicit_result)

implicit = [ ImplicitFiniteDifference("EP", S0, r, sd, T, dt, dx, K, 4, 16) for dx in dX ]
implicit_result = pd.concat( [pd.DataFrame(S), pd.DataFrame(np.transpose(implicit)), pd.DataFrame(BS_put)  ] , axis=1)
implicit_result.columns = ['S0', 'sd*sqrt(dt)', 'sd*sqrt(3*dt)', 'sd*sqrt(4*dt)', 'BS_put']
implicit_result['Error_dX1'] = (abs(implicit_result['sd*sqrt(dt)'] - implicit_result['BS_put'])).astype(float)
implicit_result['Error_dX2'] = (abs(implicit_result['sd*sqrt(3*dt)'] - implicit_result['BS_put'])).astype(float)
implicit_result['Error_dX3'] = (abs(implicit_result['sd*sqrt(4*dt)'] - implicit_result['BS_put'])).astype(float)
print('Implicit Finite Difference Method:\n', implicit_result)

cranknicolson = [ CrankNicolsonFiniteDifference("EP", S0, r, sd, T, dt, dx, K, 4, 16) for dx in dX ]
cranknicolson_result = pd.concat( [pd.DataFrame(S), pd.DataFrame(np.transpose(cranknicolson)), pd.DataFrame(BS_put)  ] , axis=1)
cranknicolson_result.columns = ['S0', 'sd*sqrt(dt)', 'sd*sqrt(3*dt)', 'sd*sqrt(4*dt)', 'BS_put']
cranknicolson_result['Error_dX1'] = (abs(cranknicolson_result['sd*sqrt(dt)'] - cranknicolson_result['BS_put'])).astype(float)
cranknicolson_result['Error_dX2'] = (abs(cranknicolson_result['sd*sqrt(3*dt)'] - cranknicolson_result['BS_put'])).astype(float)
cranknicolson_result['Error_dX3'] = (abs(cranknicolson_result['sd*sqrt(4*dt)'] - cranknicolson_result['BS_put'])).astype(float)
print('Crank-Nicolson Finite Difference Method:\n', cranknicolson_result)
