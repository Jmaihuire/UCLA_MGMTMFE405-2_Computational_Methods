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

bond_1, call_1 = cirPrice_mc(r0, sd, kappa, r_bar, t, S, L, cir_r, pureBondPrice, N, steps, 0, 0, K, T)
print('2.(a). European call value using MC (under CIR)', bond_1, call_1)
