#%%
[[] ]
a = [[0] * 5 for i in range(10)] # 5: column, i: row
a[0][1] = 5
print(a)


        #print(np.matrix(option)) # get 
        #print(np.matrix(index))


    
        for i in range(2*n-1, -1, -1):
            ev = [ max(X - a[step-1], 0) for a in paths ]
            y = [ exp(-1*r*dt*y)*max(X - paths[x][]) for x in paths for y in range(1, step-j+1, 1) ]

            for j in range(step-1, -1, -1):
                #option[i][j] = 
                ev = max(X - paths[i][j], 0)
                y = sum([ exp(-1*r*dt*x)*option[i][j+x] for x in range(1, step-j+1, 1)])
                print(y)
                ecv = func(paths[i][j])
                if ev > ecv:
                    index[i][j] = 1
                    index[i][j+1] = 0
    
    #option = [ [0]*(step+1) for i in range(n*2) ] # 0 2d list
    #index = [ [0]*(step+1) for i in range(n*2) ] # 0 2d list
    for i in range(n):
        path = list()
        path_anti = list()
        z = np.random.normal(0, 1, step)
        W = [sqrtdt*x for x in z]
        W_anti = [sqrtdt*(-1)*x for x in z]

        # generate paths
        path.append(S0)
        path_anti.append(S0)
        [path.append( path[-1]*exp((r-sd*sd/2)*dt + sd*w) ) for w in W]
        [path_anti.append( path_anti[-1]*exp((r-sd*sd/2)*dt + sd*w_anti) ) for w_anti in W_anti]

        # add each path to paths
        paths.append(path)
        paths.append(path_anti)

    # American put
    if(type == "AP"):
        #final nodes (exercise value)
        #ev_T = [max(X - i[step], 0) for i in paths]
        for i in range(2*n-1, -1, -1):
            option[i][step] = max(K - paths[i][step], 0) # EV(ST)
            if option[i][step] != 0: # if EV > 0
                index[i][step] = 1 # indicates the option should be exercised in these nodes
                # index == 0 indicates the option should be continued
        
        print(np.matrix(paths))
        #print(paths[:][step])
        print( [i[step] for i in paths] ) # last element of each list
        
        for i in range(1, steps+1, 1):
            X = [ a[steps-i] for a in paths ]
            ev = [max(K - a) for a in X ]

            y = np.dot(index, exp(-1*r*dt)*option)
            print(y)
            f_1 = [func1(x) for x in X]
            f_2 = [func2(x) for x in X]
            b = [ yy*ff for yy, ff in zip(y, f) ]
            A = [ f1 ]
    elif(type == "AC"): # call paths
        # final nodes (exercise value)
        for i in range(n-1, -1, -1): # row
            option[i][step] = max(paths[i][step] - X, 0)
            if option[i][step] != 0:
                index[i][step] = 1
        #for i in range(n-1, -1, -1): # row
        #    for j in range(n-1, i-1, -1): # column
                # exercise value
        #        ev = max(stocks[i, j] - K, 0)
                # continuation value
        #       cv = discount*(p*option[i, j+1] + (1-p)*option[i+1, j+1])
        #       option[i,j] = max(ev, cv)

        #print(np.matrix(option)) # get 
        #print(np.matrix(index))

        #for i in range(n-2, -1, -1):
        #    for j in range(n-1, i-1, -1):
        #        ev = max(K - stocks[i, j], 0)
        #        cv = discount*(p*option[i, j+1] + (1-p)*option[i+1, j+1])
        #        option[i, j] = max(ev, cv)



    #print('cv:\n', cv)
    #print('index:\n', index)
    #print('option:\n', option)
    #print(index[:, np.argmax(index, axis=1)])
    #print(index*discount*option)
    #print(final*discount*option)
    #print(np.sum(final*discount*option)/(2*n))