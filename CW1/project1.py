"""
Code for Scientific Computation Project 1
Please add college id here
CID: 01849526
"""


#===== Code for Part 1=====#
def part1(Xin,istar):
    """
    Sort list of integers in non-decreasing order
    Input:
    Xin: List of N integers
    istar: integer between 0 and N-1 (0<=istar<=N-1)
    Output:
    X: Sorted list
    """
    X = Xin.copy() 
    for i,x in enumerate(X[1:],1):
        if i<=istar:
            ind = 0
            for j in range(i-1,-1,-1):
                if x>=X[j]:
                    ind = j+1
                    break                   
        else:
            a = 0
            b = i-1
            while a <= b:
                c = (a+b) // 2
                if X[c] < x:
                    a = c + 1
                else:
                    b = c - 1
            ind = a
        
        X[ind+1:i+1] = X[ind:i]
        X[ind] = x

    return X

import numpy as np
import matplotlib.pyplot as plt
import time
import random

def part1_time():
    """Examine dependence of walltimes of part1 function on N and istar
        
        Output:
        figure: subplots of walltimes for istar = 0, N/2 and N - 1
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # first plot istar = 0
    tvals0 = []
    # second plot istar = N/2
    tvalsh = []
    # third plot istar = N-1
    tvalsN = []
    
    Nvals = np.linspace(1, 1000, 10, dtype=int)
    
    for N in Nvals:

        # find average t for each N
        x = []
        y = []
        z = []
        
        X = np.random.randint(1, 50, N)
        
        for i in range(100):
            # Measure time for istar = 0
            start_time = time.time()
            _ = part1(X, 0)
            x.append(time.time() - start_time)
            
            # Measure time for istar = N/2
            start_time = time.time()
            _ = part1(X, N // 2)
            y.append(time.time() - start_time)
            
            # Measure time for istar = N - 1
            start_time = time.time()
            _ = part1(X, N - 1)
            z.append(time.time() - start_time)
        
        tvals0.append(np.mean(x))
        tvalsh.append(np.mean(y))
        tvalsN.append(np.mean(z))

    # Plotting the results
    axs[0].plot(Nvals, tvals0, marker='o', linestyle='--', label='Istar = 0')
    axs[0].plot(Nvals, tvalsh, marker='o', linestyle='--', label='Istar = N/2')
    axs[0].plot(Nvals, tvalsN, marker='o', linestyle='--', label='Istar = N-1')
    axs[0].set_title('Wall Times vs. N for Different Istar Values')
    axs[0].set_xlabel('N')
    axs[0].set_ylabel('Wall Time (seconds)')
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].loglog(Nvals, tvals0, marker='o', linestyle='--')
    axs[1].set_title('Istar = 0 Log-Log Plot')
    axs[1].set_xlabel('N')
    axs[1].set_ylabel('Wall Time (seconds)')
    axs[1].grid(True)
    
    plt.tight_layout()
    
    return fig

figure1 = part1_time()
plt.show(figure1)

#===== Code for Part 2=====#

def part2(S,T,m):
    """Find locations in S of all length-m sequences in T
    Input:
    S,T: length-n and length-l gene sequences provided as strings

    Output:
    L: A list of lists where L[i] is a list containing all locations 
    in S where the length-m sequence starting at T[i] can be found.
   """
    #Size parameters
    n = len(S) 
    l = len(T) 
    L = [[] for i in range(l-m+1)] #use/discard as needed
    
    # Idea: use Rabin Karp as that is what we have been taught.
    # convert to base 4
    def char2base4(S):
        c2b = {}
        c2b['A'] = 0
        c2b['C'] = 1
        c2b['G'] = 2
        c2b['T'] = 3
        ls = []
        for s in S:
            ls.append(c2b[s])
        return ls
    
    X = char2base4(S)
    Y = char2base4(T)
    
    # compute hashes of Y and each m-length sequence in X
    # i.e. convert to base 10 mod Prime using Horner's method as in the notes
    # choose a really big prime so that we essentially don't have to check for collisions
    def heval(ls, Base=4, Prime=1e9 + 7):
        f = 0
        for l in ls[:-1]:
            f = Base * (l+f)
        h = (f + ls[-1]) % Prime
        return h
    
    X_hash = heval(X[:m])
    Y_hash = heval(Y[:m])
    Y_hashmap = {}
    Y_hashmap[Y_hash] = [0]
    bm = pow(4, int(m), int(1e9 + 7))
    
    # initialize hashmap with rolling hash
    for ind in range(1, l - m + 1):
        Y_hash = (4 * Y_hash - int(Y[ind - 1]) * bm + int(Y[ind - 1 + m])) % (1e9 + 7)
        if Y_hash in Y_hashmap:
            Y_hashmap[Y_hash].append(ind)
        else:
            Y_hashmap[Y_hash] = [ind]
    
    # check the first m-substring of X
    if X_hash in Y_hashmap:
        for k in Y_hashmap[X_hash]:
            # check they are the same substring
            if X[:m] == Y[k:k+m]:
                L[k].append(0)
    
    # at each iteration we have to check if X_hash is in the hashmap and that the strings match
    for i in range(1, n - m + 1):
        X_hash = (4 * X_hash - int(X[i - 1]) * bm + int(X[i - 1 + m])) % (1e9 + 7)
        if X_hash in Y_hashmap:
            # this for loop only happens when there 
            for k in Y_hashmap[X_hash]:
                # check they are the same substring
                if X[i:i+m] == Y[k:k+m]:
                    L[k].append(i)
                
    return L


if __name__=='__main__':
    #Small example for part 2
    S = 'ATCGTACTAGTTATCGT'
    T = 'ATCGT'
    m = 3
    out = part2(S,T,m)

    #Large gene sequence from which S and T test sequences can be constructed
    infile = open("test_sequence.txt") #file from lab 3
    sequence = infile.read()
    infile.close()