"""
Code for Scientific Computation Project 2
Please add college id here
CID: 01849526
"""
import heapq
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import os
import time

#===== Codes for Part 1=====#
def searchGPT(graph, source, target):
    # Initialize distances with infinity for all nodes except the source
    distances = {node: float('inf') for node in graph}
    distances[source] = 0

    # Initialize a priority queue to keep track of nodes to explore
    priority_queue = [(0, source)]  # (distance, node)

    # Initialize a dictionary to store the parent node of each node in the shortest path
    parents = {}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # If the current node is the target, reconstruct the path and return it
        if current_node == target:
            path = []
            while current_node in parents:
                path.insert(0, current_node)
                current_node = parents[current_node]
            path.insert(0, source)
            return current_distance,path

        # If the current distance is greater than the known distance, skip this node
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = max(distances[current_node], weight['weight'])
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parents[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf')  # No path exists


def searchPKR(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    """

    Fdict = {}
    Mdict = {}
    Mlist = []
    dmin = float('inf')
    n = len(G)
    G.add_node(n)
    heapq.heappush(Mlist,[0,s])
    Mdict[s]=Mlist[0]
    found = False

    while len(Mlist)>0:
        dmin,nmin = heapq.heappop(Mlist)
        if nmin == x:
            found = True
            break

        Fdict[nmin] = dmin

        for m,en,wn in G.edges(nmin,data='weight'):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = max(dmin,wn)
                if dcomp<Mdict[en][0]:
                    l = Mdict.pop(en) # is this even doing anything
                    l[1] = n # what is this doing? surely can be removed
                    lnew = [dcomp,en]
                    heapq.heappush(Mlist,lnew)
                    Mdict[en]=lnew
            else:
                dcomp = max(dmin,wn)
                lnew = [dcomp,en]
                heapq.heappush(Mlist, lnew)
                Mdict[en] = lnew

    return dmin


def searchPKR2(G,s,x):
    """Input:
    G: weighted NetworkX graph with n nodes (numbered as 0,1,...,n-1)
    s: an integer corresponding to a node in G
    x: an integer corresponding to a node in G
    Output: Should produce equivalent output to searchGPT given same input
    """
    Fdict = {}
    Mdict = {}
    Mlist = []
    dist = float('inf')
    # include the path in the heap queue
    heapq.heappush(Mlist,[0, s, [s]])
    Mdict[s]=Mlist[0]

    while len(Mlist)>0:
        dist, node, path = heapq.heappop(Mlist)
        if node == x:
            return dist, path
        Fdict[node] = dist
        for m, en, wn in G.edges(node, data='weight'):
            if en in Fdict:
                pass
            elif en in Mdict:
                dcomp = max(dist, wn)
                print(f"dcomp: {dcomp}")
                if dcomp < Mdict[en][0]:
                    # update path queue by appending node to the end
                    lnew = [dcomp, en, path + [en]]
                    heapq.heappush(Mlist,lnew)
                    print("Queue: ", Mlist)
                    Mdict[en] = lnew
            else:
                dcomp = max(dist, wn)
                # update path queue by appending node to the end
                lnew = [dcomp, en, path + [en]]
                heapq.heappush(Mlist, lnew)
                print("Queue: ", Mlist)
                Mdict[en] = lnew

    return float('inf'), []

#===== Code for Part 2=====#
def part2q1(y0,tf=1,Nt=5000):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    import numpy as np
    
    #Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,n))
    yarray[0,:] = y0
    beta = 10000/np.pi**2
    alpha = 1-2*beta
    
    def RHS(t,y):
        """
        Compute RHS of model
        """        
        dydt = np.zeros_like(y)
        for i in range(1,n-1):
            dydt[i] = alpha*y[i]-y[i]**3 + beta*(y[i+1]+y[i-1])
        
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[-1])
        dydt[-1] = alpha*y[-1]-y[-1]**3 + beta*(y[0]+y[-2])

        return dydt 


    #Compute numerical solutions
    dt = tarray[1]
    for i in range(Nt):
        yarray[i+1,:] = yarray[i,:]+dt*RHS(0,yarray[i,:])

    return tarray,yarray

def part2q1new(y0,tf=40,Nt=800):
    """
    Part 2, question 1
    Simulate system of n nonlinear ODEs

    Input:
    y0: Initial condition, size n array
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)
    seed: ensures same intial condition is generated with each simulation
    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x n array containing y at
            each time step including the initial condition.
    """
    
    from scipy.integrate import solve_ivp
    
    #Set up parameters, arrays
    n = y0.size
    tarray = np.linspace(0,tf,Nt+1)
    #yarray = np.zeros((Nt+1,n))
    #yarray[0,:] = y0
    beta = 10000/np.pi**2
    alpha = 1-2*beta
    
    def RHS(t,y):
        """
        Compute RHS of model
        """        
        dydt = np.zeros_like(y)
        
        # do numpy slicing instead of for loop
        dydt[1:n-1] = alpha*y[1:n-1]-y[1:n-1]**3 + beta*(y[2:]+y[0:n-2])
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[-1])
        dydt[-1] = alpha*y[-1]-y[-1]**3 + beta*(y[0]+y[-2])

        return dydt
    
    # solve using scipy and ensure errors are within tolerance
    sol = solve_ivp(RHS, [tarray[0],tarray[-1]], y0, t_eval=tarray, method='BDF', atol=1e-9, rtol=1e-9) 
    yarray = sol.y
    
    return tarray, yarray.T

# let's load in the data and do some timed tests
data = np.load('project2.npy') #modify/discard as needed
y0A = data[0,:] #first initial condition
y0B = data[1,:] #second initial condition

t1 = time.time()
_, solnew = part2q1new(y0A, tf=1, Nt=5000)
t2 = time.time()
dt1 = t2 - t1

t3 = time.time()
_, solold = part2q1(y0A)
t4 = time.time()
dt2 = t4 - t3

print(dt1, dt2)
# find the max absolute value - to investigate whether errors are within the range
# did this for both A and B
print(max(solnew.min(), solnew.max(), key=abs))


def part2q2(display=True):
    """
    Add code used for part 2 question 2.
    Code to load initial conditions is included below
    """

    data = np.load('project2.npy') #modify/discard as needed
    y0A = data[0,:] #first initial condition
    y0B = data[1,:] #second initial condition

    tA, solA = part2q1new(y0A)
    tB, solB = part2q1new(y0B)
    
    # create plot of paths for initial conditions A and B
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(tA, solA)
    axes[0].set_title('Solution A')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('y')
    axes[1].plot(tB, solB)
    axes[1].set_title('Solution B')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('y')

    plt.tight_layout()
    file_path = os.path.join(os.getcwd(), 'paths.png')
    plt.savefig(file_path)
    if display:
        plt.show()
    else:
        plt.close()
        
    return fig, solA, solB

beta = 10000/np.pi**2
alpha = 1-2*beta

# find Jacobian and eigenvalues to determine stability of equilibria
# compute jacobian efficiently i.e. numpy vectorization instead of for loops.
def jacobian(y, alpha, beta):
    # see pdf file for working
    n = len(y)
    J = np.zeros((n, n))

    # diagonal elements
    i = np.arange(1, n-1)
    J[i, i] = alpha - 3*y[i]**2

    # off diagonals
    J[i, i-1] = beta
    J[i, i+1] = beta

    # boundary elements
    J[0, 0] = alpha - 3*y[0]**2
    J[0, 1] = beta
    J[0, -1] = beta
    J[-1, -1] = alpha - 3*y[-1]**2
    J[-1, -2] = beta
    J[-1, 0] = beta
    
    # in this case the jacobian is real symmetric and sparse - useful later on.
    return J

# we know from analysing the graph that the equilibrium points are at t=40 (i.e. last row) for both initial conditions.
eqA = part2q2(False)[1][-1]
eqB = part2q2(False)[2][-1]
# get jacobians
JA = jacobian(eqA, alpha, beta)
JB = jacobian(eqB, alpha, beta)

# Having looked at the python documentation we will use the scipy.sparse.linalg.eigsh method to get eigenvalues
# both jacobians are sparse real symmetric matrices.

eigsA, _ = scipy.sparse.linalg.eigsh(JA, k=1000)
eigsB, _ = scipy.sparse.linalg.eigsh(JB, k=1000)

print("eigenvalues corresponding to the first set of initial conditions:", eigsA)
print("eigenvalues corresponding to the second set of initial conditions:", eigsB)

# created a function to look at the perturbations around the equilibrium points
def part2q2perturb(display=True):
    """
    Add code used for part 2 question 2.
    Code to load initial conditions is included below
    """

    data = np.load('project2.npy') #modify/discard as needed
    y0A = data[0,:] #first initial condition
    y0B = data[1,:] #second initial condition

    _, solA = part2q1new(y0A)
    _, solB = part2q1new(y0B)
    
    # find peturbations around the equilibrium points by adding random gaussian noise
    tpertA, solpertA = part2q1new(solA[-1] + np.random.normal(0, 0.1, len(y0A)))
    tpertB, solpertB = part2q1new(solB[-1] + np.random.normal(0, 0.1, len(y0A)))
    
    # create plot of paths for initial conditions A and B
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(tpertA, solpertA)
    axes[0].set_title('Perturbations around Equilibrium A')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('y')
    axes[1].plot(tpertB, solpertB)
    axes[1].set_title('Perturbations around Equilibrium B')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('y')

    plt.tight_layout()
    file_path = os.path.join(os.getcwd(), 'perturbations.png')
    plt.savefig(file_path)
    if display:
        plt.show()
    else:
        plt.close()
        
    return fig, solpertA, solpertB

part2q2perturb()

def part2q3(tf=10,Nt=1000,mu=0.2,seed=1):
    """
    Input:
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf
    mu: model parameter
    seed: ensures same random numbers are generated with each simulation

    Output:
    tarray: size Nt+1 array
    X size n x Nt+1 array containing solution
    """

    #Set initial condition
    y0 = np.array([0.3,0.4,0.5])
    np.random.seed(seed)
    n = y0.size #must be n=3
    Y = np.zeros((Nt+1,n)) #may require substantial memory if Nt, m, and n are all very large
    Y[0,:] = y0

    Dt = tf/Nt
    tarray = np.linspace(0,tf,Nt+1)
    beta = 0.04/np.pi**2
    alpha = 1-2*beta

    def RHS(t,y):
        """
        Compute RHS of model
        """
        dydt = np.array([0.,0.,0.])
        dydt[0] = alpha*y[0]-y[0]**3 + beta*(y[1]+y[2])
        dydt[1] = alpha*y[1]-y[1]**3 + beta*(y[0]+y[2])
        dydt[2] = alpha*y[2]-y[2]**3 + beta*(y[0]+y[1])

        return dydt 
    
    # "Brownian step"
    dW= np.sqrt(Dt)*np.random.normal(size=(Nt,n))

    #Iterate over Nt time steps
    for j in range(Nt):
        y = Y[j,:]
        F = RHS(0,y)
        # update using E-M method
        Y[j+1,0] = y[0]+Dt*F[0]+mu*dW[j,0]
        Y[j+1,1] = y[1]+Dt*F[1]+mu*dW[j,1]
        Y[j+1,2] = y[2]+Dt*F[2]+mu*dW[j,2]

    return tarray,Y

def part2q3analyze(tf=40, Nt=1000, M=1000):
    """
    Code for part 2, question 3
    Input:
    M: number of simulations
    display: whether to display the plots (default is True)
    Output:
    plots of ensemble average and variance
    """
    
    # specify some 3 mu values
    muvals = [0.3, 0.5, 1.0]
    tarray = np.linspace(0, tf, Nt + 1)
    
    # need a mean path for each mu hence the extra dimension
    y_means = np.zeros((Nt+1, len(muvals), 3))
    y_vars = y_means.copy()

    for i, mu in enumerate(muvals):
        # using lab 6 and copying that dimension structure
        y_sols = np.zeros((Nt+1, M, 3))
        for sim in range(M):
            sol = part2q3(tf=tf, Nt=Nt, mu=mu, seed=sim)[1]
            y_sols[:, sim, :] = sol

        y_means[:, i, :] = y_sols.mean(axis=1)
        y_vars[:, i, :] = y_sols.var(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # ensemble means
    for i in range(3):
            for j, mu in enumerate(muvals):
                axes[0].plot(tarray, y_means[:, j, i], label=f'$y_{i+1}$, $\mu={mu}$')
    
    axes[0].set_title(r'Ensemble Mean of $Y_i(t)$')
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$\overline{Y_i(t)}$')
    axes[0].legend()

    # ensemble variances
    for i in range(3):
        for j, mu in enumerate(muvals):
            axes[1].plot(tarray, y_vars[:, j, i], label=f'$y_{i+1}$, $\mu={mu}$')

    axes[1].set_title(r'Ensemble Variance of $Y_i(t)$')
    axes[1].set_xlabel(r'$t$')
    axes[1].set_ylabel(r'$\overline{Y_i(t)^2}$')
    axes[1].legend()

    plt.tight_layout()
    
    file_path = os.path.join(os.getcwd(), 'figure1.png')
    plt.savefig(file_path)
    
    return fig

fig = part2q3analyze()