"""Scientific Computation Project 3
01849526
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy
import time
from scipy.signal import welch
from scipy.spatial.distance import pdist
#===== Code for Part 1=====#

def plot_field(lat,lon,u,time,levels=20): # did not use - used my own contourf
    """
    Generate contour plot of u at particular time
    Use if/as needed
    Input:
    lat,lon: latitude and longitude arrays
    u: full array of wind speed data
    time: time at which wind speed will be plotted (index between 0 and 364)
    levels: number of contour levels in plot
    """
    plt.figure()
    plt.contourf(lon,lat,u[time,:,:],levels)
    plt.axis('equal')
    plt.grid()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    
    return None


def part1():#add input if needed
    """
    Code for part 1
    """ 

    #--- load data ---#
    d = np.load('data1.npz')
    lat = d['lat'];lon = d['lon'];u=d['u']
    #print(u.shape)
    #-------------------------------------#
    
    # perform PCA to find variations in time - so time as rows
    # perform PCA to find variations in space - space as rows
    # fourier on locations - seasonality
    
    # convert to 2D matrix so that we can perform PCA
    u_2D = u.reshape(u.shape[0], -1)
    # centered matrix
    X = (u_2D.T - u_2D.mean(axis=1)).T
    
    # simple data analysis
    # find max, min and median windspeed at a latitude over a year
    # average wind speed across long at each lat
    u_lat = np.mean(u, axis=2)
    max_ws = np.max(u_lat, axis=0)
    min_ws = np.min(u_lat, axis=0)
    median_ws = np.median(u_lat, axis=0)

    for i, latitude in enumerate(lat):
        print(f"Latitude: {latitude}, Max Wind Speed: {round(max_ws[i], 3)}, Min Wind Speed: {round(min_ws[i], 3)}, Median Wind Speed: {round(median_ws[i],3)}, Range: {round(max_ws[i]-min_ws[i], 3)}")
    
    # perform SVD version of PCA
    def PCA(X):
        U, S, Vt = scipy.linalg.svd(X)
        # find explained variance / scree plot to determine number of components to keep
        explained_var = S**2 / np.sum(S**2)
        # find elbow point - point where the greatest downfall in explained_var occurs
        # reverse list so that in increasing order
        var_diff = np.diff(explained_var[::-1])
        # adjust for the reversal at the start
        elbow_point = 365 - (np.argmax(var_diff) + 1)
        n_components = elbow_point + 1
        return U, S, Vt, explained_var, elbow_point, n_components
    
    U, S, Vt, explained_var, elbow_point, n_components = PCA(X)
    # plot explained variance
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))
    axs[0].plot(explained_var[:30], marker="o")
    axs[0].set_xlabel('Number of Principal Components')
    axs[0].set_ylabel('Explained Variance')
    axs[0].set_title('Explained Variance Plot')
    axs[0].axvline(x=elbow_point, color='r', linestyle='--', label='Elbow Point')
    axs[0].legend()
    print(f"number of significant components of U is {n_components}")
    # choose significant components
    U_reduced = U[:, :n_components]
    # plot PC1 in the time domain and of U_lat
    axs[1].plot(U_reduced[:, 0])
    axs[1].set_title("Wind Velocity Variation with respect to Time across the First Principal Component")
    axs[1].set_ylabel("Wind Velocity")
    axs[1].set_xlabel("Day")
    axs[2].plot(U_reduced[:, 1])
    axs[2].set_title("Wind Velocity Variation with respect to Time across the Second Principal Component")
    axs[2].set_ylabel("Wind Velocity")
    axs[2].set_xlabel("Day")
    
    fig.suptitle("Analysis of Variance with respect to Time", fontsize=28)
    fig.savefig("Time_PCA.png")
    
    # --------------------------------------------------------------------- #
    
    # now we consider the transpose of the matrix to get spatial trends
    u_t = u.transpose(1, 2, 0)
    u_2Dt = u_t.reshape(-1, u_t.shape[2])
    X1 = (u_2Dt.T - u_2Dt.mean(axis=1)).T
    
    U1, S1, Vt1, explained_var1, elbow_point1, n_components1 = PCA(X1)
    # since the n_components from the elbow point don't explain that much variance
    # I chose to cutoff when 80% of the variance was explained
    cumulative_var = np.cumsum(explained_var1) * 100
    cutoff = np.argmax(cumulative_var >= 80)
    n_components1 = cutoff + 1
    fig1, axs1 = plt.subplots(nrows=1, ncols=3, figsize=(32, 6))
    # explained variance
    axs1[0].plot(explained_var1[:100], marker="o")
    axs1[0].set_xlabel('Number of Principal Components')
    axs1[0].set_ylabel('Explained Variance')
    axs1[0].set_title('Explained Variance Plot')
    # cunmulative explained variance
    axs1[1].plot(cumulative_var[:100], marker="o")
    axs1[1].set_xlabel('Number of Principal Components')
    axs1[1].set_ylabel('Cumulative Explained Variance')
    axs1[1].set_title('Cumulative Explained Variance Plot')
    axs1[1].axvline(x=cutoff, color='r', linestyle='--', label='Cutoff Point')
    axs1[1].legend()    
    print(f"number of significant components of U' is {n_components1 + 1}")
    # find signifcant components and convert first component to a 16 x 144 matrix to plot
    U1_reduced = U1[:, :n_components1]
    PC1 = U1_reduced[:, 0].reshape(16, 144)
    # plot PC1
    axs1[2].contourf(lon, lat, PC1, levels=20)
    axs1[2].axis('equal')
    axs1[2].set_title('PC1')
    axs1[2].set_xlabel('longitude')
    axs1[2].set_ylabel('latitude')
    axs1[2].grid()
    contour_plot1 = axs1[2].contourf(lon, lat, PC1, levels=20)
    # colorbars
    plt.colorbar(contour_plot1, ax = axs1[2])
    
    fig1.suptitle("Analysis of Variance with respect to Geospatial Coordinates", fontsize=36)
    plt.tight_layout()
    fig1.savefig("Geospatial_PCA.png")
    plt.show()
    
    # ----------------------------------------------------------------------- #
    
    # fourier analysis
    # I will look at how the average wind speed changes with time over each latitude column - fft
    # I will also look at the fft of each time PC.
    
    # latitudes
    centred_lat = u_lat - np.mean(u_lat, axis=0, keepdims=True)
    n = u_lat.shape[0]
    sampling_rate = 1.0
    fft_results = np.zeros_like(u_lat, dtype=np.complex128)
    fig2, axs2 = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))
    for j in range(u_lat.shape[1]):
        column_data = centred_lat[:, j]
        # compute real FFT
        fft_result = np.fft.fft(column_data)
        fft_results[:, j] = fft_result
    
    # I look at power spectra wihtout using welch's method - so that I get raw results instead of smoother versions.
    # PC1
    centred_PC1t = U_reduced[:, 0] - np.mean(U_reduced[:, 0])
    fft_PC1 = np.fft.fft(centred_PC1t)
    # PC2
    centred_PC2t = U_reduced[:, 1] - np.mean(U_reduced[:, 0])
    fft_PC2 = np.fft.fft(centred_PC2t)
    frequencies = np.fft.fftshift(np.fft.fftfreq(n, d=sampling_rate))
    column_index = 0
    energy = np.abs(fft_results[:, column_index]) ** 2 / 365
    axs2[0].plot(frequencies, np.fft.fftshift(energy))
    axs2[0].set_title(f'FFT Spectrum for Latitude {lat[column_index % 16]}')
    axs2[0].set_xlabel('Frequency')
    axs2[0].set_ylabel('Amplitude^2')
    axs2[0].axvline(x=1/90, color='r', label="Seasonal Frequency")
    axs2[0].legend()
    axs2[1].plot(frequencies, np.fft.fftshift(np.abs(fft_PC1)**2 / 2304))
    axs2[1].set_title('Power Spectrum for PC1')
    axs2[1].set_xlabel('Frequency')
    axs2[1].set_ylabel('Amplitude^2')
    axs2[1].axvline(x=1/145, color='r', label="145 Days")
    axs2[1].legend()
    axs2[2].plot(frequencies, np.fft.fftshift(np.abs(fft_PC2)**2/ 2304))
    axs2[2].set_title('FFT Spectrum for PC2')
    axs2[2].set_xlabel('Frequency')
    axs2[2].set_ylabel('Amplitude^2')
    
    fig2.suptitle("Fourier Analysis", fontsize=30)
    plt.tight_layout()
    fig2.savefig("Fourier_Analysis.png")
    plt.show()
    
part1()

#===== Code for Part 2=====#
def part2(f,method=2):
    """
    Question 2.1 i)
    Input:
        f: m x n array
        method: 1 or 2, interpolation method to use
    Output:
        fI: interpolated data (using method)
    """

    m,n = f.shape
    fI = np.zeros((m-1,n)) #use/modify as needed

    if method==1:
        fI = 0.5*(f[:-1,:]+f[1:,:])
    else:
        #Coefficients for method 2
        alpha = 0.3
        a = 1.5
        b = 0.1
        
        #coefficients for near-boundary points
        a_bc,b_bc,c_bc,d_bc = (5/16,15/16,-5/16,1/16)
        
        # create banded system solution
        b_vec = np.zeros(fI.shape)
        b_vec[0, :] = a_bc * f[0, :] + b_bc * f[1, :] + c_bc * f[2, :] + d_bc * f[3, :]
        b_vec[-1, :] = a_bc * f[-1, :] + b_bc * f[-2, :] + c_bc * f[-3, :] + d_bc * f[-4, :]
        b_vec[1:-1, :] = b / 2 * (f[3:m, :] + f[0:m-3, :]) + a / 2 * (f[2:m-1, :] + f[1:m-2, :])
        
        # create banded matrix in form suitable for solve_banded
        A = np.zeros((3, fI.shape[0]))
        A[0, 2:] = alpha
        A[1, :] = 1
        A[2, :-2] = alpha
        
        fI = scipy.linalg.solve_banded((1,1), A, b_vec)
    return fI

def part2_analyze():
    """
    Add input/output as needed
    """
    
    def grid_points(n, m):
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, m)
        xg, yg = np.meshgrid(x, y)
        dy = y[1] - y[0]
        yI = y[:-1] + dy / 2 
        xI, yI = np.meshgrid(x, yI)  # grid for interpolated data
        return xg, yg, xI, yI
    
    n_test = [100, 500, 1000]
    n_array = np.linspace(100, 1000, 10, dtype=int)
    m_array = np.linspace(100, 1000, 10, dtype=int)
    k = 2 * np.pi
    
    fig1, axs1 = plt.subplots(1, 3, figsize=(20, 8))
    fig2, axs2 = plt.subplots()
    
    # accuracy test - by analysing how the mean error decreases with n and m
    for i, n in enumerate(n_test):
        errors_1 = np.zeros(len(m_array))
        errors_2 = np.zeros(len(m_array))
        for j, m in enumerate(m_array):
            xg, yg, xI, yI = grid_points(n, m)
            # choose sin function - may be useful to check other functions
            # chose sin because notes said to use a sinusoidal function
            f = np.sin(k * (xg + yg))
            fI_1 = part2(f, method=1)
            fI_2 = part2(f, method=2)
            y_true = np.sin(k * (xI + yI))
            e1 = abs(y_true - fI_1)
            e2 = abs(y_true - fI_2)
            # calculate average along the columns and then average over this new array
            error_1 = np.mean(np.mean(e1, axis=0))
            error_2 = np.mean(np.mean(e2, axis=0))
            errors_1[j] = error_1
            errors_2[j] = error_2
        rf_1 = errors_1[0] / errors_1
        rf_2 = errors_2[0] / errors_2
        print(f"Reduction Factor in Method 2 for n={n}:", rf_2)
        print(f"Reduction Factor in Method 1 for n={n}:", rf_1)

        # we can see from the plots that n makes little to no difference in the error as hypothesised
        # see document for explanation
        axs1[i].semilogy(m_array, errors_1, marker="o", label="method 1")
        axs1[i].semilogy(m_array, errors_2, marker="o", label="method 2")
        axs1[i].set_title(f"Errors for n={n}")
        axs1[i].legend()
        axs1[i].set_ylabel("Average Error")
        axs1[i].set_xlabel(r"Number of rows $m$")
    
    fig1.savefig("AccuracyTests.png")
    
    # timing tests - here just let m = n so that we can have a more comprehensible x axis
    # we should expect O(n^2) for both
    times1 = np.zeros(len(n_array))
    times2 = np.zeros(len(n_array))
    
    for ind, n in enumerate(n_array):
        reps1 = np.zeros(300)
        reps2 = np.zeros(300)
        for m in range(300):
            xgt, ygt, xIt, yIt = grid_points(n, n)
            f = np.sin(k * (xgt + ygt))
            t1 = time.time()
            fI_t1 = part2(f, method=1)
            t2 = time.time()
            reps1[i] = t2 - t1
            t3 = time.time()
            fI_t2 = part2(f, method=2)
            t4 = time.time()
            reps2[i] = t4 - t3
        dt1 = np.mean(reps1)
        dt2 = np.mean(reps2)
        times1[ind] = dt1
        times2[ind] = dt2
    
    # line of best fit for method1
    c1 = np.polyfit(n_array**2, times1, 1)
    s1, _ = c1
    yfit1 = np.polyval(c1, n_array**2)
    
    # line of best fit for method2
    c2 = np.polyfit(n_array**2, times2, 1)
    s2, _ = c2
    yfit2 = np.polyval(c2, n_array**2)
    
    fig1.suptitle("Analysis of Accuracy", fontsize=28)
    axs2.plot(n_array**2, yfit1, label=f"method 1, slope={s1}")
    axs2.plot(n_array**2, yfit2, label=f"method 2, slope={s2}")
    axs2.legend()
    axs2.set_title("Analysis of Efficiency")
    axs2.set_ylabel(r"Wall time")
    axs2.set_xlabel(r"$n^2$")
    plt.tight_layout()
    plt.show()
    print("Ratio of Wall Times Between Methods: ", round(s2 / s1, 3))
    fig2.savefig("TimingTests.png")
    return None

part2_analyze()





#===== Code for Part 3=====#
def part3q1(y0,alpha,beta,b,c,tf=200,Nt=800,err=1e-6,method="RK45"):
    """
    Part 3 question 1
    Simulate system of 2n nonlinear ODEs

    Input:
    y0: Initial condition, size 2*n array
    alpha,beta,b,c: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x 2*n array containing y at
            each time step including the initial condition.
    """
    
    #Set up parameters, arrays

    n = y0.size//2
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,2*n))
    yarray[0,:] = y0


    def RHS(t,y):
        """
        Compute RHS of model
        """
        #add code here
        u = y[:n];v=y[n:]
        r2 = u**2+v**2
        nu = r2*u
        nv = r2*v
        cu = np.roll(u,1)+np.roll(u,-1)
        cv = np.roll(v,1)+np.roll(v,-1)

        dydt = alpha*y
        dydt[:n] += beta*(cu-b*cv)-nu+c*nv+b*(1-alpha)*v
        dydt[n:] += beta*(cv+b*cu)-nv-c*nu-b*(1-alpha)*u

        return dydt


    sol = solve_ivp(RHS, (tarray[0],tarray[-1]), y0, t_eval=tarray, method=method,atol=err,rtol=err)
    yarray = sol.y.T 
    return tarray,yarray


def part3_analyze(display = False):#add/remove input variables if needed
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """

    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0 = np.zeros(2*n)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    # from looking at contours we see that the solution becomes chaotic at c=1.3
    c_arr = [0.5, 1, 1.3, 1.5]
    #for transient, modify tf and other parameters as needed
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45')
    y0 = y[-1,:]
    fig, axs = plt.subplots(nrows=2,ncols=2, figsize=(15, 8))
    for i, c in enumerate(c_arr):
        t,y = part3q1(y0,alpha,beta,b,c, method='RK45',err=1e-6)
        # get rid of first 100 and last 100 points as specified in question
        u,v = y[:,100:n-100],y[:,n:]

        if display:
            # contours
            row = i // 2
            col = i % 2
            axs[row, col].contourf(np.arange(n-200),t,u,20)
            axs[row, col].set_title(f"Contour plot for c={c}")
            contour = axs[row, col].contourf(np.arange(n-200),t,u,20)
            plt.colorbar(contour, ax=axs[row,col])
            ## orbit diagram <- didn't give meaningful plot
            #x = u[:, 0]
            #cplot = np.ones_like(x)*c
            #plt.plot(cplot, x, 'k.', markersize=2)
    
    #fig.savefig("Contours.png")
    
    # detailed analysis for c=c - fractal dimension using time delays and welch's method
    # change 0.5<=c<=1.5 for different results
    c=1.3
    t, y = part3q1(y0, alpha, beta, b, c, tf=20, Nt=2, method='RK45')
    y0 = y[-1,:]
    t, y = part3q1(y0, alpha, beta, b, c, method='RK45')
    u, v = y[:, 100:n-100], y[:, n:]
    
    ##calculate correlation between consecutive coordinates
    #corr = np.zeros(u.shape[1]-1)
    #for i in range(u.shape[1]-1):
        #coef = np.corrcoef(u[:, i], u[:, i+1])[0,1]
        #corr[i] = coef
    
    # consecutive coordinates are highly correlated so we do time delays to work out the fractal dimension
    # then compare to the fractal dimension on the actual solution
    
    # we have to find the optimal m and optimal tau
    # let's first find the dominant frequency using welch's method
    # then we use a grid search which gives us the minimum correlation between consecutive pairs
    
    # welch's method to get dominant frequency of the first coord of u
    # let's plot any two columns of u together
    fig2, axs2 = plt.subplots(ncols = 3, figsize=(25, 5))
    x = u[:, 0] # change pair of columns if need be
    y = u[:, 101]
    axs2[0].plot(t, x, label="x")
    axs2[0].plot(t, y, label="y") # plot shows highly sinusoidal behaviour
    axs2[0].set_title(f"Trajectories of x and y for c={c}")
    axs2[0].legend()
    axs2[0].set_xlabel("Time")
    
    dt = (t[-1] - t[0]) / (len(t) - 1)
    fxx, Pxx = welch(x, fs=1/dt)
    axs2[1].semilogy(fxx,Pxx)
    axs2[1].set_xlabel(r'Frequency')
    axs2[1].set_ylabel(r'$P_{xx}$')
    axs2[1].set_title(f"Power Spectrum c={c}")
    axs2[1].grid()
    f = fxx[Pxx==Pxx.max()][0]
    print(f"dominating frequency (c={c}) = ", f)
    print(f"dt, 1/f (c={c}) = ", dt, 1/f)
    
    # we have f so we can work out ranges of tau
    taus = np.linspace(1/(10*f), 1/(5*f), 10)
    ms = np.linspace(2, 60, 59, dtype=int)
    #mean_corr = np.zeros((len(ms), len(taus)))
    
    ## grid search - uncomment to perform we get min_index = (39, 8) - c=1.3
    #for i, m in enumerate(ms):
        #for j, tau in enumerate(taus):
            ## create lag vector of size m
            #Del = int(tau/dt)
            #v1 = np.vstack([x[m*Del:]] + [x[(m-k)*Del:(-k)*Del] for k in range(1, m)])
            #corr=np.zeros(m-1)
            #for ind in range(m-1):
                #coeff = np.corrcoef(v1[i], v1[i+1])[0, 1]
                #corr[ind] = coeff
            #mean_corr[i, j] = np.mean(corr)
    
    #min_index = np.unravel_index(np.argmin(mean_corr), mean_corr.shape)
    m = ms[39] # will have to change for different c
    tau = taus[8] # will have to change for different c
    Del = int(tau/dt)
    v1 = np.vstack([x[m*Del:]] + [x[(m-k)*Del:(-k)*Del] for k in range(1, m)])
    
    D1 = pdist(v1)
    D = pdist(u)
    eps_arr = np.linspace(1e2, 1e-1, 1000)
    C = np.zeros(len(eps_arr))
    C1 = np.zeros(len(eps_arr))
    
    for i, eps in enumerate(eps_arr):
        nC2 = u.shape[0] * (u.shape[0] - 1) / 2
        D1 = D1[D1<eps]
        C1[i] = D1.size / nC2
        D = D[D<eps]
        C[i] = D.size / nC2
    
    # raw data
    coeffs = np.polyfit(np.log(eps_arr[:-9]), np.log(C[:-9]), 1)
    slope, _ = coeffs
    y_fit = np.polyval(coeffs, np.log(eps_arr[:-9]))
    
    # lagged data
    coeffs_lag = np.polyfit(np.log(eps_arr[500:-99]), np.log(C1[500:-99]), 1)
    slope_lag, _ = coeffs_lag
    y_lag = np.polyval(coeffs_lag, np.log(eps_arr[500:-99]))
    
    axs2[2].loglog(eps_arr, C1, label="time-delay")
    axs2[2].loglog(eps_arr, C, label=f"c={c}")
    axs2[2].loglog(eps_arr[:-9], np.exp(y_fit), label=f"C(E), c={c}, slope = {round(slope,3)}", color="red", linestyle="--")
    axs2[2].loglog(eps_arr[500:-99], np.exp(y_lag), label=f"C(E), time-delay, slope = {round(slope_lag,3)}", color="purple", linestyle="--")
    axs2[2].legend(loc="upper left")
    fig2.suptitle(f"Analysis at c={c}", fontsize=20)
    fig2.savefig("DetailedAnalysis.png")
    plt.tight_layout()
    
    return None

part3_analyze()

def part3q2(x,c=1.0):
    """
    Code for part 3, question 2
    """
    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    y0 = np.zeros(2*n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)

    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #Compute solution
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    A = y[:,:n]

    #Analyze code here
    l1,v1 = np.linalg.eigh(A.T.dot(A))
    v2 = A.dot(v1)
    A2 = (v2[:,:x]).dot((v1[:,:x]).T)
    e = np.sum((A2.real-A)**2)

    return A2.real,e


if __name__=='__main__':
    x=None #Included so file can be imported
    #Add code here to call functions above if needed
