"""
Project 4 code
CID: Add your CID here
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from hottbox.core import Tensor
from hottbox.metrics.decomposition import residual_rel_error
import time


def load_image(normalize=True,display=False):
    """"
    Load and return test image as numpy array
    """
    from scipy.datasets import face
    A = face()
    if normalize:
        A = A.astype(float)/255
    if display:
        plt.figure()
        plt.imshow(A)
    return A


#---------------------------
# Code for Part 1
#---------------------------
def truncated_SVD(Z, delta):
    """
    Input
    Z: an (N, M) matrix
    delta: the error term

    Returns
    U_truncated: truncated U in SVD
    S_truncated: truncated S in SVD
    Vh_truncated: truncated V^T in SVD
    """

    U, S, Vh = np.linalg.svd(Z, full_matrices=False)
    # Ensure the singular values are sorted from smallest to largest
    cum_sum = np.cumsum(S[::-1]**2)
    idx = np.searchsorted(cum_sum, delta**2)
    N = S.shape[0]
    rank = N - idx + 1
    U_truncated = U[:, :rank]
    S_truncated = np.diag(S[:rank])
    Vh_truncated = Vh[:rank, :]
    return U_truncated, S_truncated, Vh_truncated

def decompose1(A,eps):
    """
    Implementation of Algorithm 3 from KCSM
    Input:
    A: tensor stored as numpy array 
    eps: accuracy parameter
    Output:
    Glist: list containing core matrices [G1,G2,...]
    """

    N = len(A.shape) # order of A
    # initialise
    Rlist = np.ones(N, dtype=int)
    Glist = []
    delta = scipy.linalg.norm(A) * eps / np.sqrt(N - 1)
    # mode-1 unfold matrix
    Z = A.reshape(A.shape[0], -1, order='F')
     # iteratively perform truncated SVD
    for n in range(1, N):
        U_d, S_d, Vh_d = truncated_SVD(Z, delta)
        Rlist[n] = U_d.shape[1]
        SVt = S_d @ Vh_d
        Glist.append(U_d.reshape(Rlist[n-1], A.shape[n-1], Rlist[n], order='F'))
        Z = SVt.reshape(Rlist[n]*A.shape[n], -1, order='F')
    Glist.append(Z.reshape(Rlist[N-1], A.shape[N-1], 1, order='F'))

    return Glist

def reconstruct(Glist):
    """
    Reconstruction of tensor from TT decomposition core matrices
    Input:
    Glist: list containing core matrices [G1,G2,...]
    Output:
    Anew: reconstructed tensor stored as numpy array
    """
    N = len(Glist)
    G_1 = Glist[0]
    G_N = Glist[-1]

    # strip the dimensions of size 1 from G_1 and G_N
    Afac = np.squeeze(G_1, axis=0)
    Bfac = np.squeeze(G_N, axis=-1)
    # first product with A
    Anew = np.tensordot(Afac, Glist[1], axes=([1], [0]))
    for n in range(2, N-1):
        Anew = np.tensordot(Anew, Glist[n], axes=([2], [0]))
    # final product with B
    Anew = np.tensordot(Anew, Bfac, axes=([2], [0]))
    return Anew

def decompose2(A,Rlist):
    """
    Implementation of modified Algorithm 3 from KCSM with rank provided as input
    Input:
    A: tensor stored as numpy array 
    Rlist: list of values for rank, [R1,R2,...,R(N-1)]
    Output:
    Glist: list containing core matrices [G1,G2,...,GN]
    """
    Glist = []
    Rlist.insert(0, 1) # add 1 as the first rank
    N = len(A.shape)
    Z = A.reshape(A.shape[0], -1, order='F')
    for n in range(1, N):
        U, S, Vh = np.linalg.svd(Z, full_matrices=False)
        U_k, S_k, Vh_k = U[:, :Rlist[n]], np.diag(S[:Rlist[n]]), Vh[:Rlist[n], :]
        SVt = S_k @ Vh_k
        Glist.append(U_k.reshape(Rlist[n-1], A.shape[n-1], Rlist[n], order='F'))
        Z = SVt.reshape(Rlist[n] * A.shape[n], -1, order='F')
    Glist.append(Z.reshape(Rlist[N-1], A.shape[N-1], 1, order='F'))
    return Glist

def part1(A, eps, Rlist, reps, display=False):
    """
    Add code here for part 1, question 2 if needed
    """

    if len(Rlist) != len(A.shape) - 1:
        raise ValueError("Length of Rlist should be one less than the number of dimensions of A.")
    
    def TTSVD_time(A, eps, Rlist):
        t1 = time.time()
        Glist1 = decompose1(A, eps)
        t2 = time.time()
        dt1 = t2 - t1

        t3 = time.time()
        Glist2 = decompose2(A, Rlist)
        t4 = time.time()
        dt2 = t4 - t3
        return dt1, dt2, Glist1, Glist2

    dt1 = []
    dt2 = []
    for n in range(reps):
        t1, t2, Glist1, Glist2 = TTSVD_time(A, eps)
        dt1.append(t1)
        dt2.append(dt2)
    
    if display:
        # output image to see how the two do accuracy wise
        Anew1 = reconstruct(Glist1)
        Anew2 = reconstruct(Glist2)
        plt.figure()
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(Anew1)
        axs[1].imshow(Anew2)
    
    return dt1, dt2

#-------------------------
# Code for Part 2
#-------------------------

def HOSVD(A, eps):
    """
    Implementation of the HOSVD algorithm in numpy as seen in KCSM
    Input:
    A: order N tensor stored as a numpy array
    eps: accuracy parameter
    Output:
    Anew: reconstructed order N tensor as a numpy array
    """

    N = len(A.shape)
    factors = [None] * N 
    G = Tensor(A.copy())
    delta = (eps / np.sqrt(N-1)) * scipy.linalg.norm(A)
    for n in range(N):
        # mode-n unfold - must first move the n-th mode to the zero-th position
        X = np.reshape(np.moveaxis(A, n, 0), (A.shape[n], -1))
        U_d, _, _ = truncated_SVD(X, delta)
        factors[n] = U_d
        # core matrix calculation
        G = G.mode_n_product(U_d.T, n)

    # reconstruction
    Anew = G.copy()
    for n in range(N):
        Anew =  Anew.mode_n_product(factors[n], n)
    
    return Anew.data, factors

def repeated_LRF(A, eps):
    """
    Implementation of the repeated low rank factorization method
    Input:
    A: Tensor Order 3 stored as a numpy array
    eps: accuracy parameter
    Output:
    Anew: reconstructed Tensor as a numpy array
    """
    N = len(A.shape)
    delta = scipy.linalg.norm(A) * (eps / np.sqrt(N-1))
    Anew = np.zeros_like(A)
    # needed for compression rate calculations
    U_shapes = []
    S_shapes = []
    V_shapes = []
    for i in range(N):
        X = A[:, :, i]
        U_d, S_d, V_d = truncated_SVD(X, delta)
        # reconstruct step
        U_shapes.append(U_d.shape)
        S_shapes.append(S_d.shape)
        V_shapes.append(V_d.shape)
        X_d = U_d @ S_d @ V_d
        Anew[:, :, i] = X_d
    return Anew, U_shapes, S_shapes, V_shapes

def part2_image(A, eps=0, eps_arr=None, reps=10, display=False, runloop=True):
    """
    Implementation of the 3 methods for reconstructing an image
    Inputs:
    A: tensor of image as numpy array
    eps: accuracy parameter for images
    eps_arr: array of epsilons for tests
    reps: repeats in timing tests
    Ouputs:
    re_SVD: residual error from repeated low rank factorization
    re_HOSVD: "" HOSVD
    re_TTSVD: "" TTSVD
    """
    # accuracy
    re1 = np.zeros_like(eps_arr)
    re2 = np.zeros_like(eps_arr)
    re3 = np.zeros_like(eps_arr)
    times1 = np.zeros_like(eps_arr)
    times2 = np.zeros_like(eps_arr)
    times3 = np.zeros_like(eps_arr)
    cr1 = np.zeros_like(eps_arr)
    cr2 = np.zeros_like(eps_arr)
    cr3 = np.zeros_like(eps_arr)

    if runloop:
        for i, e in enumerate(eps_arr):
            # timings
            dt1 = np.zeros(reps)
            dt2 = np.zeros(reps)
            dt3 = np.zeros(reps)
            for j in range(reps):
                t1 = time.time()
                Glist = decompose1(A, e)
                A_TTSVD = reconstruct(Glist)
                t2 = time.time()
                dt1[j] = t2 - t1
                t3 = time.time()
                A_HOSVD, factors = HOSVD(A, e)
                t4 = time.time()
                dt2[j] = t4 - t3
                t5 = time.time()
                A_LRF, U_shapes, S_shapes, V_shapes = repeated_LRF(A, e)
                t6 = time.time()
                dt3[j] = t6 - t5
            times1[i], times2[i], times3[i] = np.mean(dt1), np.mean(dt2), np.mean(dt3)

            # compression rates
            crTTSVD = np.sum([np.prod(G.shape) for G in Glist]) / np.prod(A.shape)
            crHOSVD = np.sum([np.prod(fac.shape) for fac in factors]) / np.prod(A.shape)
            crLRF = 0
            for idx in range(len(U_shapes)):
                u = np.prod(U_shapes[idx])
                s = np.prod(S_shapes[idx])
                v = np.prod(V_shapes[idx])
                crLRF += u + s + v
            cr1[i], cr2[i], cr3[i] = crTTSVD, crHOSVD, crLRF / np.prod(A.shape)

            # accuracy
            reTTSVD = residual_rel_error(Tensor(A), Tensor(A_TTSVD))
            reHOSVD = residual_rel_error(Tensor(A), Tensor(A_HOSVD))
            reLRF = residual_rel_error(Tensor(A), Tensor(A_LRF))
            re1[i] = reTTSVD
            re2[i] = reHOSVD
            re3[i] = reLRF

        # plots
        fig1, axs1 = plt.subplots(1, 3, figsize=(12, 4))
        # accuracy plots v1
        axs1[0].plot(eps_arr, re1, label="TTSVD", marker='o')
        axs1[0].plot(eps_arr, re2, label="HOSVD", marker='o')
        axs1[0].plot(eps_arr, re3, label="repeated LRF", marker='o')
        axs1[0].legend(loc="upper left")
        axs1[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs1[0].set_ylabel("Residual Relative Error")
        axs1[0].set_xlabel(r"$\epsilon$")
        axs1[0].set_title(r"RRE vs $\epsilon$")
        # accuracy plots v2
        axs1[1].plot(cr1, re1, label="TTSVD", marker='o')
        axs1[1].plot(cr2, re2, label="HOSVD", marker='o')
        axs1[1].plot(cr3, re3, label="repeated LRF", marker='o')
        axs1[1].legend(loc="upper right")
        axs1[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs1[1].set_ylabel("Residual Relative Error")
        axs1[1].set_xlabel("Compression Rates")
        axs1[1].set_title("RRE vs Compression Rates")
        # timing plots
        axs1[2].plot(eps_arr, times1, label="TTSVD", marker='o')
        axs1[2].plot(eps_arr, times2, label="HOSVD", marker='o')
        axs1[2].plot(eps_arr, times3, label="repeated LRF", marker='o')
        axs1[2].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs1[2].legend(loc="upper right")
        axs1[2].set_ylabel("Runtimes")
        axs1[2].set_xlabel(r"$\epsilon$")
        axs1[2].set_title("Timing Plot")
        plt.tight_layout()
        plt.show()

    # images
    if display:
        fig2, axs2 = plt.subplots(2, 2)
        axs2[0, 0].imshow(A)
        axs2[0, 0].set_title("Original", fontsize=12)
        axs2[0, 1].imshow(reconstruct(decompose1(A, eps)))
        axs2[0, 1].set_title("TTSVD", fontsize=12)
        axs2[1, 1].imshow(repeated_LRF(A, eps)[0])
        axs2[1, 1].set_title("Repeated LRF", fontsize=12)
        axs2[1, 0].imshow(HOSVD(A, eps)[0])
        axs2[1, 0].set_title("HOSVD", fontsize=12)
        fig2.suptitle(f"Reconstructed Images epsilon = {eps}")
        plt.tight_layout()
        plt.show()
    
    return "Finished"

def part2_vid(A, eps):
    return None

def video2numpy(fname='project4.mp4'):
    """
    Convert mp4 video with filename fname into numpy array
    """
    import cv2
    cap = cv2.VideoCapture(fname)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    A = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, A[fc] = cap.read()
        fc += 1

    cap.release()
    
    return A.astype(float)/255 #Scales A to contain values between 0 and 1

def numpy2video(output_fname, A, fps=30):
    """
    Convert numpy array A into mp4 video and save as output_fname
    fps: frames per second.
    """
    import cv2
    video_array = A*255 #assumes A contains values between 0 and 1
    video_array  = video_array.astype('uint8')
    height, width, _ = video_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_fname, fourcc, fps, (width, height))

    for frame in video_array:
        out.write(frame)

    out.release()

    return None

#----------------------
if __name__=='__main__':
    A_im = load_image()
    eps_arr = np.linspace(0.1, 0.6, num=10)
    #print(part2_image(A_im, eps_arr))
    print(part2_image(A_im, 0.15, display=True, runloop=False))
    print(part2_image(A_im, 0.25, display=True, runloop=False))
    print(part2_image(A_im, 0.5, display=True, runloop=False))
    

