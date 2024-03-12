import numpy as np
import scipy
from numpy import linalg as LA
from scipy.linalg import toeplitz
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd


# %% updateChol

def updateChol(R_I, n, N, R, explicitA, activeSet, newIndex, zeroTol):
    # global opts_tr, zeroTol

    flag = 0
    
    newVec = R[:,newIndex]

    if len(activeSet) == 0:
        R_I0 = np.sqrt(np.sum(newVec**2))
    else:
        if explicitA:
            if len(np.array([R_I]).flatten()) == 1:
                p = scipy.linalg.solve(np.array(R_I).reshape(-1,1), np.matmul(R[:,activeSet].transpose(), R[:,newIndex]), transposed = True, lower = False)
            else:
                p = scipy.linalg.solve(R_I, np.matmul(R[:,activeSet].transpose(), R[:,newIndex]), transposed = True, lower = False)
            
        else:
            #AnewVec = feval(R,2,n,length(activeSet),newVec,activeSet,N);
            #p = linsolve(R_I,AnewVec,opts_tr);
            raise Exception("This part is not written. Need some works done")

            pass
        q = np.sum(newVec**2) - np.sum(p ** 2)
        if q <= zeroTol:
            flag = 1
            R_I0 = R_I.copy()
        else:
            if len(np.array([R_I]).shape) == 1:
                R_I = np.array([R_I]).reshape(-1,1)
            #print(R_I)
            R_I0 = np.zeros([np.array(R_I).shape[0] + 1,R_I.shape[1] + 1])
            R_I0[0:R_I.shape[0],0:R_I.shape[1]] = R_I
            R_I0[0:R_I.shape[0],-1] = p
            R_I0[-1,-1] = np.sqrt(q)
            

    return R_I0, flag
    
def downdateChol(R, j):
    # global opts_tr, zeroTol

    def planerot(x):
        # http://statweb.stanford.edu/~susan/courses/b494/index/node30.html
        if x[1] != 0:
            r = LA.norm(x)
            G = np.zeros(len(x) + 2)
            G[:len(x)] = x / r
            G[-2] = -x[1] / r
            G[-1] = x[0] / r
        else:
            G = np.eye(2)
        return G, x


    R1 = np.zeros([R.shape[0],R.shape[1] - 1])
    R1[:,:j] = R[:,:j]
    R1[:,j:] = R[:,j+1:]
    m = R1.shape[0]; n = R1.shape[1]

    for k in range(j,n):
        p = np.array([k,k+1])
        G,R[p,k] = planerot(R[p,k])
        if k < n:
            R[p,k+1:n] = G * R[p, k+1:n]

    return R[:n,:]
    

# %% Lasso
def lasso(R, s, sr, maxIters, epsilon):
    N = len(s)
    W = R.shape[1]

    OptTol = -10
    solFreq = 0
    resStop2 = .0005
    lmbdaStop = 0

    # Global var for linsolve functions..

    optsUT = True
    opts_trUT = True
    opts_trTRANSA = True
    zeroTol = 1e-5


    x = np.zeros(W)
    x_old = np.zeros(W)
    iter = 0

    c = np.matmul(R.transpose(), s.reshape(-1,1)).reshape(-1)

    lmbda = np.max(c)

    if lmbda < 0:
        raise Exception("y is not expressible as a non-negative linear combination of the columns of X")
    
    newIndices = np.argwhere(np.abs(c-lmbda) < zeroTol).flatten()

    collinearIndices = []
    beta = []
    duals = []
    res = s

    if (lmbdaStop > 0 and lmbda < lmbdaStop) or ((epsilon > 0) and (LA.norm(res) < epsilon)):
        activationHist = []
        numIters = 0
    
    R_I = []
    activeSet = []

    for j in range(0, len(newIndices)):
        iter = iter + 1
        R_I, flag = updateChol(R_I, N, W, R, 1, activeSet,newIndices[j],zeroTol)
        activeSet.append(newIndices[j])
    activationHist = activeSet.copy()
        
    # Loop
    done = 0
    while done == 0:
        if len(activationHist) == 4:
            lmbda = np.max(c)
            newIndices = np.argwhere(np.abs(c-lmbda) < zeroTol).flatten()
            activeSet = []
            for j in range(0, len(newIndices)):
                iter = iter + 1
                R_I, flag = updateChol(R_I, N, W, R, 1 , activeSet, newIndices[j],zeroTol)
                activeSet.append(newIndices[j])
            [activationHist.append(ele) for ele in activeSet]
        else:
            lmbda = c[activeSet[0]]

        dx = np.zeros(W)

        if len(np.array([R_I]).flatten()) == 1: 
            z = scipy.linalg.solve(R_I.reshape([-1,1]), np.sign(c[np.array(activeSet).flatten()].reshape(-1,1)),transposed = True, lower = False)
        else:
            z = scipy.linalg.solve(R_I, np.sign(c[np.array(activeSet).flatten()].reshape(-1,1)),transposed = True, lower = False)
        
        
        if len(np.array([R_I]).flatten()) == 1:
            dx[np.array(activeSet).flatten()] = scipy.linalg.solve(R_I.reshape([-1,1]), z,transposed = False, lower = False)
        else:
            dx[np.array(activeSet).flatten()] = scipy.linalg.solve(R_I, z,transposed = False, lower = False).flatten()
        
        
        v = np.matmul(R[:,np.array(activeSet).flatten()], dx[np.array(activeSet).flatten()].reshape(-1,1))
        ATv = np.matmul(R.transpose(), v).flatten()
        

        gammaI = np.Inf
        removeIndices = []

        inactiveSet = np.arange(0,W)
        if len(np.array(activeSet).flatten()) > 0:
            inactiveSet[np.array(activeSet).flatten()] = -1

        if len(np.array(collinearIndices).flatten()) > 0:
            inactiveSet[np.array(collinearIndices).flatten()] = -1
        
        inactiveSet = np.argwhere(inactiveSet >= 0).flatten()

        

        if len(inactiveSet) == 0:
            gammaIc = 1
            newIndices = []
        else:
            epsilon = 1e-12
            gammaArr = (lmbda - c[inactiveSet]) / (1 - ATv[inactiveSet] + epsilon)
            
            gammaArr[gammaArr < zeroTol] = np.Inf
            gammaIc = np.min(gammaArr)
            Imin = np.argmin(gammaArr)
            newIndices = inactiveSet[(np.abs(gammaArr - gammaIc) < zeroTol)]

        
        gammaMin = min(gammaIc, gammaI)
        
        


        x = x + gammaMin * dx
        res = res - gammaMin * v.flatten()
        c = c - gammaMin * ATv


        if ((lmbda - gammaMin) < OptTol) or ((lmbdaStop > 0) and (lmbda <= lmbdaStop)) or ((epsilon > 0) and (LA.norm(res) <= epsilon)):
            newIndices = []
            removeIndices = []
            done = 1

            if (lmbda - gammaMin) < OptTol:
                #print(lmbda-gammaMin)
                pass
        if LA.norm(res[0:sr*20]) <= resStop2:
            done = 1
            if LA.norm(res[sr*20:sr*40]) <= resStop2:
                done = 1
                if LA.norm(res[sr*40:sr*60]) <= resStop2:
                    done = 1
        
        if gammaIc <= gammaI and len(newIndices) > 0:
            for j in range(0, len(newIndices)):
                iter = iter + 1
                R_I, flag = updateChol(R_I, N, W, R, 1, np.array(activeSet).flatten(), newIndices[j],zeroTol)

                if flag:
                    collinearIndices.append(newIndices[j])
                else:
                    activeSet.append(newIndices[j])
                    activationHist.append(newIndices[j])



        if gammaI <= gammaIc:
            for j in range(0, len(removeIndices)):
                iter = iter+ 1
                col = np.argwhere(np.array(activeSet).flatten() == removeIndices[j]).flatten()

                R_I = downdateChol(R_I, col)
                activeSet.pop(col)
                collinearIndices = []
            
            x[np.array(removeIndices).flatten()] = 0
            activationHist.append(-removeIndices)
        if iter >= maxIters:
            done = 1
        
        if len(np.argwhere(x<0).flatten()) > 0:
            x = x_old.copy()
            done = 1
        else:
            x_old = x.copy()
        
        if done or ((solFreq > 0) and not (iter % solFreq)):
            beta.append(x)
            duals.append(v)
    numIters = iter
    return np.array(beta).reshape(-1,1), numIters, activationHist, duals, lmbda, res


        
        
        

    


# %%

def sparsEDA(signalIn,sr,epsilon,Kmax,dmin,rho):

    # Exceptions
    if len(signalIn)/sr < 80:
        raise AssertionError("Signal not enough large. longer than 80 seconds")
    if np.sum(np.isnan(signalIn)) > 0:
        raise AssertionError("Signal contains NaN")

    # Preprocessing
    signalAdd = np.zeros(len(signalIn) + (20*sr) + (60*sr))
    signalAdd[0:20*sr] = signalIn[0]
    signalAdd[20*sr:20*sr+len(signalIn)] = signalIn
    signalAdd[20*sr+len(signalIn):] = signalIn[-1]

    if sr == 16: # test purpose for a given example
        signalAdd = signal.resample_poly(signalAdd,1,2)
        Nss = int((8 * len(signalIn) / sr) + (0.5))
        sr = 8

    elif sr>8:
        raise AssertionError("resample to ~8 Hz ")
    else:
        Nss = len(signalIn)
    Ns = len(signalAdd)
    b0 = 0

    pointerS = (20*sr)
    pointerE = pointerS + Nss
    signalRs = signalAdd[pointerS:pointerE]

    # overlap Save
    durationR = 70
    Lreg = int(20*sr*3)
    L = 10 * sr
    N = durationR * sr
    T = 6

    Rzeros = np.zeros([N+L,Lreg * 5])
    srF = sr * np.array([0.5,0.75,1,1.25,1.5])

    for j in range(0,len(srF)):
        t_rf = np.arange(0,10+1e-10,1/srF[j]) # 10 sec
        taus = np.array([0.5, 2, 1])
        rf_biexp = np.exp(-t_rf / taus[1]) - np.exp(-t_rf/taus[0])
        rf_est = taus[2] * rf_biexp
        rf_est = rf_est / np.sqrt(np.sum(rf_est**2))

        rf_est_zeropad = np.zeros(len(rf_est) + (N-len(rf_est)))
        rf_est_zeropad[:len(rf_est)] = rf_est
        Rzeros[0:N,j*Lreg:(j+1)*Lreg] = toeplitz(rf_est_zeropad, np.zeros(Lreg))

    

    R0 = Rzeros[0:N, 0:5*Lreg]
    R = np.zeros([N,T + Lreg * 5])
    R[0:N, T:] = R0
    
    # SCL
    R[0:Lreg,0] = np.linspace(0,1,Lreg)
    R[0:Lreg,1] = -np.linspace(0,1,Lreg)
    R[int(Lreg/3):Lreg,2] = np.linspace(0,2/3,int((2*Lreg)/3))
    R[int(Lreg/3):Lreg,3] = -np.linspace(0,2/3,int((2*Lreg)/3))
    R[int(2*Lreg/3):Lreg,4] = np.linspace(0,1/3,int(Lreg/3))
    R[int(2*Lreg/3):Lreg,5] = -np.linspace(0,1/3,int(Lreg/3))
    Cte = np.sum(R[:,0]**2)
    R[:,0:6] = R[:,0:6] / np.sqrt(Cte)

    pd.DataFrame(R).to_csv('d1.csv',index=None,columns=None)
    
    # Loop
    cutS = 0
    cutE = N
    slcAux = np.zeros(Ns)
    driverAux = np.zeros(Ns)
    resAux = np.zeros(Ns)
    aux = 0

    while cutE < Ns:
        aux = aux + 1
        signalCut = signalAdd[cutS:cutE]

        if b0 == 0:
            b0 = signalCut[0]

        signalCutIn = signalCut - b0
        beta,_, activationHist,_,_,_ = lasso(R,signalCutIn,sr,Kmax,epsilon)
        
        
        signalEst = (np.matmul(R, beta) + b0).reshape(-1)
        

        #remAout = (signalCut - signalEst).^2;
        #res2 = sum(remAout(20*sr+1:(40*sr)));
        #res3 = sum(remAout(40*sr+1:(60*sr)));

        remAout = (signalCut - signalEst)**2
        res2 = np.sum(remAout[20*sr:40*sr])
        res3 = np.sum(remAout[40*sr:60*sr])

        jump = 1
        if res2 < 1:
            jump = 2
            if res3 < 1:
                jump = 3


        SCL = np.matmul(R[:,0:6], beta[0:6,:]) + b0

        SCRline = beta[6:,:]

        SCRaux = np.zeros([Lreg,5])
        SCRaux[:] = SCRline.reshape([5,Lreg]).transpose()
        driver = SCRaux.sum(axis=1)
        
        b0 = np.matmul(R[jump*20*sr-1,0:6], beta[0:6,:]) + b0

        driverAux[cutS:cutS + (jump*20*sr)] = driver[0:jump*sr*20]
        slcAux[cutS:cutS + (jump*20*sr)] = SCL[0:jump*sr*20].reshape(-1)
        resAux[cutS:cutS + (jump*20*sr)] = remAout[0:jump*sr*20]
        cutS = cutS + jump * 20 * sr
        cutE = cutS + N


    

    SCRaux    = driverAux[pointerS:pointerE]
    SCL       = slcAux[pointerS:pointerE]
    MSE       = resAux[pointerS:pointerE]

    
            
    

    # PP
    ind = np.argwhere(SCRaux > 0).reshape(-1)
    driver = np.zeros(len(SCRaux))
    if ind.shape[0] == 0:
        return driver, SCL, MSE
    scr_temp = SCRaux[ind]
    ind2 = np.argsort(scr_temp)[::-1]
    scr_ord = scr_temp[ind2]
    scr_fin = [ scr_ord[0] ]
    ind_fin = [ ind[ind2[0]] ]

    for i in range (1,len(ind2)):
        if np.all(np.abs(   ind[ind2[i]] - ind_fin)   >= dmin):
            scr_fin.append(scr_ord[i])
            ind_fin.append(ind[ind2[i]])

    driver[np.array(ind_fin)] = np.array(scr_fin)

    scr_max = scr_fin[0]
    threshold = rho * scr_max
    driver[driver < threshold] = 0
    
    return driver, SCL, MSE
