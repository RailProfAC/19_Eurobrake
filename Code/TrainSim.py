# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import time
import datetime

savedate = str(datetime.datetime.now())
t = time.time()

plt.close('all')

NWagonsFleet = 100
NWagonsTrain = 10
NMC = 1000 # Number of MC samples
M = 100 #Trains assembled
N = 10 #Brakings observed per Train
# Wagon data 
# Mass      
mass = 90000
# Brake pad friction
mu = 0.12
# Brake rigging efficiency
eta = 0.81
# Nominal pad braking force
bfNom = 4.8800000e+05
# Maximum standard deviation
sdMax = 0.05
# Resulting retardation force
rBF = bfNom*eta*mu

# Function to map Wagon to its braking force
brakeforce = lambda Obj : rBF * np.random.normal(1, Obj[1])
#brakeforceExp = lambda Obj : rBF * (1 + kTrain*Obj[2])
brakeforceMC = lambda Obj : rBF * np.random.normal(1, 2*Obj[2], NMC)
#brakeforceMC = lambda Obj : rBF * (1 + np.random.standard_t(1, NMC))
#brakeforceMC = lambda Obj : rBF * (1 + Obj[2]*np.random.standard_t(5,  NMC))
# Populate train fleet
Fleet = []
for i in range(0, NWagonsFleet):
    #        Wagon number, SDtrue, Sdest, n
    sd = np.random.uniform(0, sdMax)
    wagon = [i, sd, 2*sd, 0]
    Fleet.append(wagon)
  
# Perform multiple Brakings
aObsList = []
aNomList = []
relList = []
sdList = []
kList = []
sigmaSave = [[] for k in range(NWagonsFleet)]
bfEst = np.zeros(NWagonsTrain)

for i in range(0, M):
    # Build a train
    Train = []
    WagonsSelected = np.random.choice(range(1, NWagonsFleet+1)
    , size =  NWagonsTrain, replace = False)
    for j in WagonsSelected:
        Train.append(Fleet[j-1])
    # Perform braking
    for i in range(0, N):
        bfTrue = list(map(brakeforce, Train))
        aObs = sum(bfTrue)/(NWagonsTrain * mass)
        aObs = aObs + aObs*np.random.normal(0, 0.05)
        aNom = (NWagonsTrain * rBF)/(NWagonsTrain * mass)
        # Estimate observed brakeforce of train
        bfTrain = aObs*(NWagonsTrain * mass)
        sdTrain = np.mean(np.array(Train)[1:NWagonsTrain, 1])
        kTrain = (aObs/aNom-1)/sdTrain
        bfExp = np.array(list(map(brakeforceMC, Train)))
        bfEst = np.zeros(bfExp.shape,dtype=float)
        for j in range(0, NWagonsTrain):
            # Predict: Estimate each wagon's contribution
            mask = np.ones(NWagonsTrain,dtype=bool)
            mask[j]=0
            bfEst[j] = 1/rBF*(bfTrain - np.sum(bfExp[mask,:],0))
            # Correct
            Fleet[Train[j][0]][3] += 1
            #sigma = np.sqrt((bfEst[j] - rBF)**2/(Fleet[Train[j][0]][3]-1))
            var = np.sqrt(NWagonsTrain)*np.var(bfEst[j,:])
            var2 = Fleet[Train[j][0]][2]**2
            #Fleet[Train[j][0]][2] = 1/(Fleet[Train[j][0]][3])*(Fleet[Train[j][0]][3]*Fleet[Train[j][0]][2] + sigma/rBF/np.sqrt(NWagonsTrain**2))
            Fleet[Train[j][0]][2] = np.sqrt(var*var2/(var+var2))
            sigmaSave[Train[j][0]].append(Fleet[Train[j][0]][2])
#Plotting
R = range(0,NWagonsFleet-1)
F = np.array(Fleet)
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(R, F[1:NWagonsFleet, 1], label = 'true')
axarr[0].plot(R, F[1:NWagonsFleet, 2], label = 'est.')
axarr[0].legend()
#axarr[0].set_title('Estimated vs. true SD')
axarr[1].scatter(R, F[1:NWagonsFleet, 3])

error = F[1:NWagonsFleet, 2] - F[1:NWagonsFleet, 1]

print('Pearson:', st.pearsonr(F[1:NWagonsFleet, 1],F[1:NWagonsFleet, 2]))
print('RMS: ', np.sqrt(np.dot(error.T, error)))


np.savetxt('TrainSim.dat', F, '%5.5f',  delimiter=' ',   newline='\n', header= 'No SDtrue Sdest n')

    # Save some data
#    aObsList.append(aObs)
#    aNomList.append(aNom)
#    sdList.append(sdTrain)
#    kList.append(kTrain)
#
#plt.figure()
#plt.hist(bfTrue)
#plt.hist(bfExp)
#plt.hist(bfEst)

#plt.hist(aNomList)
#    # Brake cylinder pressure
#    # Train setup
#    trainPos = np.cumsum(np.random.choice(lengthArray, NWagons))
#    trainMass = np.random.uniform(minMass, maxMass, NWagons)
#    trainBForce = np.ones(NWagons)*bfNom * mu * eta * cp                                  
#    tfAverage = np.mean(trainBForce * (tf + trainPos/c))/np.mean(trainBForce)
#    
#    # Time for braking process
#    tmaxEst = v0/(sum(trainBForce)/sum(trainMass))
#    # Two stage brake model
#    #trainBForce = np.repeat(np.expand_dims(trainBForce, 1), 
#      #                      np.rint(0.5/dt*tfAverage) + np.rint(1/dt*tmaxEst), axis = 1)
#    #bForce = np.concatenate(
#     #                       (np.zeros((NWagons, np.rint(0.5/dt*tfAverage))),
#     #                                np.ones((NWagons, np.rint(1/dt*tmaxEst)))), 1)
#    #bForce = trainBForce*bForce
#    s = v0**2/(2*sum(trainBForce)/sum(trainMass))+v0*tfAverage
#    sBrake.append(s)
#    muBrake.append(np.mean(mu))
#    tfBrake.append(tfAverage)
#    etaBrake.append(np.mean(eta))
#    cpBrake.append(3.8*np.mean(cp))
#    L = 1/k*np.exp((1-k**2)*(np.mean(mu)-muSD)**2/(2*k**2*muSD**2))
#    Lmu.append(L)
#
#
#    
#Lmu = np.array(Lmu)
#sBrake = np.array(sBrake)
#Ind = np.ones(sBrake.shape)
#p = 1/N*np.sum(np.multiply(Lmu[sBrake > skrit], Ind[sBrake > skrit]))
#n = np.sum(Ind[sBrake > skrit])
#print('Braking distance > ', skrit,' m', p)
#print('Instances counted: ', n)
#
#    
#sBrakeAv = np.average(sBrake, weights = Lmu)
#
##print('Mean:  ', np.mean(sBrake))
##print('MeanW: ', sBrakeAv)
##print('Std :  ',  np.std(sBrake))
##print('StdW:  ', np.sqrt(np.average((sBrake-sBrakeAv)**2, weights=Lmu)))
##print('1e-9:  ' , np.percentile(sBrake, 100*(1-1e-9)))
##print('Time:  ', time.time() - t)
##
##mhiststring = 'plots/'+savedate+'mhist.pdf'
##histstring = 'plots/'+savedate+'hist.pdf'
## Plotting
##plt.xkcd()
#fig = plt.figure()
#fig.add_subplot(221)
#plt.hist2d(sBrake, muBrake,bins)
#plt.xlabel('s/m')
#plt.ylabel('mu/1')
#fig.add_subplot(222)
#plt.hist2d(sBrake, tfBrake,bins)
#plt.xlabel('s/m')
#plt.ylabel('tf/s')
#fig.add_subplot(223)
#plt.hist2d(sBrake, etaBrake,bins)
#plt.xlabel('s/m')
#plt.ylabel('eta/1')
#fig.add_subplot(224)
#plt.hist2d(sBrake, cpBrake,bins)
#plt.xlabel('s/m')
#plt.ylabel('C/bar')
#fig.tight_layout()
###plt.savefig(mhiststring)
#
##fig2 = plt.figure()
##plt.hist(sBrake, bins, alpha = 0.5)
##plt.hist(sBrake, bins, weights = Lmu, alpha = 0.5)
##plt.xlabel('s/m')
##plt.ylabel('frequency')
##plt.savefig(histstring)
##plt.show()
#
###fig = plt.figure()
###ax = fig.add_subplot(111, projection='3d')
###ax.scatter(sBrake, muBrake, tfBrake, marker = ".")
###ax.set_xlabel('s')
###ax.set_ylabel('mu')
###ax.set_zlabel('tf')
###fig = plt.figure()
###ax = fig.add_subplot(111, projection='3d')
###ax.scatter(sBrake, etaBrake, tfBrake, marker = ".")
###ax.set_xlabel('s')
###ax.set_ylabel('eta')
###ax.set_zlabel('tf')
###fig = plt.figure()
###ax = fig.add_subplot(111, projection='3d')
###ax.scatter(sBrake, muBrake, cpBrake, marker = ".")
###ax.set_xlabel('s')
###ax.set_ylabel('mu')
###ax.set_zlabel('cp')
###fig = plt.figure()
###ax = fig.add_subplot(111, projection='3d')
###ax.scatter(sBrake, etaBrake, cpBrake, marker = ".")
###ax.set_xlabel('s')
###ax.set_ylabel('eta')
###ax.set_zlabel('cp')
#
##plt.show()
#
#
#
##np.savetxt('Testfile', z)
