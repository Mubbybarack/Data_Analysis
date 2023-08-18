# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 08:56:28 2023

@author: mabello
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import sqrt
import math
import statistics
import seaborn as sns
import scipy.stats as stats


#%% Pt3Sn_Pt100 Inverse problem

data_3_100_I = np.loadtxt(open("C:/Users/mabello/Desktop/Research/PtSn_Paper/result_Pt3Sn_Pt100_I.csv","rb"), delimiter=',')


############### TOF propylene ##############################################

TOF_log_3_100_I_PP = []
TOF_err_check_3_100_I = []
TOF_3_100_I = -data_3_100_I[:,1]
TOF_err_3_100_I = data_3_100_I[:,29]
log_check_3_100_I = np.log10(TOF_3_100_I)
for i in range(len(TOF_3_100_I)):
    if np.isfinite(log_check_3_100_I)[i] == True:
        TOF_log_3_100_I_PP.append(log_check_3_100_I[i])
        TOF_err_check_3_100_I.append(TOF_err_3_100_I[i])

# r_3_100_I = np.random.normal(0, 1, len(TOF_log_3_100_I_PP))
# TOF_obs_3_100_I_PP = TOF_log_3_100_I_PP + np.multiply(r_3_100_I,TOF_err_check_3_100_I)
r_3_100_I = np.random.normal(0, 1, len(log_check_3_100_I))
TOF_obs_3_100_I_PP = log_check_3_100_I + np.multiply(r_3_100_I,TOF_err_check_3_100_I)


############### TOF propane ##############################################

TOF_log_3_100_I_P = []
TOF_err_check_3_100_I = []
TOF_3_100_I = data_3_100_I[:,0]
TOF_err_3_100_I = data_3_100_I[:,29]
log_check_3_100_I = np.log10(TOF_3_100_I)
for i in range(len(TOF_3_100_I)):
    if np.isfinite(log_check_3_100_I)[i] == True:
        TOF_log_3_100_I_P.append(log_check_3_100_I[i])
        TOF_err_check_3_100_I.append(TOF_err_3_100_I[i])

r_3_100_I = np.random.normal(0, 1, len(TOF_log_3_100_I_P))
TOF_obs_3_100_I_P = TOF_log_3_100_I_P + np.multiply(r_3_100_I,TOF_err_check_3_100_I)


############## Selectivity ###################################

select_obs_3_100_I = []
selectivity_3_100_I = -data_3_100_I[:,1]/data_3_100_I[:,0]
selec_logit_3_100_I = np.log(selectivity_3_100_I/(1-selectivity_3_100_I))
r = np.random.normal(0, 1, len(selectivity_3_100_I))
selec_error_3_100 = data_3_100_I[:,30]
selec_obs_3_100_I = selec_logit_3_100_I + r*selec_error_3_100

for i in range(len(selec_obs_3_100_I)):
    if np.isfinite(selec_obs_3_100_I)[i] == True:
        select_obs_3_100_I.append(selec_obs_3_100_I[i])

# selectivity_obs_3_100_I = np.exp(select_obs_3_100_I)/(1 + np.exp(select_obs_3_100_I))

selectivity_obs_3_100_I = np.exp(selec_obs_3_100_I)/(1 + np.exp(selec_obs_3_100_I))


############## Non-Selectivity ###################################

select_obs_3_100_I = []
selectivity_3_100_I = -data_3_100_I[:,1]/data_3_100_I[:,0]
selec_logit_3_100_I = np.log(selectivity_3_100_I/(1-selectivity_3_100_I))
r = np.random.normal(0, 1, len(selectivity_3_100_I))
selec_error_3_100 = data_3_100_I[:,30]
selec_obs_3_100_I = selec_logit_3_100_I + r*selec_error_3_100

for i in range(len(selec_obs_3_100_I)):
    if np.isfinite(selec_obs_3_100_I)[i] == True:
        select_obs_3_100_I.append(selec_obs_3_100_I[i])

selectivity_obs_3_100_I = np.exp(select_obs_3_100_I)/(1 + np.exp(select_obs_3_100_I))


nonselectivity_obs_3_100_I = [None]*len(selectivity_obs_3_100_I)
for i in range(len(selectivity_obs_3_100_I)):      
    nonselectivity_obs_3_100_I[i] = 1-selectivity_obs_3_100_I[i]


################## Activation energy based on propylene ###########################

Ea_obs_3_100_I_PP = []
Ea_3_100_I = data_3_100_I[:,23]
Ea_check_3_100_I = Ea_3_100_I

for i in range(len(Ea_check_3_100_I)):
    if np.isfinite(Ea_check_3_100_I)[i] == True:
        Ea_obs_3_100_I_PP.append(Ea_check_3_100_I[i])
        

################## Activation energy based on propane ###########################

Ea_obs_3_100_I_P = []
Ea_3_100_I = data_3_100_I[:,22]
Ea_check_3_100_I = Ea_3_100_I

for i in range(len(Ea_check_3_100_I)):
    if np.isfinite(Ea_check_3_100_I)[i] == True:
        Ea_obs_3_100_I_P.append(Ea_check_3_100_I[i])        


################ propane order based on propylene ##################################

CH3CH3r_obs_3_100_I_PP = []
CH3CH3r_3_100_I = data_3_100_I[:,9]
CH3CH3r_check_3_100_I = CH3CH3r_3_100_I 

for i in range(len(CH3CH3r_check_3_100_I)):
    if np.isfinite(CH3CH3r_check_3_100_I)[i] == True:
        CH3CH3r_obs_3_100_I_PP.append(CH3CH3r_check_3_100_I[i])
        
        
################ propane order based on propane ##################################

CH3CH3r_obs_3_100_I_P = []
CH3CH3r_3_100_I = data_3_100_I[:,8]
CH3CH3r_check_3_100_I = CH3CH3r_3_100_I 

for i in range(len(CH3CH3r_check_3_100_I)):
    if np.isfinite(CH3CH3r_check_3_100_I)[i] == True:
        CH3CH3r_obs_3_100_I_P.append(CH3CH3r_check_3_100_I[i])        
        

# ############### H2 order based on propylene ##################################
H2r_3_100_I = data_3_100_I[:,16]
H2r_obs_3_100_I_PP = H2r_3_100_I 

# ############### H2 order based on propylene ##################################
H2r_3_100_I = data_3_100_I[:,15]
H2r_obs_3_100_I_P = H2r_3_100_I 



#%% PtSn_Pt100 surface

data_1_100_I = np.loadtxt(open("C:/Users/mabello/Desktop/Research/PtSn_Paper/result_PtSn_Pt100_I.csv","rb"), delimiter=',')


############### TOF propylene ##############################################

TOF_log_1_100_I_PP = []
TOF_err_check_1_100_I = []
TOF_1_100_I = -data_1_100_I[:,1]
TOF_err_1_100_I = data_1_100_I[:,29]
log_check_1_100_I = np.log10(TOF_1_100_I)
for i in range(len(TOF_1_100_I)):
    if np.isfinite(log_check_1_100_I)[i] == True:
        TOF_log_1_100_I_PP.append(log_check_1_100_I[i])
        TOF_err_check_1_100_I.append(TOF_err_1_100_I[i])

# r_1_100_I = np.random.normal(0, 1, len(TOF_log_1_100_I_PP))
# TOF_obs_1_100_I_PP = TOF_log_1_100_I_PP + np.multiply(r_1_100_I,TOF_err_check_1_100_I)

r_1_100_I = np.random.normal(0, 1, len(log_check_1_100_I))
TOF_obs_1_100_I_PP = log_check_1_100_I + np.multiply(r_1_100_I,TOF_err_1_100_I)


############### TOF propane ##############################################

TOF_log_1_100_I_P = []
TOF_err_check_1_100_I = []
TOF_1_100_I = data_1_100_I[:,0]
TOF_err_1_100_I = data_1_100_I[:,29]
log_check_1_100_I = np.log10(TOF_1_100_I)
for i in range(len(TOF_1_100_I)):
    if np.isfinite(log_check_1_100_I)[i] == True:
        TOF_log_1_100_I_P.append(log_check_1_100_I[i])
        TOF_err_check_1_100_I.append(TOF_err_1_100_I[i])

r_1_100_I = np.random.normal(0, 1, len(TOF_log_1_100_I_P))
TOF_obs_1_100_I_P = TOF_log_1_100_I_P + np.multiply(r_1_100_I,TOF_err_check_1_100_I)


############## Selectivity ###################################

select_obs_1_100_I = []
selectivity_1_100_I = -data_1_100_I[:,1]/data_1_100_I[:,0]
selec_logit_1_100_I = np.log(selectivity_1_100_I/(1-selectivity_1_100_I))
r = np.random.normal(0, 1, len(selectivity_1_100_I))
selec_error_3_100 = data_1_100_I[:,30]
selec_obs_1_100_I = selec_logit_1_100_I + r*selec_error_3_100

for i in range(len(selec_obs_1_100_I)):
    if np.isfinite(selec_obs_1_100_I)[i] == True:
        select_obs_1_100_I.append(selec_obs_1_100_I[i])

# selectivity_obs_1_100_I = np.exp(select_obs_1_100_I)/(1 + np.exp(select_obs_1_100_I))
selectivity_obs_1_100_I = np.exp(selec_obs_1_100_I)/(1 + np.exp(selec_obs_1_100_I))


############## Non-Selectivity ###################################

select_obs_1_100_I = []
selectivity_1_100_I = -data_1_100_I[:,1]/data_1_100_I[:,0]
selec_logit_1_100_I = np.log(selectivity_1_100_I/(1-selectivity_1_100_I))
r = np.random.normal(0, 1, len(selectivity_1_100_I))
selec_error_3_100 = data_1_100_I[:,30]
selec_obs_1_100_I = selec_logit_1_100_I + r*selec_error_3_100

for i in range(len(selec_obs_1_100_I)):
    if np.isfinite(selec_obs_1_100_I)[i] == True:
        select_obs_1_100_I.append(selec_obs_1_100_I[i])

selectivity_obs_1_100_I = np.exp(select_obs_1_100_I)/(1 + np.exp(select_obs_1_100_I))


nonselectivity_obs_1_100_I = [None]*len(selectivity_obs_1_100_I)
for i in range(len(selectivity_obs_1_100_I)):      
    nonselectivity_obs_1_100_I[i] = 1-selectivity_obs_1_100_I[i]


################## Activation energy based on propylene ###########################

Ea_obs_1_100_I_PP = []
Ea_1_100_I = data_1_100_I[:,23]
Ea_check_1_100_I = Ea_1_100_I

for i in range(len(Ea_check_1_100_I)):
    if np.isfinite(Ea_check_1_100_I)[i] == True:
        Ea_obs_1_100_I_PP.append(Ea_check_1_100_I[i])
        

################## Activation energy based on propane ###########################

Ea_obs_1_100_I_P = []
Ea_1_100_I = data_1_100_I[:,22]
Ea_check_1_100_I = Ea_1_100_I

for i in range(len(Ea_check_1_100_I)):
    if np.isfinite(Ea_check_1_100_I)[i] == True:
        Ea_obs_1_100_I_P.append(Ea_check_1_100_I[i])        


################ propane order based on propylene ##################################

CH3CH3r_obs_1_100_I_PP = []
CH3CH3r_1_100_I = data_1_100_I[:,9]
CH3CH3r_check_1_100_I = CH3CH3r_1_100_I 

for i in range(len(CH3CH3r_check_1_100_I)):
    if np.isfinite(CH3CH3r_check_1_100_I)[i] == True:
        CH3CH3r_obs_1_100_I_PP.append(CH3CH3r_check_1_100_I[i])
        
        
################ propane order based on propane ##################################

CH3CH3r_obs_1_100_I_P = []
CH3CH3r_1_100_I = data_1_100_I[:,8]
CH3CH3r_check_1_100_I = CH3CH3r_1_100_I 

for i in range(len(CH3CH3r_check_1_100_I)):
    if np.isfinite(CH3CH3r_check_1_100_I)[i] == True:
        CH3CH3r_obs_1_100_I_P.append(CH3CH3r_check_1_100_I[i])        
        

# ############### H2 order based on propylene ##################################
H2r_1_100_I = data_1_100_I[:,16]
H2r_obs_1_100_I_PP = H2r_1_100_I 

# ############### H2 order based on propylene ##################################
H2r_1_100_I = data_1_100_I[:,15]
H2r_obs_1_100_I_P = H2r_1_100_I 



#%% Pt3Sn_Pt111 surface

data_3_111_I = np.loadtxt(open("C:/Users/mabello/Desktop/Research/PtSn_Paper/result_Pt3Sn_Pt111_I.csv","rb"), delimiter=',')


############### TOF propylene ##############################################

TOF_log_3_111_I_PP = []
TOF_err_check_3_111_I = []
TOF_3_111_I = -data_3_111_I[:,1]
TOF_err_3_111_I = data_3_111_I[:,29]
log_check_3_111_I = np.log10(TOF_3_111_I)
for i in range(len(TOF_3_111_I)):
    if np.isfinite(log_check_3_111_I)[i] == True:
        TOF_log_3_111_I_PP.append(log_check_3_111_I[i])
        TOF_err_check_3_111_I.append(TOF_err_3_111_I[i])

# r_3_111_I = np.random.normal(0, 1, len(TOF_log_3_111_I_PP))
# TOF_obs_3_111_I_PP = TOF_log_3_111_I_PP + np.multiply(r_3_111_I,TOF_err_check_3_111_I)

r_3_111_I = np.random.normal(0, 1, len(log_check_3_111_I))
TOF_obs_3_111_I_PP = log_check_3_111_I + np.multiply(r_3_111_I,TOF_err_3_111_I)


############### TOF propane ##############################################

TOF_log_3_111_I_P = []
TOF_err_check_3_111_I = []
TOF_3_111_I = data_3_111_I[:,0]
TOF_err_3_111_I = data_3_111_I[:,29]
log_check_3_111_I = np.log10(TOF_3_111_I)
for i in range(len(TOF_3_111_I)):
    if np.isfinite(log_check_3_111_I)[i] == True:
        TOF_log_3_111_I_P.append(log_check_3_111_I[i])
        TOF_err_check_3_111_I.append(TOF_err_3_111_I[i])

r_3_111_I = np.random.normal(0, 1, len(TOF_log_3_111_I_P))
TOF_obs_3_111_I_P = TOF_log_3_111_I_P + np.multiply(r_3_111_I,TOF_err_check_3_111_I)


############## Selectivity ###################################

select_obs_3_111_I = []
selectivity_3_111_I = -data_3_111_I[:,1]/data_3_111_I[:,0]
selec_logit_3_111_I = np.log(selectivity_3_111_I/(1-selectivity_3_111_I))
r = np.random.normal(0, 1, len(selectivity_3_111_I))
selec_error_3_100 = data_3_111_I[:,30]
selec_obs_3_111_I = selec_logit_3_111_I + r*selec_error_3_100

for i in range(len(selec_obs_3_111_I)):
    if np.isfinite(selec_obs_3_111_I)[i] == True:
        select_obs_3_111_I.append(selec_obs_3_111_I[i])

# selectivity_obs_3_111_I = np.exp(select_obs_3_111_I)/(1 + np.exp(select_obs_3_111_I))

selectivity_obs_3_111_I = np.exp(selec_obs_3_111_I)/(1 + np.exp(selec_obs_3_111_I))


############## Non-Selectivity ###################################

select_obs_3_111_I = []
selectivity_3_111_I = -data_3_111_I[:,1]/data_3_111_I[:,0]
selec_logit_3_111_I = np.log(selectivity_3_111_I/(1-selectivity_3_111_I))
r = np.random.normal(0, 1, len(selectivity_3_111_I))
selec_error_3_100 = data_3_111_I[:,30]
selec_obs_3_111_I = selec_logit_3_111_I + r*selec_error_3_100

for i in range(len(selec_obs_3_111_I)):
    if np.isfinite(selec_obs_3_111_I)[i] == True:
        select_obs_3_111_I.append(selec_obs_3_111_I[i])

selectivity_obs_3_111_I = np.exp(select_obs_3_111_I)/(1 + np.exp(select_obs_3_111_I))


nonselectivity_obs_3_111_I = [None]*len(selectivity_obs_3_111_I)
for i in range(len(selectivity_obs_3_111_I)):      
    nonselectivity_obs_3_111_I[i] = 1-selectivity_obs_3_111_I[i]


################## Activation energy based on propylene ###########################

Ea_obs_3_111_I_PP = []
Ea_3_111_I = data_3_111_I[:,23]
Ea_check_3_111_I = Ea_3_111_I

for i in range(len(Ea_check_3_111_I)):
    if np.isfinite(Ea_check_3_111_I)[i] == True:
        Ea_obs_3_111_I_PP.append(Ea_check_3_111_I[i])
        

################## Activation energy based on propane ###########################

Ea_obs_3_111_I_P = []
Ea_3_111_I = data_3_111_I[:,22]
Ea_check_3_111_I = Ea_3_111_I

for i in range(len(Ea_check_3_111_I)):
    if np.isfinite(Ea_check_3_111_I)[i] == True:
        Ea_obs_3_111_I_P.append(Ea_check_3_111_I[i])        


################ propane order based on propylene ##################################

CH3CH3r_obs_3_111_I_PP = []
CH3CH3r_3_111_I = data_3_111_I[:,9]
CH3CH3r_check_3_111_I = CH3CH3r_3_111_I 

for i in range(len(CH3CH3r_check_3_111_I)):
    if np.isfinite(CH3CH3r_check_3_111_I)[i] == True:
        CH3CH3r_obs_3_111_I_PP.append(CH3CH3r_check_3_111_I[i])
        
        
################ propane order based on propane ##################################

CH3CH3r_obs_3_111_I_P = []
CH3CH3r_3_111_I = data_3_111_I[:,8]
CH3CH3r_check_3_111_I = CH3CH3r_3_111_I 

for i in range(len(CH3CH3r_check_3_111_I)):
    if np.isfinite(CH3CH3r_check_3_111_I)[i] == True:
        CH3CH3r_obs_3_111_I_P.append(CH3CH3r_check_3_111_I[i])        
        

# ############### H2 order based on propylene ##################################
H2r_3_111_I = data_3_111_I[:,16]
H2r_obs_3_111_I_PP = H2r_3_111_I 

# ############### H2 order based on propylene ##################################
H2r_3_111_I = data_3_111_I[:,15]
H2r_obs_3_111_I_P = H2r_3_111_I 



#%% Pt2Sn/Pt(211) surface

data_2_211_I = np.loadtxt(open("C:/Users/mabello/Desktop/Research/PtSn_Paper/result_Pt2Sn_Pt211_I.csv","rb"), delimiter=',')


############### TOF propylene ##############################################

TOF_log_2_211_I_PP = []
TOF_err_check_2_211_I = []
TOF_2_211_I = -data_2_211_I[:,1]
TOF_err_2_211_I = data_2_211_I[:,29]
log_check_2_211_I = np.log10(TOF_2_211_I)
for i in range(len(TOF_2_211_I)):
    if np.isfinite(log_check_2_211_I)[i] == True:
        TOF_log_2_211_I_PP.append(log_check_2_211_I[i])
        TOF_err_check_2_211_I.append(TOF_err_2_211_I[i])

# r_2_211_I = np.random.normal(0, 1, len(TOF_log_2_211_I_PP))
# TOF_obs_2_211_I_PP = TOF_log_2_211_I_PP + np.multiply(r_2_211_I,TOF_err_check_2_211_I)

r_2_211_I = np.random.normal(0, 1, len(log_check_2_211_I))
TOF_obs_2_211_I_PP = log_check_2_211_I + np.multiply(r_2_211_I,TOF_err_2_211_I)


############### TOF propane ##############################################

TOF_log_2_211_I_P = []
TOF_err_check_2_211_I = []
TOF_2_211_I = data_2_211_I[:,0]
TOF_err_2_211_I = data_2_211_I[:,29]
log_check_2_211_I = np.log10(TOF_2_211_I)
for i in range(len(TOF_2_211_I)):
    if np.isfinite(log_check_2_211_I)[i] == True:
        TOF_log_2_211_I_P.append(log_check_2_211_I[i])
        TOF_err_check_2_211_I.append(TOF_err_2_211_I[i])

r_2_211_I = np.random.normal(0, 1, len(TOF_log_2_211_I_P))
TOF_obs_2_211_I_P = TOF_log_2_211_I_P + np.multiply(r_2_211_I,TOF_err_check_2_211_I)


############## Selectivity ###################################

select_obs_2_211_I = []
selectivity_2_211_I = -data_2_211_I[:,1]/data_2_211_I[:,0]
selec_logit_2_211_I = np.log(selectivity_2_211_I/(1-selectivity_2_211_I))
r = np.random.normal(0, 1, len(selectivity_2_211_I))
selec_error_3_100 = data_2_211_I[:,30]
selec_obs_2_211_I = selec_logit_2_211_I + r*selec_error_3_100

for i in range(len(selec_obs_2_211_I)):
    if np.isfinite(selec_obs_2_211_I)[i] == True:
        select_obs_2_211_I.append(selec_obs_2_211_I[i])

selectivity_obs_2_211_I = np.exp(selec_obs_2_211_I)/(1 + np.exp(selec_obs_2_211_I))


############## Non-Selectivity ###################################

select_obs_2_211_I = []
selectivity_2_211_I = -data_2_211_I[:,1]/data_2_211_I[:,0]
selec_logit_2_211_I = np.log(selectivity_2_211_I/(1-selectivity_2_211_I))
r = np.random.normal(0, 1, len(selectivity_2_211_I))
selec_error_3_100 = data_2_211_I[:,30]
selec_obs_2_211_I = selec_logit_2_211_I + r*selec_error_3_100

for i in range(len(selec_obs_2_211_I)):
    if np.isfinite(selec_obs_2_211_I)[i] == True:
        select_obs_2_211_I.append(selec_obs_2_211_I[i])

selectivity_obs_2_211_I = np.exp(select_obs_2_211_I)/(1 + np.exp(select_obs_2_211_I))


nonselectivity_obs_2_211_I = [None]*len(selectivity_obs_2_211_I)
for i in range(len(selectivity_obs_2_211_I)):      
    nonselectivity_obs_2_211_I[i] = 1-selectivity_obs_2_211_I[i]


################## Activation energy based on propylene ###########################

Ea_obs_2_211_I_PP = []
Ea_2_211_I = data_2_211_I[:,23]
Ea_check_2_211_I = Ea_2_211_I

for i in range(len(Ea_check_2_211_I)):
    if np.isfinite(Ea_check_2_211_I)[i] == True:
        Ea_obs_2_211_I_PP.append(Ea_check_2_211_I[i])
        

################## Activation energy based on propane ###########################

Ea_obs_2_211_I_P = []
Ea_2_211_I = data_2_211_I[:,22]
Ea_check_2_211_I = Ea_2_211_I

for i in range(len(Ea_check_2_211_I)):
    if np.isfinite(Ea_check_2_211_I)[i] == True:
        Ea_obs_2_211_I_P.append(Ea_check_2_211_I[i])        


################ propane order based on propylene ##################################

CH3CH3r_obs_2_211_I_PP = []
CH3CH3r_2_211_I = data_2_211_I[:,9]
CH3CH3r_check_2_211_I = CH3CH3r_2_211_I 

for i in range(len(CH3CH3r_check_2_211_I)):
    if np.isfinite(CH3CH3r_check_2_211_I)[i] == True:
        CH3CH3r_obs_2_211_I_PP.append(CH3CH3r_check_2_211_I[i])
        
        
################ propane order based on propane ##################################

CH3CH3r_obs_2_211_I_P = []
CH3CH3r_2_211_I = data_2_211_I[:,8]
CH3CH3r_check_2_211_I = CH3CH3r_2_211_I 

for i in range(len(CH3CH3r_check_2_211_I)):
    if np.isfinite(CH3CH3r_check_2_211_I)[i] == True:
        CH3CH3r_obs_2_211_I_P.append(CH3CH3r_check_2_211_I[i])        
        

# ############### H2 order based on propylene ##################################
H2r_2_211_I = data_2_211_I[:,16]
H2r_obs_2_211_I_PP = H2r_2_211_I 

# ############### H2 order based on propylene ##################################
H2r_2_211_I = data_2_211_I[:,15]
H2r_obs_2_211_I_P = H2r_2_211_I 





#%% Plots

log_TOF_exp = np.log10(0.6) 
selec_exp = 0.97


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 10,
        }

################### TOF Propane

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
# plt.plot([np.log10(31.66),np.log10(31.66)],[0,0.5], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(TOF_obs_3_100_I_P, 1000, density=True, alpha=0.7, label='Prior')
# count, bins1, ignored = plt.hist(TOF_obs_I, 1000, density=True, alpha=0.7, color='C4', label='Calibrated Bayesian')
plt.plot([log_TOF_exp,log_TOF_exp],[0,0.5],'k--', lw ="1",label='Experiment')

plt.xlabel('$log_{10}(TOF$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-6,6)
plt.ylim(0,0.6)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_TOF_P.png', dpi=500)


################### TOF Propylene

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
# plt.plot([np.log10(22.66),np.log10(22.66)],[0,0.5], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(TOF_obs_3_100_I_PP, 100, density=True, alpha=0.7, label='Pt3Sn/Pt(100)')
count, bins, ignored = plt.hist(TOF_obs_1_100_I_PP, 100, density=True, alpha=0.7, label='PtSn/Pt(100)')
count, bins, ignored = plt.hist(TOF_obs_3_111_I_PP, 100, density=True, alpha=0.7, label='Pt3Sn/Pt(111)')
count, bins, ignored = plt.hist(TOF_obs_2_211_I_PP, 100, density=True, alpha=0.7, label='Pt3Sn/Pt(111)')
# count, bins1, ignored = plt.hist(TOF_obs_I, 1000, density=True, alpha=0.7, color='C4', label='Calibrated Bayesian')
plt.plot([np.log10(0.58),np.log10(0.58)],[0,1],'k--', lw ="1",label='Experiment')

plt.xlabel('$log_{10}(TOF)$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-6,6)
# plt.ylim(0,0.6)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_TOF_PP.png', dpi=500)


################### Selectivity to propylene

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
# plt.plot([0.72,0.72],[0,4], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(selectivity_obs_3_100_I, 1000, density=True, alpha=0.7, label='Prior')
count, bins, ignored = plt.hist(selectivity_obs_1_100_I, 1000, density=True, alpha=0.7, label='Prior')
count, bins, ignored = plt.hist(selectivity_obs_3_111_I, 1000, density=True, alpha=0.7, label='Prior')
count, bins, ignored = plt.hist(selectivity_obs_2_211_I, 1000, density=True, alpha=0.7, label='Prior')
plt.plot([selec_exp,selec_exp],[0,25],'k--', lw ="1",label='Experiment')

plt.xlabel('Selectivity to Propylene', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-6,6)
plt.ylim(0,30)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_selec_PP.png', dpi=500)

################### Selectivity to non-propylene

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
plt.plot([(1-0.72),(1-0.72)],[0,4], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(nonselectivity_obs_3_100_I, 1000, density=True, alpha=0.7, label='Prior')
# count, bins1, ignored = plt.hist(TOF_obs_I, 1000, density=True, alpha=0.7, color='C4', label='Calibrated Bayesian')
plt.plot([(1-selec_exp),(1-selec_exp)],[0,4],'k--', lw ="1",label='Experiment')

plt.xlabel('Selectivity to Non-propylene', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-6,6)
plt.ylim(0,5)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_selec_Non-PP.png', dpi=500)

################### Propane reaction order based on TOF propylene

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
plt.plot([(0.98),(0.98)],[0,4], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(CH3CH3r_obs_3_100_I_PP, 100, density=True, alpha=0.7, label='Prior')
# count, bins1, ignored = plt.hist(TOF_obs_I, 1000, density=True, alpha=0.7, color='C4', label='Calibrated Bayesian')
# plt.plot([(1-selec_exp),(1-selec_exp)],[0,4],'k--', lw ="1",label='Experiment')

plt.xlabel('Propane Reaction Order', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-1,2)
plt.ylim(0,5)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_CH3CH3r_PP.png', dpi=500)

################### Propane reaction order based on TOF propane

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
plt.plot([(0.98),(0.98)],[0,4], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(CH3CH3r_obs_3_100_I_P, 100, density=True, alpha=0.7, label='Prior')
# count, bins1, ignored = plt.hist(TOF_obs_I, 1000, density=True, alpha=0.7, color='C4', label='Calibrated Bayesian')
# plt.plot([(1-selec_exp),(1-selec_exp)],[0,4],'k--', lw ="1",label='Experiment')

plt.xlabel('Propane Reaction Order', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-1,2)
plt.ylim(0,5)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_CH3CH3r_P.png', dpi=500)

################### Activation Energy based on TOF propylene

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
plt.plot([(0.43),(0.43)],[0,1], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(Ea_obs_3_100_I_PP, 1000, density=True, alpha=0.7, label='Prior')
# count, bins1, ignored = plt.hist(TOF_obs_I, 1000, density=True, alpha=0.7, color='C4', label='Calibrated Bayesian')
# plt.plot([(1-selec_exp),(1-selec_exp)],[0,4],'k--', lw ="1",label='Experiment')

plt.xlabel('Apparent Activation Energy', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-2,5)
# plt.ylim(0,5)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_Ea_PP.png', dpi=500)

################### Activation Energy based on TOF propane

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
plt.plot([(0.67),(0.67)],[0,1], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(Ea_obs_3_100_I_P, 1000, density=True, alpha=0.7, label='Prior')
# count, bins1, ignored = plt.hist(TOF_obs_I, 1000, density=True, alpha=0.7, color='C4', label='Calibrated Bayesian')
# plt.plot([(1-selec_exp),(1-selec_exp)],[0,4],'k--', lw ="1",label='Experiment')

plt.xlabel('Apparent Activation Energy', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-2,5)
# plt.ylim(0,5)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_Ea_P.png', dpi=500)

################### H2 reaction order based on TOF propylene

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
plt.plot([(0.24),(0.24)],[0,1], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(H2r_obs_3_100_I_PP, 100, density=True, alpha=0.7, label='Prior')
# count, bins1, ignored = plt.hist(TOF_obs_I, 1000, density=True, alpha=0.7, color='C4', label='Calibrated Bayesian')
# plt.plot([(1-selec_exp),(1-selec_exp)],[0,4],'k--', lw ="1",label='Experiment')

plt.xlabel('$H_2$ Reaction Order', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-1,2)
# plt.ylim(0,5)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_H2r_PP.png', dpi=500)

################### H2 reaction order based on TOF propane

plt.figure()
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### Pt(100)
plt.plot([(0.01),(0.01)],[0,1], lw ="1", color='C1', label='Regular MKM')
count, bins, ignored = plt.hist(H2r_obs_3_100_I_P, 100, density=True, alpha=0.7, label='Prior')
# count, bins1, ignored = plt.hist(TOF_obs_I, 1000, density=True, alpha=0.7, color='C4', label='Calibrated Bayesian')
# plt.plot([(1-selec_exp),(1-selec_exp)],[0,4],'k--', lw ="1",label='Experiment')

plt.xlabel('$H_2$ Reaction Order', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-1,2)
# plt.ylim(0,5)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('$Pt_3Sn$', fontdict=font)
plt.tight_layout()

# plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_H2r_P.png', dpi=500)


#%% Figure a

################### TOF Propylene

plt.figure(figsize=(6,4))
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### a
plt.subplot(2,2,1)
count, bins, ignored = plt.hist(TOF_obs_3_100_I_PP, 1000, density=True, alpha=0.7, label='Pt3Sn/Pt(100)')
count, bins, ignored = plt.hist(TOF_obs_1_100_I_PP, 1000, density=True, alpha=0.7, label='PtSn/Pt(100)')
count, bins, ignored = plt.hist(TOF_obs_3_111_I_PP, 1000, density=True, alpha=0.7, color='C2', label='Pt3Sn/Pt(111)')
count, bins, ignored = plt.hist(TOF_obs_2_211_I_PP, 1000, density=True, alpha=0.6, color='grey', label='Pt2Sn/Pt(211)')
plt.plot([np.log10(0.58),np.log10(0.58)],[0,0.75],'k--', lw ="1",label='Experiment')

plt.xlabel('$log_{10}(TOF_{Propylene})$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-5,5)
# plt.ylim(0,1)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('a.)', fontdict=font)
plt.tight_layout()


###### b
plt.subplot(2,2,3)
count, bins, ignored = plt.hist(TOF_obs_3_111_I_PP, 1000, density=True, alpha=0.7, color='C2', label='Pt3Sn/Pt(111)')
count, bins, ignored = plt.hist(TOF_obs_2_211_I_PP, 1000, density=True, alpha=0.6, color='grey', label='Pt3Sn/Pt(111)')
plt.plot([np.log10(0.58),np.log10(0.58)],[0,0.75],'k--', lw ="1",label='Experiment')

plt.xlabel('$log_{10}(TOF_{Propylene})$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-4,4)
# plt.ylim(0,0.6)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('b.)', fontdict=font)
plt.tight_layout()


###### c
plt.subplot(2,2,2)
count, bins, ignored = plt.hist(selectivity_obs_3_100_I, 1000, density=True, alpha=0.7, label='$Pt_3Sn/Pt(100)$')
count, bins, ignored = plt.hist(selectivity_obs_1_100_I, 1000, density=True, alpha=0.7, label='$PtSn/Pt(100)$')
plt.plot([selec_exp,selec_exp],[0,20],'k--', lw ="1",label='Reported Value')

plt.xlabel('Selectivity to Propylene', fontdict=font)
# plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-6,6)
plt.ylim(0,25)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('c.)', fontdict=font)
plt.tight_layout()


###### d
plt.subplot(2,2,4)
count, bins, ignored = plt.hist(selectivity_obs_3_111_I, 1000, density=True, alpha=0.7, color='C2', label='$Pt_3Sn/Pt(111)$')
count, bins, ignored = plt.hist(selectivity_obs_2_211_I, 1000, density=True, alpha=0.6, color='grey', label='$Pt_2Sn/Pt(211)$')
plt.plot([selec_exp,selec_exp],[0,20],'k--', lw ="1",label='Reported Value')

plt.xlabel('Selectivity to Propylene', fontdict=font)
# plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-6,6)
plt.ylim(0,25)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('d.)', fontdict=font)
plt.tight_layout()


plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/PtSn_I_PtSn_TOFSelec.png', dpi=500)



#%% Figure b

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 10,
        }


plt.figure(figsize=(6,4))
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### a
plt.subplot(2,3,1)
count, bins, ignored = plt.hist(Ea_obs_3_100_I_PP, 100, density=True, alpha=0.7, label='$Pt_3Sn/Pt(100)$')
count, bins, ignored = plt.hist(Ea_obs_1_100_I_PP, 100, density=True, alpha=0.7, label='$PtSn/Pt(100)$')

plt.xlabel('App. Act. Energy (eV)', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-2,5)
# plt.ylim(0,5)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('a.)', fontdict=font)
plt.tight_layout()


###### b
plt.subplot(2,3,4)
count, bins, ignored = plt.hist(Ea_obs_3_111_I_PP, 100, density=True, alpha=0.7, color='grey', label='$Pt_3Sn/Pt(100)$')
count, bins, ignored = plt.hist(Ea_obs_2_211_I_PP, 100, density=True, alpha=0.7, color='C2', label='$PtSn/Pt(100)$')

plt.xlabel('App. Act. Energy (eV)', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-1,3)
# plt.ylim(0,5)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('d.)', fontdict=font)
plt.tight_layout()


###### c
plt.subplot(2,3,2)
count, bins, ignored = plt.hist(CH3CH3r_obs_3_100_I_PP, 100, density=True, alpha=0.7, label='$Pt_3Sn/Pt(100)$')
count, bins, ignored = plt.hist(CH3CH3r_obs_1_100_I_PP, 100, density=True, alpha=0.7, label='$PtSn/Pt(100)$')

plt.xlabel('Propane Rxn Order', fontdict=font)
# plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-0.5,1.5)
plt.ylim(0,3)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('b.)', fontdict=font)
plt.tight_layout()


###### d
plt.subplot(2,3,5)
count, bins, ignored = plt.hist(CH3CH3r_obs_3_111_I_PP, 100, density=True, alpha=0.7, color='grey', label='$Pt_3Sn/Pt(111)$')
count, bins, ignored = plt.hist(CH3CH3r_obs_2_211_I_PP, 100, density=True, alpha=0.7, color='C2', label='$Pt_2Sn/Pt(211)$')

plt.xlabel('Propane Rxn Order', fontdict=font)
# plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-0.5,1.5)
plt.ylim(0,3)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('e.)', fontdict=font)
plt.tight_layout()


###### e
plt.subplot(2,3,3)
count, bins, ignored = plt.hist(H2r_obs_3_100_I_PP, 100, density=True, alpha=0.7, label='$Pt_3Sn/Pt(100)$')
count, bins, ignored = plt.hist(H2r_obs_1_100_I_PP, 100, density=True, alpha=0.7, label='$PtSn/Pt(100)$')

plt.xlabel('$H_2$ Rxn Order', fontdict=font)
# plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-3,1)
# plt.ylim(0,3)
# plt.grid(True)
# plt.legend(fontsize="7")
plt.title('c.)', fontdict=font)
plt.tight_layout()


###### f
plt.subplot(2,3,6)
count, bins, ignored = plt.hist(H2r_obs_3_111_I_PP, 100, density=True, alpha=0.7, color='grey', label='$Pt_3Sn/Pt(111)$')
count, bins, ignored = plt.hist(H2r_obs_2_211_I_PP, 100, density=True, alpha=0.7, color='C2', label='$Pt_2Sn/Pt(211)$')

plt.xlabel('$H_2$ Rxn Order', fontdict=font)
# plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-3,1)
# plt.ylim(0,3)
# plt.grid(True)
# plt.legend(fontsize="7")
plt.title('f.)', fontdict=font)
plt.tight_layout()


plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/PtSn_I_PtSn_EaOrder_PP.png', dpi=500)


#%% Figure c

################### TOF Propylene

plt.figure(figsize=(6,4))
# plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8) 

###### a
plt.subplot(2,2,1)
count, bins, ignored = plt.hist(TOF_obs_3_100_I_PP, 1000, density=True, alpha=0.7, label='Pt3Sn/Pt(100)')
count, bins, ignored = plt.hist(TOF_obs_1_100_I_PP, 1000, density=True, alpha=0.7, label='PtSn/Pt(100)')
plt.plot([np.log10(0.58),np.log10(0.58)],[0,0.75],'k--', lw ="1",label='Experiment')

plt.xlabel('$log_{10}(TOF_{Propylene})$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-4,4)
# plt.ylim(0,0.6)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('a.)', fontdict=font)
plt.tight_layout()


###### b
plt.subplot(2,2,3)
count, bins, ignored = plt.hist(TOF_obs_3_111_I_PP, 1000, density=True, alpha=0.7, color='C2', label='Pt3Sn/Pt(111)')
count, bins, ignored = plt.hist(TOF_obs_2_211_I_PP, 1000, density=True, alpha=0.6, color='grey', label='Pt3Sn/Pt(111)')
plt.plot([np.log10(0.58),np.log10(0.58)],[0,0.75],'k--', lw ="1",label='Experiment')

plt.xlabel('$log_{10}(TOF_{Propylene})$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-4,4)
# plt.ylim(0,0.6)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('b.)', fontdict=font)
plt.tight_layout()


###### c
plt.subplot(2,2,2)
count, bins, ignored = plt.hist(selectivity_obs_3_100_I, 1000, density=True, alpha=0.7, label='$Pt_3Sn/Pt(100)$')
count, bins, ignored = plt.hist(selectivity_obs_1_100_I, 1000, density=True, alpha=0.7, label='$PtSn/Pt(100)$')
plt.plot([selec_exp,selec_exp],[0,20],'k--', lw ="1",label='Reported Value')

plt.xlabel('Selectivity to Propylene', fontdict=font)
# plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-6,6)
plt.ylim(0,25)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('c.)', fontdict=font)
plt.tight_layout()


###### d
plt.subplot(2,2,4)
count, bins, ignored = plt.hist(selectivity_obs_3_111_I, 1000, density=True, alpha=0.7, color='C2', label='$Pt_3Sn/Pt(111)$')
count, bins, ignored = plt.hist(selectivity_obs_2_211_I, 1000, density=True, alpha=0.6, color='grey', label='$Pt_2Sn/Pt(211)$')
plt.plot([selec_exp,selec_exp],[0,20],'k--', lw ="1",label='Reported Value')

plt.xlabel('Selectivity to Propylene', fontdict=font)
# plt.ylabel('Probability Density', fontdict=font)
# plt.xlim(-6,6)
plt.ylim(0,25)
# plt.grid(True)
plt.legend(fontsize="8")
plt.title('d.)', fontdict=font)
plt.tight_layout()


plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/PtSn_I_PtSn_TOFSelec.png', dpi=500)



#%%% DKRC plot

#### Based on Propylene

#Pt3Sn/Pt(100)
KRC_3_100 = np.zeros((50000,130))
density_3_100 = []*130
bins_use_3_100 = np.zeros((101,130))

for j in range(130):
    KRC_check = data_3_100_I[:,31+j]
    KRC_test = []
    KRC_3_100_len = []
    for i in range(len(KRC_check)):
        if np.isfinite(KRC_check)[i] == True:
            KRC_test.append(KRC_check[i])
    KRC_3_100[:,j] = KRC_test
    KRC_3_100_len.append(len(KRC_test))
    count, bins, ignored = plt.hist(KRC_3_100[:,j], 100, density=True)
    bins_use_3_100[:,j] = bins


test_3_100 = np.mean(KRC_3_100, axis=0)
index_3_100 = []
for i in range(len(test_3_100)):
    if test_3_100[i] > 0.2 or test_3_100[i] < -0.2 :
        index_3_100.append(i)

title_3_100 = ['$CH_3CH_2CH_3 --> CH_3CHCH_3 + H$', '$CH_3CH_2CH_3 --> CH_3CH_2CH_2 + H$', 
           '$CH_3CHCH_3 --> CH_3CHCH_2 + H$', '$CH_3CH_2CH_2 --> CH_3CHCH_2 + H$']

#PtSn/Pt(100)
KRC_1_100 = np.zeros((50000,130))
density_1_100 = []*130
bins_use_1_100 = np.zeros((101,130))

for i in range(130):
    KRC_1_100[:,i] = data_1_100_I[:,31+i]
    count, bins, ignored = plt.hist(KRC_1_100[:,i], 100, density=True)
    bins_use_1_100[:,i] = bins


test_1_100 = np.mean(KRC_1_100, axis=0)
index_1_100 = []
for i in range(len(test_1_100)):
    if test_1_100[i] > 0.2 or test_3_100[i] < -0.2:
        index_1_100.append(i)
        
title_1_100 = ['$CH_3CH_2CH_3 --> CH_3CHCH_3 + H$', '$CH_3CH_2CH_3 --> CH_3CH_2CH_2 + H$', 
           '$CH_3CHCH_3 --> CH_3CHCH_2 + H$', '$CH_3CH_2CH_2 --> CH_3CHCH_2 + H$']
        
#Pt3Sn/Pt(111)
KRC_3_111 = np.zeros((50000,130))
density_3_111 = []*130
bins_use_3_111 = np.zeros((101,130))

for i in range(130):
    KRC_3_111[:,i] = data_3_111_I[:,31+i]
    count, bins, ignored = plt.hist(KRC_3_111[:,i], 100, density=True)
    bins_use_3_111[:,i] = bins


test_3_111 = np.mean(KRC_3_111, axis=0)
index_3_111 = []
for i in range(len(test_3_111)):
    if test_3_111[i] > 0.2 or test_3_100[i] < -0.2:
        index_3_111.append(i)
        
title_3_111 = ['$CH_3CH_2CH_3 --> CH_3CHCH_3 + H$', '$CH_3CH_2CH_3 --> CH_3CH_2CH_2 + H$', 
           '$CH_3CHCH_3 --> CH_3CHCH_2 + H$', '$CH_3CH_2CH_2 --> CH_3CHCH_2 + H$']

#Pt2Sn/Pt(211)
KRC_2_211 = np.zeros((50000,130))
density_2_211 = []*130
bins_use_2_211 = np.zeros((101,130))

for i in range(130):
    KRC_2_211[:,i] = data_2_211_I[:,31+i]
    count, bins, ignored = plt.hist(KRC_2_211[:,i], 100, density=True)
    bins_use_2_211[:,i] = bins


test_2_211 = np.mean(KRC_2_211, axis=0)
index_2_211 = []
for i in range(len(test_2_211)):
    if test_2_211[i] > 0.2 or test_3_100[i] < -0.2:
        index_2_211.append(i)
        
title_2_211 = ['$CH_3CH_2CH_3 --> CH_3CHCH_3 + H$', '$CH_3CH_2CH_3 --> CH_3CH_2CH_2 + H$', 
           '$CH_3CHCH_3 --> CH_3CHCH_2 + H$', '$CH_3CH_2CH_2 --> CH_3CHCH_2 + H$']


KRC_check_3_100 = data_3_100_I[:,31+j]
for i in range(50000):
    if np.isfinite(KRC_check_3_100)[i] == True:
        KRC_test.append(KRC_check[i])   




plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8)

plt.subplot(2,2,1)
# j =0
# for i in [0]:
    
density = stats.gaussian_kde(KRC_3_100[:,0])
plt.plot(bins_use_3_100[:,0], density(bins_use_3_100[:,0]), color='C0', lw ="2", label=title_3_100[0])
# j += 1
plt.xlabel('$Pt_3Sn/Pt(100)$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-1,2)
# plt.ylim(0,75)
# plt.grid(True)
# plt.legend(fontsize="8")
# plt.title('a.) Pt(100)', fontdict=font)
plt.tight_layout()
    
    
plt.subplot(2,2,2)
# j =0
# for i in [1,2,3]:
    
density = stats.gaussian_kde(KRC_1_100[:,1])
plt.plot(bins_use_1_100[:,1], density(bins_use_1_100[:,1]), color='C1', lw ="2", label=title_1_100[1])
density = stats.gaussian_kde(KRC_1_100[:,2])
plt.plot(bins_use_1_100[:,2], density(bins_use_1_100[:,2]), color='C2', lw ="2", label=title_1_100[2])
density = stats.gaussian_kde(KRC_1_100[:,3])
plt.plot(bins_use_1_100[:,3], density(bins_use_1_100[:,3]), color='C3', lw ="2", label=title_1_100[3])
# j += 1
plt.xlabel('$PtSn/Pt(100)$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-1,2)
# plt.ylim(0,75)
# plt.grid(True)
# plt.legend(fontsize="8")
# plt.title('a.) Pt(100)', fontdict=font)
plt.tight_layout()    
    
    
plt.subplot(2,2,3)
# j =0
# for i in [0,1,2]:
    
density = stats.gaussian_kde(KRC_3_111[:,0])
plt.plot(bins_use_3_111[:,0], density(bins_use_3_111[:,0]), color='C0', lw ="2", label=title_3_111[0])
density = stats.gaussian_kde(KRC_3_111[:,1])
plt.plot(bins_use_3_111[:,1], density(bins_use_3_111[:,1]), color='C1', lw ="2", label=title_3_111[1])
density = stats.gaussian_kde(KRC_3_111[:,2])
plt.plot(bins_use_3_111[:,2], density(bins_use_3_111[:,2]), color='C2', lw ="2", label=title_3_111[2])
# j += 1
plt.xlabel('$Pt_3Sn/Pt(111)$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-1,2)
# plt.ylim(0,75)
# plt.grid(True)
# plt.legend(fontsize="8")
# plt.title('a.) Pt(100)', fontdict=font)
plt.tight_layout()        
    
    
plt.subplot(2,2,4)
# j =0
# for i in [0,1]:
    
density = stats.gaussian_kde(KRC_2_211[:,0])
plt.plot(bins_use_2_211[:,0], density(bins_use_2_211[:,0]), color='C0', lw ="2", label=title_2_211[0])
density = stats.gaussian_kde(KRC_2_211[:,1])
plt.plot(bins_use_2_211[:,1], density(bins_use_2_211[:,1]), color='C1', lw ="2", label=title_2_211[1])
# j += 1
plt.xlabel('$Pt_2Sn/Pt(211)$', fontdict=font)
plt.ylabel('Probability Density', fontdict=font)
plt.xlim(-1,2)
# plt.ylim(0,75)
# plt.grid(True)
# plt.legend(fontsize="8")
# plt.title('a.) Pt(100)', fontdict=font)
plt.tight_layout()        


plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/PtSn_I_DKRC_PP.png', dpi=500)

#### Based on Propane

KRC_3_100 = np.zeros((50000,130))
density_3_100 = []*130
bins_use_3_100 = np.zeros((101,130))

for i in range(130):
    KRC_3_100[:,i] = data_3_100_I[:,31+i]
    count, bins, ignored = plt.hist(KRC[:,i], 100, density=True)
    bins_use[:,i] = bins


test = np.mean(KRC, axis=0)
index = []
for i in range(len(test)):
    if test[i] > 0.1:
        index.append(i)
        
        
        
        
        
        

title = ['$CH_3CH_2CH_3 --> CH_3CHCH_3 + H$', '$CH_3CH_2CH_3 --> CH_3CH_2CH_2 + H$']
j =0

plt.figure()
plt.rc('xtick', labelsize=8) 
plt.rc('ytick', labelsize=8)
    
for i in [0,1]:
    
    density = stats.gaussian_kde(KRC[:,i])
    plt.plot(bins_use[:,i], density(bins_use[:,i]), lw ="1", label=title[j])
    j += 1
    plt.xlabel('Degree of Kinetic rate control', fontdict=font)
    plt.ylabel('Probability Density', fontdict=font)
    plt.xlim(-1,2)
    # plt.ylim(0,75)
    plt.grid(True)
    plt.legend(fontsize="8")
    # plt.title('a.) Pt(100)', fontdict=font)
    plt.tight_layout()
    
    plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/Pt3Sn_F_DKRC_P.png', dpi=500)
    
    
#%% pairwise plot

#%% pairwise plot

sel = np.isfinite(selectivity_obs_3_100_I)
TOF =np.isfinite(TOF_obs_3_100_I_PP)

count = []
for i in range(50000):
    if (sel[i] == False or TOF[i] == False or selectivity_obs_3_100_I[i] > 1 or selectivity_obs_3_100_I[i] < 0) :
        count.append(i)
                     
sel_update = np.delete(selectivity_obs_3_100_I,count,axis=0)
TOF_update = np.delete(TOF_obs_3_100_I_PP,count,axis=0)

js = []
for i in range(len(sel_update)):
    if sel_update[i] > 1 :
        js.append(i)
        
 ##########       
sel_1_100 = np.isfinite(selectivity_obs_1_100_I)
TOF_1_100 =np.isfinite(TOF_obs_1_100_I_PP)

count = []
for i in range(50000):
    if (sel_1_100[i] == False or TOF_1_100[i] == False or selectivity_obs_1_100_I[i] > 1 or selectivity_obs_1_100_I[i] < 0 ) :
        count.append(i)
                     
sel_1_100_update = np.delete(selectivity_obs_1_100_I,count,axis=0)
TOF_1_100_update = np.delete(TOF_obs_1_100_I_PP,count,axis=0)        
        
js = []
for i in range(len(sel_1_100_update)):
    if sel_1_100_update[i] < 0 :
        js.append(i)  
        
        
 ##########       
sel_3_111 = np.isfinite(selectivity_obs_3_111_I)
TOF_3_111 =np.isfinite(TOF_obs_3_111_I_PP)

count = []
for i in range(50000):
    if (sel_3_111[i] == False or TOF_3_111[i] == False or selectivity_obs_3_111_I[i] > 1 or selectivity_obs_3_111_I[i] < 0 ) :
        count.append(i)
                     
sel_3_111_update = np.delete(selectivity_obs_3_111_I,count,axis=0)
TOF_3_111_update = np.delete(TOF_obs_3_111_I_PP,count,axis=0)        
        
js = []
for i in range(len(sel_3_111_update)):
    if sel_3_111_update[i] < 0 :
        js.append(i)  
        
        
 ##########       
sel_2_211 = np.isfinite(selectivity_obs_2_211_I)
TOF_2_211 =np.isfinite(TOF_obs_2_211_I_PP)

count = []
for i in range(50000):
    if (sel_2_211[i] == False or TOF_2_211[i] == False or selectivity_obs_2_211_I[i] > 1 or selectivity_obs_2_211_I[i] < 0 ) :
        count.append(i)
                     
sel_2_211_update = np.delete(selectivity_obs_2_211_I,count,axis=0)
TOF_2_211_update = np.delete(TOF_obs_2_211_I_PP,count,axis=0)        
        
js = []
for i in range(len(sel_2_211_update)):
    if sel_2_211_update[i] < 0 :
        js.append(i)                 


# selectivity = np.mean(selec_update,0)

plt.figure(figsize=(6,4))

plt.subplot(2,2,1)
plt.scatter(sel_update, TOF_update, s=5)

plt.xlabel('Selectivity to Propylene', fontdict=font)
plt.ylabel('$log_{10}(TOF_{Propylene})$', fontdict=font)
# plt.xlim(-0.1,1.1)
# plt.ylim(0,1)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('a.)', fontdict=font)
plt.tight_layout()

plt.subplot(2,2,2)
plt.scatter(sel_1_100_update, TOF_1_100_update, s=5)

plt.xlabel('Selectivity to Propylene', fontdict=font)
plt.ylabel('$log_{10}(TOF_{Propylene})$', fontdict=font)
# plt.xlim(-0.1,1.1)
# plt.ylim(0,1)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('b.)', fontdict=font)
plt.tight_layout()

plt.subplot(2,2,3)
plt.scatter(sel_3_111_update, TOF_3_111_update, s=5)

plt.xlabel('Selectivity to Propylene', fontdict=font)
plt.ylabel('$log_{10}(TOF_{Propylene})$', fontdict=font)
# plt.xlim(-0.1,1.1)
# plt.ylim(0,1)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('c.)', fontdict=font)
plt.tight_layout()

plt.subplot(2,2,4)
plt.scatter(sel_2_211_update, TOF_2_211_update, s=5)

plt.xlabel('Selectivity to Propylene', fontdict=font)
plt.ylabel('$log_{10}(TOF_{Propylene})$', fontdict=font)
# plt.xlim(-0.1,1.1)
# plt.ylim(0,1)
# plt.grid(True)
# plt.legend(fontsize="8")
plt.title('d.)', fontdict=font)
plt.tight_layout()


plt.savefig('C:/Users/mabello/Desktop/Research/UQ_paper/Plots/pairWise_cal.png', dpi=500)