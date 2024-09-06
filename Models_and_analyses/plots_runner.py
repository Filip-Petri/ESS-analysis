

from Load_data import *
#%% 
from DCF_analysis import *
from LD_class_and_real_revenue import *
from SD_class_and_real_revenue import *
from EWS_class_and_real_revenue import *


import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import os

import pyscipopt

#%% General params
OL = 24*8
f_disc = 0.08
f_tax = 0.22

tc = 147.3
tp = 12.1

#ST params
E_st = 10
C_st = 1
P_st = E_st*C_st
st_eff_u = 0.922
st_eff_d = 1/0.922
st_l_eff = 0.999
E_min_st = 0.2*E_st
st_cpx = -(394.37*E_st + 368.8*P_st)*1000
st_opx = 9.22*P_st*1000
st_cl = 5000

#LT params
E_lt = 25
C_lt = 1/5
P_lt = E_lt*C_lt
lt_eff_u = 0.825
lt_eff_d = 1/0.825
lt_l_eff = 0.998
E_min_lt = 0*E_lt
lt_cpx = -(410.88*E_lt + 414.69*P_lt)*1000
lt_opx = 12.36*P_lt*1000
lt_cl = 10000
K = C_lt

#Hydrogen params
P_h = 10
E_h = 1000
h_DMD = 100
E_min_lt = 0*E_lt
h_cpx = -(2000*P_h)*1000
h_opx = 20*P_h*1000
#%% Prices and activation
P2 = P.copy()
#P2["FCRN"] = GP23["FCR"]
P2["2024"] = 0
P2["H"] = np.ones(len((P)))*2

A2 = Act.copy()
# A2["FCRN_U"] = fcr_u
# A2["FCRN_D"] = fcr_d



#%%Short duration battery: base case
bs = st_Battery(OL)
#bs.excl_markets(["FCRD"])

bs.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,tp)
st_res,st_e0 = bs.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")

st_soe,st_rec_c,st_rec_d,st_rev,st_cyc = yr_rev_cyc_ST(bs, st_res, A2, P2,tc,tp)

st_soe2,st_rec_c2,st_rec_d2,st_rev2,st_cyc2 = yr_rev_cyc_ST(bs, st_res, A2, P2,tc,tp)



st_npv,st_irr,st_c = dcf(st_rev,10,0.08,f_tax,st_opx,st_cpx)



#npv,irr,c = dcf(rr,15,0.08,0.22,-9219.1*E*C,np.multiply(-394,E*10**3) - 368.76*10**3*E*C)

#%% Long duration battery: base case
bl = LT_Battery(OL)
bl.excl_markets(["FCRD"])

bl.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,tp)
bl.set_k(K)
lt_res,lt_e0 = bl.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
lt_soe,lt_rec_c,lt_rec_d,lt_rev,lt_cyc = yr_rev_cyc_LT(bl, lt_res, A2, P2,tc,tp)

#set A2=0, 
lt_soe2,lt_rec_c2,lt_rec_d2,lt_rev2,lt_cyc2 = yr_rev_cyc_LT(bl, lt_res, A2, P2,tc,tp)


lt_npv,lt_irr,lt_c = dcf(lt_rev,15,0.08,f_tax,lt_opx,lt_cpx)

#(lt_res["p_aFRR_U"][lt_res["p_aFRR_U"]>0.01][lt_res["p_aFRR_U"]<1])
#(lt_res["p_mFRR_U"][lt_res["p_mFRR_U"]>0.01][lt_res["p_mFRR_U"]<1])
#(lt_res["p_aFRR_D"][lt_res["p_aFRR_D"]>0.01][lt_res["p_aFRR_D"]<1])



#%% Hydrogen: base case


bh = H_storage_L(OL)
#bh.excl_markets(excM)
#bh.set_params(10, 1000, 1000/2, 100, 0, 0, 147.3)
bh.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, tc)
h_res,h_e0 = bh.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
h_soe,h_rev,h_dH = yr_rev_cyc_H(bh,h_res,A2,P2,tc)

h_soe2,h_rev2,h_dH2 = yr_rev_cyc_H(bh,h_res,A2,P2,tc)

h_npv,h_irr,h_c = dcf(h_rev,25,0.08,f_tax,h_opx,h_cpx)

#dcf(rr,15,0.08,0.22,-9219.1*E*C,np.multiply(-394,E*10**3) - 368.76*10**3*E*C)



#OL=opL, P_size=10, E_size=500, mD=100, T=147.3*Mps[i], price=P2, activ = A2, cpx = -1500*1000*10, opx = -20*10*1000, lft = 30, excM=["FCRD"],fcr = True)
#npv,irr,c = dcf(h_rev,30,0.08,0.22,-20*10*1000,-1500*1000*10)

#(rev,LT,r_DC,r_T,OPEX,CAPEX)

#%% Revenue comparison plots

#Add H column for comparison
st_rev["H"] = np.zeros(len(st_rev))
lt_rev["H"] = np.zeros(len(lt_rev))






plt.figure()
bardf = pd.DataFrame([np.sum(st_rev)/sum(np.sum(st_rev)[np.sum(st_rev)>0])  ,np.sum(lt_rev)/sum(np.sum(lt_rev)[np.sum(lt_rev)>0]), np.sum(h_rev)/sum(np.sum(h_rev)[np.sum(h_rev)>0]) ],index=["Li battery","VRF battery", "EwS System"])

bardf= bardf.rename(columns={"DA":"DA revenue", 
                      "FCRN": "FCR-N",
                      "FCRD_U": "FCR-D up",
                      "FCRD_D": "FCR-D down",
                      "aFRR_U": "aFRR up",
                      "aFRR_D": "aFRR down",
                      "mFRR_U": "mFRR up",
                      "Act": "Activations",
                      "Recovery": "SoE recovery",
                      "Tariffs": "Total tariffs",
                      "H":"Hydrogen"
                      }
             )


(bardf).plot(kind="bar",stacked="True",width=1,edgecolor="white",color=plt.get_cmap('tab20')(range(len(bardf.transpose()))) )
plt.xticks(rotation=45)
plt.ylabel("Normalized Revenue")
plt.xlabel("Technology")
plt.legend(bbox_to_anchor=(1,1,0,0))
plt.show()

plt.figure()
bardf2 = pd.DataFrame([np.sum(st_rev)  ,np.sum(lt_rev),np.sum(h_rev)],index=["St","Lt","H"])

bardf2= bardf2.rename(columns={"DA":"DA revenue", 
                      "FCRN": "FCR-N",
                      "FCRD_U": "FCR-D up",
                      "FCRD_D": "FCR-D down",
                      "aFRR_U": "aFRR up",
                      "aFRR_D": "aFRR down",
                      "mFRR_U": "mFRR up",
                      "Act": "Activations",
                      "Recovery": "SoE recovery",
                      "Tariffs": "Total tariffs",
                      "H":"Hydrogen"
                      }
             )

(bardf2).plot(kind="bar",stacked="True",width=1,edgecolor="white",color=plt.get_cmap('tab20')(range(len(bardf.transpose()))) )
plt.xticks(rotation=45)
plt.ylabel("Revenue [€]")
plt.xlabel("Technology")
plt.legend(bbox_to_anchor=(1,1,0,0))
plt.show()




#dcf(lt_rev,30,0.08,0.22,-9219.1*E_lt*C_lt,np.multiply(-394,E_lt*10**3) - 368.76*10**3*E_lt*C_lt)[1]


#%%



lt_res[lt_res.columns[[1,2,4,6]] ][0:24*7].plot(kind="bar",stacked=True,xticks=np.arange(0,200,25))
lt_res[lt_res.columns[[1,3,5]] ][0:24*7].plot(kind="bar",stacked=True,xticks=np.arange(0,200,25))



#%% soe recovery plots
fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(np.concatenate( ([E_lt*0.5],lt_res["e_b"]))[0:24*7],label="Modelled SoE",color="tab:blue")
ax[0].plot(np.concatenate( ([E_lt*0.5],lt_soe))[0:24*7],label="Realized SoE",color="tab:orange")
ax[0].plot(np.concatenate( ([E_lt*0.5],lt_res["e_b"]))[0:24*7]+E_lt*C_lt,label="Recovery threshold",color="tab:red",linestyle="--")
ax[0].plot(np.concatenate( ([E_lt*0.5],lt_res["e_b"]))[0:24*7]-E_lt*C_lt,color="tab:red",linestyle="--")
ax[0].set_ylabel("SoE [MWh]")
ax[1].set_xlabel("Time [h]")
ax[1].bar(range(24*7),lt_rec_c[0:24*7].values,color="tab:green",edgecolor="green",label="Recovery charged ")
ax[1].bar(range(24*7),lt_rec_d[0:24*7].values,color="tab:red",edgecolor="red",label="Recovery discharged")
ax[1].set_ylabel("Recovered Energy [MWh]")
ax[0].legend(bbox_to_anchor=(1,1,0,0))
ax[1].legend(bbox_to_anchor=(1,1,0.45,0))
plt.show()


fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(np.concatenate( ([E_st*0.5],st_res["e_b"]))[0:24*7],label="Modelled SoE",color="tab:blue")
ax[0].plot(np.concatenate( ([E_st*0.5],st_soe))[0:24*7],label="Realized SoE",color="tab:orange")

ax[0].set_ylabel("SoE [MWh]")
ax[1].set_xlabel("Time [h]")
ax[1].bar(range(24*7),st_rec_c[0:24*7].values,color="tab:green",edgecolor="green",label="Recovery charged ")
ax[1].bar(range(24*7),st_rec_d[0:24*7].values,color="tab:red",edgecolor="red",label="Recovery discharged")
ax[1].set_ylabel("Recovered Energy [MWh]")
ax[0].legend(bbox_to_anchor=(1,1,0,0))
ax[1].legend(bbox_to_anchor=(1,1,0.45,0))
plt.show()


fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(np.concatenate( ([E_h*0.5],h_res["e_b"]))[0:24*7],label="Modelled SoE",color="tab:blue")
ax[0].plot(np.concatenate( ([E_h*0.5],h_soe))[0:24*7],label="Realized SoE",color="tab:orange")



ax[0].set_ylabel("Hydrogen in storage [kg]")
ax[1].set_xlabel("Time [h]")
ndf = pd.DataFrame({"Modelled hydrogen sold":h_res["p_H"].values,"Correction to hydrogen sold":-h_dH.values})[0:24*7]
#pd.DataFrame({"Hydrogen sold":h_res["p_H"].values,"Corrected amount":-h_dH.values})[0:24*7].plot(kind="bar",stacked=True,color=["green","red"])
ndf.plot(kind="bar",stacked=True,color=["green","red"],ax=ax[1],label=[ndf.columns],xticks=np.arange(0,200,25))


#ax[1].bar(range(24*7),st_rec_c[0:24*7].values,color="tab:green",edgecolor="green",label="Recovery charged ")
a#x[1].bar(range(24*7),h_dH[0:24*7].values,color="tab:red",edgecolor="red",label="Recovery discharged")
ax[1].set_ylabel("Hydrogen sold [H]")
ax[0].legend(bbox_to_anchor=(1,1,0,0))
ax[1].legend(bbox_to_anchor=(1,1,0.55,0))
plt.show()



#%% Market stacking table
#set capex to 0 for no cycle cost (no degradation)
#st_cpx = 0
#lt_cpx = 0
table = np.zeros((3,4))
tablecyc = np.zeros((2,4))




noM = [[],["FCRD"],["FCRD","FCRN"],["FCRD","FCRN","aFRR","mFRR"]]
for i in range(4):
    #st
    bs = st_Battery(OL)
    bs.excl_markets(noM[i])
    
    bs.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,tp)
    st_res,st_e0 = bs.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    st_soe,st_rec_c,st_rec_d,st_rev,st_cyc = yr_rev_cyc_ST(bs, st_res, A2, P2,tc,tp)
    table[0,i] = sum(np.sum(st_rev))
    tablecyc[0,i] = st_cyc
    
    #LT
    bl = LT_Battery(OL)
    bl.excl_markets(noM[i])
    bl.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,tp)
    bl.set_k(K)
    lt_res,lt_e0 = bl.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    lt_soe,lt_rec_c,lt_rec_d,lt_rev,lt_cyc = yr_rev_cyc_LT(bl, lt_res, A2, P2,tc,tp)
    table[1,i] = sum(np.sum(lt_rev))
    tablecyc[1,i] = lt_cyc
"""    
    #Hydrogen
    bh = H_storage_L(OL)
    bh.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, tc)
    bh.excl_markets(noM[i])
    h_res,h_e0 = bh.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    h_soe,h_rev,h_dH = yr_rev_cyc_H(bh,h_res,A2,P2,tc)
    table[2,i] = sum(np.sum(h_rev))
"""
print(pd.DataFrame(table,columns=["All","No FCRD","No FCR","Only DA"]))
print(pd.DataFrame(tablecyc,columns=["All","No FCRD","No FCR","Only DA"]))

#%% Technology efficiency (cyclelife/lifetime?)




#%% Sensitivity: Technology capex, opex and tariffs (DK2)

mps = [0.01,0.25,0.5,0.75,1,1.10,1.25]
st_irrs = np.zeros((4,len(mps)))
lt_irrs = np.zeros((4,len(mps)))
h_irrs = np.zeros((4,len(mps)))

for i in range(len(mps)):
    #short
    bs = st_Battery(OL)
    #bs.excl_markets(["FCRD"])
    bs.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx*mps[i],tc,tp)
    st_res,st_e0 = bs.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    st_soe,st_rec_c,st_rec_d,st_rev,st_cyc = yr_rev_cyc_ST(bs, st_res, A2, P2,tc,tp)
    st_npv,st_irr,st_c = dcf(st_rev,10,0.08,0.22,st_opx,st_cpx*mps[i])
    st_irrs[0,i] = st_irr
    
    
    #long
    bl = LT_Battery(OL)
    bl.excl_markets(["FCRD"])
    bl.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx*mps[i],tc,tp)
    bl.set_k(K)
    lt_res,lt_e0 = bl.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    lt_soe,lt_rec_c,lt_rec_d,lt_rev,lt_cyc = yr_rev_cyc_LT(bl, lt_res, A2, P2,tc,tp)
    lt_npv,lt_irr,lt_c = dcf(lt_rev,15,0.08,0.22,lt_opx,lt_cpx*mps[i])
    lt_irrs[0,i] = lt_irr
    
    
    #hydrogen
    bh = H_storage_L(OL)
    #bh.excl_markets(excM)
    #bh.set_params(10, 1000, 1000/2, 100, 0, 0, 147.3)
    bh.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx*mps[i], tc)
    h_res,h_e0 = bh.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    h_soe,h_rev,h_dH = yr_rev_cyc_H(bh,h_res,A2,P2,tc)
    h_npv,h_irr,h_c = dcf(h_rev,25,0.08,0.22,h_opx,h_cpx*mps[i])
    h_irrs[0,i] = h_irr
    
        
print("0")


bs = st_Battery(OL)
#bs.excl_markets(["FCRD"])
bs.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,tp)
st_res,st_e0 = bs.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
st_soe,st_rec_c,st_rec_d,st_rev,st_cyc = yr_rev_cyc_ST(bs, st_res, A2, P2,tc,tp)

#long
bl = LT_Battery(OL)
bl.excl_markets(["FCRD"])
bl.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,tp)
bl.set_k(K)
lt_res,lt_e0 = bl.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
lt_soe,lt_rec_c,lt_rec_d,lt_rev,lt_cyc = yr_rev_cyc_LT(bl, lt_res, A2, P2,tc,tp)

#hydrogen
bh = H_storage_L(OL)
#bh.excl_markets(excM)
bh.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, tc)
h_res,h_e0 = bh.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
h_soe,h_rev,h_dH = yr_rev_cyc_H(bh,h_res,A2,P2,tc)

for i in range(len(mps)):
    st_npv,st_irr,st_c = dcf(st_rev,10,0.08,0.22,st_opx*mps[i],st_cpx)
    st_irrs[1,i] = st_irr
    lt_npv,lt_irr,lt_c = dcf(lt_rev,15,0.08,0.22,lt_opx*mps[i],lt_cpx)
    lt_irrs[1,i] = lt_irr
    h_npv,h_irr,h_c = dcf(h_rev,25,0.08,0.22,h_opx*mps[i],h_cpx)
    h_irrs[1,i] = h_irr
print("1")


for i in range(len(mps)):
    #short
    bs = st_Battery(OL)
    #bs.excl_markets(["FCRD"])
    bs.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc*mps[i],tp)
    st_res,st_e0 = bs.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    st_soe,st_rec_c,st_rec_d,st_rev,st_cyc = yr_rev_cyc_ST(bs, st_res, A2, P2,tc*mps[i],tp)
    st_npv,st_irr,st_c = dcf(st_rev,10,0.08,0.22,st_opx,st_cpx)
    st_irrs[2,i] = st_irr
    
    #long
    bl = LT_Battery(OL)
    bl.excl_markets(["FCRD"])
    bl.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc*mps[i],tp)
    bl.set_k(K)
    try:
        lt_res,lt_e0 = bl.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
        lt_soe,lt_rec_c,lt_rec_d,lt_rev,lt_cyc = yr_rev_cyc_LT(bl, lt_res, A2, P2,tc*mps[i],tp)
        lt_npv,lt_irr,lt_c = dcf(lt_rev,15,0.08,0.22,lt_opx,lt_cpx)
        lt_irrs[2,i] = lt_irr
    except:
        print("error for lt: tc=",mps[i]*tc)
        lt_irrs[2,i] = np.nan
    #hydrogen
    bh = H_storage_L(OL)
    #bh.excl_markets(excM)
    #bh.set_params(10, 1000, 1000/2, 100, 0, 0, 147.3)
    bh.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, tc*mps[i])
    h_res,h_e0 = bh.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    h_soe,h_rev,h_dH = yr_rev_cyc_H(bh,h_res,A2,P2,tc*mps[i])
    h_npv,h_irr,h_c = dcf(h_rev,25,0.08,0.22,h_opx,h_cpx)
    h_irrs[2,i] = h_irr
print("2")


for i in range(len(mps)):
    #short
    bs = st_Battery(OL)
    #bs.excl_markets(["FCRD"])
    bs.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,tp*mps[i])
    st_res,st_e0 = bs.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    st_soe,st_rec_c,st_rec_d,st_rev,st_cyc = yr_rev_cyc_ST(bs, st_res, A2, P2,tc,tp*mps[i])
    st_npv,st_irr,st_c = dcf(st_rev,10,0.08,0.22,st_opx,st_cpx)
    st_irrs[3,i] = st_irr
    
    #long
    bl = LT_Battery(OL)
    bl.excl_markets(["FCRD"])
    bl.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,tp*mps[i])
    bl.set_k(K)
    lt_res,lt_e0 = bl.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
    lt_soe,lt_rec_c,lt_rec_d,lt_rev,lt_cyc = yr_rev_cyc_LT(bl, lt_res, A2, P2,tc,tp*mps[i])
    lt_npv,lt_irr,lt_c = dcf(lt_rev,15,0.08,0.22,lt_opx,lt_cpx)
    lt_irrs[3,i] = lt_irr

print("3")


plt.figure()
fig, ax = plt.subplots(3, sharex=True)
fig.tight_layout()
ax[2].set_xlabel("Parameter change [%]")
ax[1].set_ylabel("Change in IRR [%]")

ax[0].set_title("ST battery (LI)")
ax[0].plot(-(np.ones(7)-mps)*100,(st_irrs[0,:]-st_irrs[0,4])*100,label="CAPEX")
ax[0].plot(-(np.ones(7)-mps)*100,(st_irrs[1,:]-st_irrs[1,4])*100,label="OPEX")
ax[0].plot(-(np.ones(7)-mps)*100,(st_irrs[2,:]-st_irrs[2,4])*100,label="Consumer tariffs")
ax[0].plot(-(np.ones(7)-mps)*100,(st_irrs[3,:]-st_irrs[3,4])*100,label="Producer tariffs")
ax[0].scatter(0,0,label="Base case",color="gray")
#ax[0].axvline(0,color="gray",linestyle="--")
ax[0].set_ylim(-0.05*100,0.2*100)

ax[1].set_title("LT battery (VRF)")
ax[1].plot(-(np.ones(7)-mps)*100,(lt_irrs[0,:]-lt_irrs[0,4])*100,label="CAPEX")
ax[1].plot(-(np.ones(7)-mps)*100,(lt_irrs[1,:]-lt_irrs[1,4])*100,label="OPEX")
ax[1].plot((-(np.ones(7)-mps)*100)[[0,2,3,4,5,6]],(lt_irrs[2,[0,2,3,4,5,6]]-lt_irrs[2,4])*100,label="Consumer tariffs")
ax[1].plot(-(np.ones(7)-mps)*100,(lt_irrs[3,:]-lt_irrs[3,4])*100,label="Producer tariffs")
ax[1].scatter(0,0,label="Base case",color="gray")
ax[1].set_ylim(-0.05*100,0.2*100)

ax[2].set_title("EwS system (Alkaline)")
ax[2].plot(-(np.ones(7)-mps)*100,(h_irrs[0,:]-h_irrs[0,4])*100,label="CAPEX")
ax[2].plot(-(np.ones(7)-mps)*100,(h_irrs[1,:]-h_irrs[1,4])*100,label="OPEX")
ax[2].plot(-(np.ones(7)-mps)*100,(h_irrs[2,:]-h_irrs[2,4])*100,label="Consumer tariffs")
ax[2].scatter(0,0,label="Base case",color="gray")
#ax[1].plot(-(np.ones(7)-mps)*100,lt_irrs[3,:],label="Producer tariffs")
ax[2].set_ylim(-0.05*100,0.2*100)
ax[0].legend(bbox_to_anchor=(1,1,0,0))
plt.suptitle("DK2 Current Conditions",weight="bold")
fig.subplots_adjust(top=0.84)



# plt.plot(-(np.ones(7)-mps)*100,lt_irrs[0,:],label="CAPEX")
# plt.plot(-(np.ones(7)-mps)*100,lt_irrs[1,:],label="OPEX")
# plt.plot(-(np.ones(7)-mps)*100,lt_irrs[2,:],label="Consumer tariffs")
# plt.plot(-(np.ones(7)-mps)*100,lt_irrs[3,:],label="Producer tariffs")
# plt.ylim(0.05,0.40)


#%% FCR runs
P3 = P.copy()
P3["FCRN"] = GP23["FCR"]
P3["2024"] = 0
P3["H"] = np.ones(len((P)))*2

A3 = Act.copy()
A3["FCRN_U"] = fcr_u
A3["FCRN_D"] = fcr_d




#%%Short duration battery: deescalation
#fcr
bs_fcr = FCR_st_Battery(OL)
bs_fcr.excl_markets(["FCRD"])
bs_fcr.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,tp)
st_res_fcr,st_e0_fcr = bs_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
st_soe_fcr,st_rec_c_fcr,st_rec_d_fcr,st_rev_fcr,st_cyc_fcr = yr_rev_cyc_ST(bs_fcr, st_res_fcr, A3, P3,tc,tp)

st_npv_fcr,st_irr_fcr,st_c_fcr = dcf(st_rev_fcr,10,0.08,0.22,st_opx,st_cpx)
#dk2
# bs = st_Battery(OL)
# bs.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,tp)
# st_res,st_e0 = bs.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
# st_soe,st_rec_c,st_rec_d,st_rev,st_cyc = yr_rev_cyc_ST(bs, st_res, A2, P2,tc,tp)


# st_npv_scen,st_irr_scen,st_c_scen = dcf_de(st_rev,15,0.08,0.22,12.36*1000,st_cpx,st_rev_fcr,5)



# sum(np.sum(st_rev))
# sum(np.sum(st_rev_fcr))
# st_npv_fcr,st_irr_fcr,st_c_fcr = dcf_de(st_rev,15,0.08,0.22,9.22*1000,st_cpx,st_rev_fcr,3)


#%% Long duration battery: deesc
#fcr
bl_fcr = FCR_LT_Battery(OL)
bl_fcr.excl_markets(["FCRD"])

bl_fcr.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,tp)
bl_fcr.set_k(K)
lt_res_fcr,lt_e0_fcr = bl_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
lt_soe_fcr,lt_rec_c_fcr,lt_rec_d_fcr,lt_rev_fcr,lt_cyc_fcr = yr_rev_cyc_LT(bl_fcr, lt_res_fcr, A3, P3,tc,tp)

lt_npv_fcr,lt_irr_fcr,lt_c_fcr = dcf(lt_rev_fcr,15,0.08,0.22,lt_opx,lt_cpx)
#dk2
# bl = LT_Battery(OL)
# bl.excl_markets(["FCRD"])

# bl.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,tp)
# bl.set_k(K)
# lt_res,lt_e0 = bl.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
# lt_soe,lt_rec_c,lt_rec_d,lt_rev,lt_cyc = yr_rev_cyc_LT(bl, lt_res, A2, P2,tc,tp)



#lt_npv_scen,lt_irr_scen,lt_c_scen = dcf_de(lt_rev,15,0.08,0.22,12.36*1000,lt_cpx,lt_rev_fcr,5)




#(lt_res["p_aFRR_U"][lt_res["p_aFRR_U"]>0.01][lt_res["p_aFRR_U"]<1])
#(lt_res["p_mFRR_U"][lt_res["p_mFRR_U"]>0.01][lt_res["p_mFRR_U"]<1])
#(lt_res["p_aFRR_D"][lt_res["p_aFRR_D"]>0.01][lt_res["p_aFRR_D"]<1])




#%% Hydrogen: deesc
#fcr
#tc1 = tc*0
bh_fcr = FCR_H_storage_L(OL)
bh_fcr.excl_markets(["FCRD"])
bh_fcr.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, tc1)
h_res_fcr,h_e0_fcr = bh_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
h_soe_fcr,h_rev_fcr,h_dH_fcr = yr_rev_cyc_H(bh_fcr,h_res_fcr,A3,P3,tc1)

h_npv_fcr,h_irr_fcr,h_c_fcr = dcf(h_rev_fcr,25,0.08,0.22,h_opx,h_cpx)
#dk2
# bh = H_storage_L(OL)
# bh.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, tc)
# h_res,h_e0 = bh.run_yr(P2,A2,start_d="2023-01-01",end_d="2023-12-31 12")
# h_soe,h_rev,h_dH = yr_rev_cyc_H(bh,h_res,A2,P2,tc)


#h_npv_scen,h_irr_scen,h_c_scen = dcf_de(h_rev,15,0.08,0.22,12.36*1000,lt_cpx,h_rev_fcr,10)



#%% Revenue dist fcr
np.sum(st_rev_fcr)
(lt_rev_fcr)
(h_rev_fcr)


plt.figure()
bardf = pd.DataFrame([np.sum(st_rev_fcr)/sum(np.sum(st_rev_fcr)[np.sum(st_rev_fcr)>0])  ,np.sum(lt_rev_fcr)/sum(np.sum(lt_rev_fcr)[np.sum(lt_rev_fcr)>0]), np.sum(h_rev_fcr)/sum(np.sum(h_rev_fcr)[np.sum(h_rev_fcr)>0]) ],index=["Li battery","VRF battery", "EwS System"])

bardf= bardf.rename(columns={"DA":"DA revenue", 
                      "FCRN": "FCR",
                      "FCRD_U": "FCR-D up",
                      "FCRD_D": "FCR-D down",
                      "aFRR_U": "aFRR up",
                      "aFRR_D": "aFRR down",
                      "mFRR_U": "mFRR up",
                      "Act": "Activations",
                      "Recovery": "SoE recovery",
                      "Tariffs": "Total tariffs",
                      "H":"Hydrogen"
                      }
             )


(bardf).plot(kind="bar",stacked="True",width=1,edgecolor="white",color=plt.get_cmap('tab20')(range(len(bardf.transpose()))) )
plt.xticks(rotation=45)
plt.ylabel("Normalized Revenue")
plt.xlabel("Technology")
plt.legend(bbox_to_anchor=(1,1,0,0))
plt.show()

plt.figure()
bardf2 = pd.DataFrame([np.sum(st_rev_fcr)  ,np.sum(lt_rev_fcr),np.sum(h_rev_fcr)],index=["Li battery","VRF battery", "EwS System"])

bardf2= bardf2.rename(columns={"DA":"DA revenue", 
                      "FCRN": "FCR",
                      "FCRD_U": "FCR-D up",
                      "FCRD_D": "FCR-D down",
                      "aFRR_U": "aFRR up",
                      "aFRR_D": "aFRR down",
                      "mFRR_U": "mFRR up",
                      "Act": "Activations",
                      "Recovery": "SoE recovery",
                      "Tariffs": "Total tariffs",
                      "H":"Hydrogen"
                      }
             )
bardf2 = bardf2.drop(['FCR-D up','FCR-D down'],axis=1)

(bardf2).plot(kind="bar",stacked="True",width=1,edgecolor="white",color=plt.get_cmap('tab20')(range(len(bardf.transpose())))[[0,1,4,5,6,7,8,9,10]] )
plt.xticks(rotation=45)
plt.ylabel("Revenue [€]")
plt.xlabel("Technology")
plt.legend(bbox_to_anchor=(1,1,0,0))
plt.show()






#%% Technology capex, opex and tariffs (FCR)

mps = [0.01,0.25,0.5,0.75,1,1.10,1.25]
st_irrs_fcr = np.zeros((4,len(mps)))
lt_irrs_fcr = np.zeros((4,len(mps)))
h_irrs_fcr = np.zeros((4,len(mps)))

for i in range(len(mps)):
    #short
    bs_fcr = FCR_st_Battery(OL)
    bs_fcr.excl_markets(["FCRD"])
    bs_fcr.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx*mps[i],tc,tp)
    st_res_fcr,st_e0_fcr = bs_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    st_soe_fcr,st_rec_c_fcr,st_rec_d_fcr,st_rev_fcr,st_cyc_fcr = yr_rev_cyc_ST(bs_fcr, st_res_fcr, A3, P3,tc,tp)
    st_npv_fcr,st_irr_fcr,st_c_fcr = dcf(st_rev_fcr,10,0.08,0.22,st_opx,st_cpx*mps[i])
    st_irrs_fcr[0,i] = st_irr_fcr
    
    
    #long
    bl_fcr = FCR_LT_Battery(OL)
    bl_fcr.excl_markets(["FCRD"])
    bl_fcr.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx*mps[i],tc,tp)
    bl_fcr.set_k(K)
    lt_res_fcr,lt_e0_fcr = bl_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    lt_soe_fcr,lt_rec_c_fcr,lt_rec_d_fcr,lt_rev_fcr,lt_cyc_fcr = yr_rev_cyc_LT(bl_fcr, lt_res_fcr, A3, P3,tc,tp)
    lt_npv_fcr,lt_irr_fcr,lt_c_fcr = dcf(lt_rev_fcr,15,0.08,0.22,lt_opx,lt_cpx*mps[i])
    lt_irrs_fcr[0,i] = lt_irr_fcr
    
    
    #hydrogen
    bh_fcr = FCR_H_storage_L(OL)
    bh_fcr.excl_markets(["FCRD"])
    #bh.set_params(10, 1000, 1000/2, 100, 0, 0, 147.3)
    bh_fcr.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx*mps[i], tc)
    h_res_fcr,h_e0_fcr = bh_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    h_soe_fcr,h_rev_fcr,h_dH_fcr = yr_rev_cyc_H(bh_fcr,h_res_fcr,A3,P3,tc)
    h_npv_fcr,h_irr_fcr,h_c_fcr = dcf(h_rev_fcr,25,0.08,0.22,h_opx,h_cpx*mps[i])
    h_irrs_fcr[0,i] = h_irr_fcr
    
        
print("0")


bs_fcr = FCR_st_Battery(OL)
bs_fcr.excl_markets(["FCRD"])
bs_fcr.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,tp)
st_res_fcr,st_e0_fcr = bs_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
st_soe_fcr,st_rec_c_fcr,st_rec_d_fcr,st_rev_fcr,st_cyc_fcr = yr_rev_cyc_ST(bs_fcr, st_res_fcr, A3, P3,tc,tp)

#long
bl_fcr = FCR_LT_Battery(OL)
bl_fcr.excl_markets(["FCRD"])
bl_fcr.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,tp)
bl_fcr.set_k(K)
lt_res_fcr,lt_e0_fcr = bl_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
lt_soe_fcr,lt_rec_c_fcr,lt_rec_d_fcr,lt_rev_fcr,lt_cyc_fcr = yr_rev_cyc_LT(bl_fcr, lt_res_fcr, A3, P3,tc,tp)

#hydrogen
bh_fcr = FCR_H_storage_L(OL)
bh_fcr.excl_markets(["FCRD"])
bh_fcr.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, tc)
h_res_fcr,h_e0_fcr = bh_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
h_soe_fcr,h_rev_fcr,h_dH_fcr = yr_rev_cyc_H(bh_fcr,h_res_fcr,A3,P3,tc)

for i in range(len(mps)):
    st_npv_fcr,st_irr_fcr,st_c_fcr = dcf(st_rev_fcr,10,0.08,0.22,st_opx*mps[i],st_cpx)
    st_irrs_fcr[1,i] = st_irr_fcr
    lt_npv_fcr,lt_irr_fcr,lt_c_fcr = dcf(lt_rev_fcr,15,0.08,0.22,lt_opx*mps[i],lt_cpx)
    lt_irrs_fcr[1,i] = lt_irr_fcr
    h_npv_fcr,h_irr_fcr,h_c_fcr = dcf(h_rev_fcr,25,0.08,0.22,h_opx*mps[i],h_cpx)
    h_irrs_fcr[1,i] = h_irr_fcr
print("1")


for i in range(len(mps)):
    #short
    bs_fcr = FCR_st_Battery(OL)
    bs_fcr.excl_markets(["FCRD"])
    bs_fcr.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc*mps[i],tp)
    st_res_fcr,st_e0_fcr = bs_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    st_soe_fcr,st_rec_c_fcr,st_rec_d_fcr,st_rev_fcr,st_cyc_fcr = yr_rev_cyc_ST(bs_fcr, st_res_fcr, A3, P3,tc*mps[i],tp)
    st_npv_fcr,st_irr_fcr,st_c_fcr = dcf(st_rev_fcr,10,0.08,0.22,st_opx,st_cpx)
    st_irrs_fcr[2,i] = st_irr_fcr
    
    #long
    bl_fcr = FCR_LT_Battery(OL)
    bl_fcr.excl_markets(["FCRD"])
    bl_fcr.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc*mps[i],tp)
    bl_fcr.set_k(K)
    try:
        lt_res_fcr,lt_e0_fcr = bl_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
        lt_soe_fcr,lt_rec_c_fcr,lt_rec_d_fcr,lt_rev_fcr,lt_cyc_fcr = yr_rev_cyc_LT(bl_fcr, lt_res_fcr, A3, P3,tc*mps[i],tp)
        lt_npv_fcr,lt_irr_fcr,lt_c_fcr = dcf(lt_rev_fcr,15,0.08,0.22,lt_opx,lt_cpx)
        lt_irrs_fcr[2,i] = lt_irr_fcr
    except:
        print("error lt: tc=",mps[i]*tc)
        lt_irrs_fcr[2,i] = np.nan
    
    #hydrogen
    bh_fcr = FCR_H_storage_L(OL)
    bh_fcr.excl_markets(["FCRD"])
    #bh.set_params(10, 1000, 1000/2, 100, 0, 0, 147.3)
    bh_fcr.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, tc*mps[i])
    h_res_fcr,h_e0_fcr = bh_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    h_soe_fcr,h_rev_fcr,h_dH_fcr = yr_rev_cyc_H(bh_fcr,h_res_fcr,A3,P3,tc*mps[i])
    h_npv_fcr,h_irr_fcr,h_c_fcr = dcf(h_rev_fcr,25,0.08,0.22,h_opx,h_cpx)
    h_irrs_fcr[2,i] = h_irr_fcr
print("2")


for i in range(len(mps)):
    #short
    bs_fcr = FCR_st_Battery(OL)
    bs_fcr.excl_markets(["FCRD"])
    bs_fcr.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,tp*mps[i])
    st_res_fcr,st_e0_fcr = bs_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    st_soe_fcr,st_rec_c_fcr,st_rec_d_fcr,st_rev_fcr,st_cyc_fcr = yr_rev_cyc_ST(bs_fcr, st_res_fcr, A3, P3,tc,tp*mps[i])
    st_npv_fcr,st_irr_fcr,st_c_fcr = dcf(st_rev_fcr,10,0.08,0.22,st_opx,st_cpx)
    st_irrs_fcr[3,i] = st_irr_fcr
    
    #long
    bl_fcr = FCR_LT_Battery(OL)
    bl_fcr.excl_markets(["FCRD"])
    bl_fcr.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,tp*mps[i])
    bl_fcr.set_k(K)
    lt_res_fcr,lt_e0_fcr = bl_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    lt_soe_fcr,lt_rec_c_fcr,lt_rec_d_fcr,lt_rev_fcr,lt_cyc_fcr = yr_rev_cyc_LT(bl_fcr, lt_res_fcr, A3, P3,tc,tp*mps[i])
    lt_npv_fcr,lt_irr_fcr,lt_c_fcr = dcf(lt_rev_fcr,15,0.08,0.22,lt_opx,lt_cpx)
    lt_irrs_fcr[3,i] = lt_irr_fcr

print("3")



plt.figure()
fig, ax = plt.subplots(2, sharex=True)
fig.tight_layout()
ax[1].set_xlabel("Parameter change [%]")
ax[1].set_ylabel("IRR [%]")
ax[0].set_ylabel("IRR [%]")

ax[0].set_xlim(-60,0.2*100)
ax[1].set_xlim(-60,0.2*100)


ax[0].set_title("ST battery (LI)")
ax[0].plot(-(np.ones(7)-mps)*100,(st_irrs_fcr[0,:])*100,label="CAPEX")
ax[0].plot(-(np.ones(7)-mps)*100,(st_irrs_fcr[1,:])*100,label="OPEX")
#ax[0].plot(-(np.ones(7)-mps)*100,(st_irrs_fcr[2,:]-st_irrs_fcr[2,4])*100,label="Consumer tariffs")
#ax[0].plot(-(np.ones(7)-mps)*100,(st_irrs_fcr[3,:]-st_irrs_fcr[3,4])*100,label="Producer tariffs")
ax[0].scatter(0,st_irrs_fcr[0,4]*100,color="grey",label="Original Estimate")
ax[0].set_ylim(0.05*100,0.35*100)

ax[1].set_title("LT battery (VRF)")
ax[1].plot(-(np.ones(7)-mps)*100,(lt_irrs_fcr[0,:])*100,label="CAPEX")
ax[1].plot(-(np.ones(7)-mps)*100,(lt_irrs_fcr[1,:])*100,label="OPEX")
#ax[1].plot((-(np.ones(7)-mps)*100)[[1,2,3,4,5,6]],(lt_irrs_fcr[2,[1,2,3,4,5,6]]-lt_irrs_fcr[2,4])*100,label="Consumer tariffs")
#ax[1].plot(-(np.ones(7)-mps)*100,(lt_irrs_fcr[3,:]-lt_irrs_fcr[3,4])*100,label="Producer tariffs")
ax[1].scatter(0,lt_irrs_fcr[0,4]*100,color="grey",label="Original Estimate")
ax[1].set_ylim(0.05*100,0.25*100)


# ax[2].set_title("EwS system (Alkaline)")
# ax[2].plot(-(np.ones(7)-mps)*100,(h_irrs_fcr[0,:]-h_irrs_fcr[0,4])*100,label="CAPEX")
# ax[2].plot(-(np.ones(7)-mps)*100,(h_irrs_fcr[1,:]-h_irrs_fcr[1,4])*100,label="OPEX")
# ax[2].plot(-(np.ones(7)-mps)*100,(h_irrs_fcr[2,:]-h_irrs_fcr[2,4])*100,label="Consumer tariffs")

# ax[2].set_ylim(-0.2*100,0.2*100)
ax[0].legend(bbox_to_anchor=(1.35,1.05,0,0))
#plt.suptitle("Future FCR Conditions",weight="bold")
fig.subplots_adjust(top=0.84)


#%% Tariff impact (FCR)

ac = [0,12.1,50,100,147,200]
abs_tp_irr = np.zeros((2,len(ac)))

#producer tariffs
for i in range(len(ac)):
    #short
    bs_fcr = FCR_st_Battery(OL)
    bs_fcr.excl_markets(["FCRD"])
    bs_fcr.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,ac[i])
    st_res_fcr,st_e0_fcr = bs_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    st_soe_fcr,st_rec_c_fcr,st_rec_d_fcr,st_rev_fcr,st_cyc_fcr = yr_rev_cyc_ST(bs_fcr, st_res_fcr, A3, P3,tc,ac[i])
    st_npv_fcr,st_irr_fcr,st_c_fcr = dcf(st_rev_fcr,10,0.08,0.22,st_opx,st_cpx)
    abs_tp_irr[0,i] = st_irr_fcr
    #print(abs_tp_irr[0,i])
    #long
    bl_fcr = FCR_LT_Battery(OL)
    bl_fcr.excl_markets(["FCRD"])
    bl_fcr.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,ac[i])
    bl_fcr.set_k(K)
    lt_res_fcr,lt_e0_fcr = bl_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    lt_soe_fcr,lt_rec_c_fcr,lt_rec_d_fcr,lt_rev_fcr,lt_cyc_fcr = yr_rev_cyc_LT(bl_fcr, lt_res_fcr, A3, P3,tc,ac[i])
    lt_npv_fcr,lt_irr_fcr,lt_c_fcr = dcf(lt_rev_fcr,15,0.08,0.22,lt_opx,lt_cpx)
    abs_tp_irr[1,i] = lt_irr_fcr

    print(abs_tp_irr[1,i])
    
    
abs_tc_irr =np.zeros((3,len(ac)))




for i in range(len(ac)):
    #short
    bs_fcr = FCR_st_Battery(OL)
    bs_fcr.excl_markets(["FCRD"])
    bs_fcr.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,ac[i],tp)
    st_res_fcr,st_e0_fcr = bs_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    st_soe_fcr,st_rec_c_fcr,st_rec_d_fcr,st_rev_fcr,st_cyc_fcr = yr_rev_cyc_ST(bs_fcr, st_res_fcr, A3, P3,ac[i],tp)
    st_npv_fcr,st_irr_fcr,st_c_fcr = dcf(st_rev_fcr,10,0.08,0.22,st_opx,st_cpx)
    abs_tc_irr[0,i] = st_irr_fcr
    #print(abs_tp_irr[0,i])
    #long
    bl_fcr = FCR_LT_Battery(OL)
    bl_fcr.excl_markets(["FCRD"])
    bl_fcr.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,ac[i],tp)
    bl_fcr.set_k(K)
    lt_res_fcr,lt_e0_fcr = bl_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    lt_soe_fcr,lt_rec_c_fcr,lt_rec_d_fcr,lt_rev_fcr,lt_cyc_fcr = yr_rev_cyc_LT(bl_fcr, lt_res_fcr, A3, P3,ac[i],tp)
    lt_npv_fcr,lt_irr_fcr,lt_c_fcr = dcf(lt_rev_fcr,15,0.08,0.22,lt_opx,lt_cpx)
    abs_tc_irr[1,i] = lt_irr_fcr
    
    bh_fcr = FCR_H_storage_L(OL)
    bh_fcr.excl_markets(["FCRD"])
    #bh.set_params(10, 1000, 1000/2, 100, 0, 0, 147.3)
    bh_fcr.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, ac[i])
    h_res_fcr,h_e0_fcr = bh_fcr.run_yr(P3,A3,start_d="2023-01-01",end_d="2023-12-31 12")
    h_soe_fcr,h_rev_fcr,h_dH_fcr = yr_rev_cyc_H(bh_fcr,h_res_fcr,A3,P3,ac[i])
    h_npv_fcr,h_irr_fcr,h_c_fcr = dcf(h_rev_fcr,25,0.08,0.22,h_opx,h_cpx)
    abs_tc_irr[2,i] = h_irr_fcr


plt.figure()
plt.plot(ac,abs_tc_irr[0,:]*100,color="red",label="Consumer tariff")
plt.plot(ac,abs_tp_irr[0,:]*100,color="blue",label="Producer tariff")
plt.axhline(abs_tp_irr[0,1]*100,label="IRR with current tariffs",color="gray",linestyle="--")
plt.legend()

plt.figure()

plt.plot(ac,abs_tc_irr[1,:]*100,color="red",label="Consumer tariff")
plt.plot(ac,abs_tp_irr[1,:]*100,color="blue",label="Producer tariff")
plt.axhline(abs_tp_irr[1,1]*100,label="Current tariffs",color="gray",linestyle="--")
plt.legend()


plt.figure()
plt.plot(ac,abs_tc_irr[2,:]*100,color="red",label="Consumer tariff")
#plt.plot(ac,abs_tp_irr[1,:]*100,color="blue",label="Producer tariff")
#plt.axhline(abs_tc_irr[1,-2]*100,label="Current tariffs",color="gray",linestyle="--")
plt.legend()




plt.figure()
fig, ax = plt.subplots(3, sharex=True)
fig.tight_layout()
ax[2].set_xlabel("Tariff [EUR/MWh]")
ax[2].set_ylabel("Change in IRR [%]")

ax[0].set_title("ST battery (LI)")
ax[0].plot(ac,abs_tc_irr[0,:]*100,color="red",label="Consumer tariff")
ax[0].plot(ac,abs_tp_irr[0,:]*100,color="blue",label="Producer tariff")
ax[0].axhline(abs_tp_irr[0,1]*100,label="IRR with current tariffs",color="gray",linestyle="--")
ax[0].legend(bbox_to_anchor=(1,1,0,0))
#ax[0].set_ylim(-0.05*100,0.2*100)

ax[1].set_title("LT battery (VRF)")
ax[1].plot(ac,abs_tc_irr[1,:]*100,color="red",label="Consumer tariff")
ax[1].plot(ac,abs_tp_irr[1,:]*100,color="blue",label="Producer tariff")
ax[1].axhline(abs_tp_irr[1,1]*100,label="Current tariffs",color="gray",linestyle="--")
#ax[1].set_ylim(-0.05*100,0.2*100)

ax[2].set_title("EwS system (Alkaline)")
ax[2].plot(ac,abs_tc_irr[2,:]*100,color="red",label="Consumer tariff")




#%% Price impact (FCR)
P3 = P.copy()
P3["FCRN"] = GP23["FCR"]
P3["2024"] = 0
P3["H"] = np.ones(len((P)))*2
A3 = Act.copy()
A3["FCRN_U"] = fcr_u
A3["FCRN_D"] = fcr_d

dP = [0.75,0.9,1,1.1,1.25]

markets = ['FCRN', 'aFRR_U', 'aFRR_D', 'mFRR_U']
pI_st = np.zeros((len(markets),len(dP)))

for j in range(len(markets)):
    P4 = P3.copy()
    for i in range(len(dP)):
        P4 = P3.copy()
        P4[markets[j]] = P3[markets[j]]*dP[i]
        
        bs_fcr = FCR_st_Battery(OL)
        bs_fcr.excl_markets(["FCRD"])
        bs_fcr.set_params(P_st,E_st,E_min_st,st_eff_u,st_eff_d,st_l_eff,0.5*E_st,st_cl,st_cpx,tc,tp)
        st_res_fcr,st_e0_fcr = bs_fcr.run_yr(P4,A3,start_d="2023-01-01",end_d="2023-12-31 12")
        st_soe_fcr,st_rec_c_fcr,st_rec_d_fcr,st_rev_fcr,st_cyc_fcr = yr_rev_cyc_ST(bs_fcr, st_res_fcr, A3, P4,tc,tp)
        st_npv_fcr,st_irr_fcr,st_c_fcr = dcf(st_rev_fcr,10,0.08,0.22,st_opx,st_cpx)
        pI_st[j,i] = st_irr_fcr
    


#LT
P3 = P.copy()
P3["FCRN"] = GP23["FCR"]
P3["2024"] = 0
P3["H"] = np.ones(len((P)))*2
A3 = Act.copy()
A3["FCRN_U"] = fcr_u
A3["FCRN_D"] = fcr_d

dP = np.array([0.75,0.9,1,1.1,1.25])

markets = ['FCRN', 'aFRR_U', 'aFRR_D', 'mFRR_U']
pI_lt = np.zeros((len(markets),len(dP)))

for j in range(len(markets)):
    P4 = P3.copy()
    for i in range(len(dP)):
        P4 = P3.copy()
        P4[markets[j]] = P3[markets[j]]*dP[i]
        
        bl_fcr = FCR_LT_Battery(OL)
        bl_fcr.excl_markets(["FCRD"])
        bl_fcr.set_params(P_lt,E_lt,E_min_lt,lt_eff_u,lt_eff_d,lt_l_eff,0.5*E_lt,lt_cl,lt_cpx,tc,tp)
        bl_fcr.set_k(K)
        lt_res_fcr,lt_e0_fcr = bl_fcr.run_yr(P4,A3,start_d="2023-01-01",end_d="2023-12-31 12")
        lt_soe_fcr,lt_rec_c_fcr,lt_rec_d_fcr,lt_rev_fcr,lt_cyc_fcr = yr_rev_cyc_LT(bl_fcr, lt_res_fcr, A3, P4,tc,tp)
        lt_npv_fcr,lt_irr_fcr,lt_c_fcr = dcf(lt_rev_fcr,15,0.08,0.22,lt_opx,lt_cpx)
        pI_lt[j,i] = lt_irr_fcr


plt.plot(dP*100-100,pI_lt[0,:]*100,label="FCR Price")
plt.plot(dP*100-100,pI_lt[1,:]*100,label="aFRR Upwards Price")
plt.plot(dP*100-100,pI_lt[2,:]*100,label="aFRR Downwards Price")
plt.plot(dP*100-100,pI_lt[3,:]*100,label="mFRR Upwards Price")
plt.xlabel("Change in price [%]")
plt.ylabel("IRR [%]")
plt.legend()






plt.plot(dP*100-100,pI_st[0,:]*100,label="FCR Price")
plt.plot(dP*100-100,pI_st[1,:]*100,label="aFRR Upwards Price")
plt.plot(dP*100-100,pI_st[2,:]*100,label="aFRR Downwards Price")
plt.plot(dP*100-100,pI_st[3,:]*100,label="mFRR Upwards Price")
plt.scatter(dP[2]*100-100,pI_st[3,2]*100,color="black",label="LI battery")
plt.xlabel("Change in price [%]")
plt.ylabel("IRR [%]")

plt.scatter(dP[2]*100-100,pI_lt[3,2]*100,color="tab:gray",label="VRF battery")
plt.legend(bbox_to_anchor=(1,1,0,0))


plt.plot(dP*100-100,pI_lt[0,:]*100,color="tab:blue",label="FCR Price")
plt.plot(dP*100-100,pI_lt[1,:]*100,color="tab:orange",label="aFRR Upwards Price")
plt.plot(dP*100-100,pI_lt[2,:]*100,color="tab:green",label="aFRR Downwards Price")
plt.plot(dP*100-100,pI_lt[3,:]*100,color="tab:red",label="mFRR Upwards Price")

plt.xlabel("Change in price [%]")
plt.ylabel("IRR [%]")
#plt.legend()







""" #NOT RELEVANT AS THE IRR IS NAN FOR ALL CASES
#h
P3 = P.copy()
P3["FCRN"] = GP23["FCR"]
P3["2024"] = 0
P3["H"] = np.ones(len((P)))*2
A3 = Act.copy()
A3["FCRN_U"] = fcr_u
A3["FCRN_D"] = fcr_d



dPh = [1.1,1.25,1.5,2]

markets = ['FCRN', 'aFRR_U', 'aFRR_D', 'mFRR_U',"H"]
pI_h = np.zeros((len(markets),len(dPh)))

for j in range(len(markets)):
    P4 = P3.copy()
    for i in range(len(dPh)):
        P4 = P3.copy()
        P4[markets[j]] = P3[markets[j]]*dPh[i]
        
        #hydrogen
        bh_fcr = FCR_H_storage_L(OL)
        bh_fcr.excl_markets(["FCRD"])
        bh_fcr.set_params(P_h, E_h, E_h/2, h_DMD, 0, h_cpx, tc)
        h_res_fcr,h_e0_fcr = bh_fcr.run_yr(P4,A3,start_d="2023-01-01",end_d="2023-12-31 12")
        h_soe_fcr,h_rev_fcr,h_dH_fcr = yr_rev_cyc_H(bh_fcr,h_res_fcr,A3,P4,tc)
        h_npv_fcr,h_irr_fcr,h_c_fcr = dcf(h_rev_fcr,25,0.08,0.22,h_opx,h_cpx)
        pI_h[j,i] = h_irr_fcr
        
        
plt.plot(dPh,pI_h[0,:])
plt.plot(dPh,pI_h[1,:])
plt.plot(dPh,pI_h[2,:])
plt.plot(dPh,pI_h[3,:])
plt.plot(dPh,pI_h[3,:])
"""
#%% 
