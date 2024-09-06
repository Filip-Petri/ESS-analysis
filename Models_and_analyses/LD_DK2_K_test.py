from Load_data import *
#%% 

from LT_class_func import *

import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import os

import pyscipopt


#%% single run
OL = 8*24
f_disc = 0.08
f_tax = 0.22
bm = LT_Battery(OL)
bm.excl_markets(["FCRD"])

P2 = P.copy()
#P2["FCRN"] = GP23["FCR"]
P2["2024"]=0


E= 25
C = 1/5
eff = 0.825
eff_l = 0.998
bm.set_params(E*C,E,0,eff,1/eff,eff_l,0.5*E,0,0,147.3,12.1)
bm.set_k(C)
res,e0 = bm.run_yr(P2,Act,start_d="2023-01-01",end_d="2023-12-31 12")

tt,ttrc,ttrd,rr,cc = yr_rev_cyc_LT(bm, res, Act, P2,147.3,12.1)



fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(np.concatenate( ([E*0.5],res["e_b"]))[0:24*7],label="Modelled SoE",color="tab:blue")
ax[0].plot(np.concatenate( ([E*0.5],tt))[0:24*7],label="Realized SoE",color="tab:orange")
ax[0].plot(np.concatenate( ([E*0.5],res["e_b"]))[0:24*7]+E*C,label="Recovery threshold",color="tab:red",linestyle="--")
ax[0].plot(np.concatenate( ([E*0.5],res["e_b"]))[0:24*7]-E*C,color="tab:red",linestyle="--")
ax[0].set_ylabel("SoE [MWh]")
ax[1].set_xlabel("Time [h]")
ax[1].bar(range(24*7),ttrc[0:24*7].values,color="tab:green",edgecolor="green",label="Recovery charged ")
ax[1].bar(range(24*7),ttrd[0:24*7].values,color="tab:red",edgecolor="red",label="Recovery discharged")
ax[1].set_ylabel("Recovered Energy [MWh]")
ax[0].legend(bbox_to_anchor=(1,1,0,0))
ax[1].legend(bbox_to_anchor=(1,1,0.45,0))
plt.show()

plt.figure()
plt.plot(np.concatenate( ([E*0.5],res["e_b"]))[0:24*7],label="Modelled SoE",color="tab:blue")
plt.plot(np.concatenate( ([E*0.5],tt))[0:24*7],label="Realized SoE",color="tab:orange")
plt.plot(np.concatenate( ([E*0.5],res["e_b"]))[0:24*7]+E*C,label="Recovery threshold",color="tab:red",linestyle="--")
plt.plot(np.concatenate( ([E*0.5],res["e_b"]))[0:24*7]-E*C,color="tab:red",linestyle="--")
plt.legend(bbox_to_anchor=(1,1,0,0))
plt.xlabel("Time [h]")
plt.ylabel("State of Energy [MWh]")
plt.show()


plt.plot(range(24*7),ttrc[0:24*7].values,color="green")
plt.plot(range(24*7),ttrd[0:24*7].values,color="red")

npv,irr,c = dcf(rr,30,0.08,0.22,0,-411*E*1000-416*E*C*1000)


#%% 5hr Test different k-values (reserved power)



OL = 8*24
f_disc = 0.08
f_tax = 0.22
bm = LT_Battery(OL)
bm.excl_markets(["FCRD"])
lt_cpx = -(410.88*E_lt + 414.69*P_lt)*1000
lt_cl = 10000

P2 = P.copy()
#P2["FCRN"] = GP23["FCR"]
P2["2024"]=0


j = 0

ks = [0,0.01,0.02,0.05,0.075,0.1,0.2,0.3,0.4,0.5,1]#[0,1/5*C,2/5*C,3/5*C,4/5*C,C,2*C,3*C,4*C,5*C]
e5 = np.zeros(len(ks))
rt5 = np.zeros(len(ks))


for k in ks:
        
    E= 25
    C = 1/5
    eff = 0.825
    eff_l = 0.998
    bm.set_params(E*C,E,E*0,eff,1/eff,eff_l,0.5*E,lt_cl,lt_cpx,147.3,12.1)
    bm.set_k(k)
    res,e0 = bm.run_yr(P2,Act,start_d="2023-01-01",end_d="2023-12-31 12")
    
    tt,ttrc,ttrd,rr,cc = yr_rev_cyc_LT(bm, res, Act, P2,147.3,12.1)
    e5[j] = 100*np.sum((Act["aFRR_U"]*res["p_aFRR_U"] + Act["mFRR_U"]*res["p_mFRR_U"] + Act["FCRN_U"]*res["p_FCRN"] + Act["FCRD_U"]*res["p_FCRD_U"] + (Act["aFRR_D"]*res["p_aFRR_D"] +  Act["FCRN_D"]*res["p_FCRN"] + Act["FCRD_D"]*res["p_FCRD_D"]))[tt>0][tt<25] ) /np.sum(Act["aFRR_U"]*res["p_aFRR_U"] + Act["mFRR_U"]*res["p_mFRR_U"] + Act["FCRN_U"]*res["p_FCRN"] + Act["FCRD_U"]*res["p_FCRD_U"] + (Act["aFRR_D"]*res["p_aFRR_D"] +  Act["FCRN_D"]*res["p_FCRN"] + Act["FCRD_D"]*res["p_FCRD_D"]))
    #sum(tt<0) + sum(tt>25)
    rt5[j] = sum(np.sum(rr))
    j+=1


#%% 10 hr


OL = 8*24
f_disc = 0.08
f_tax = 0.22
bm = LT_Battery(OL)
bm.excl_markets(["FCRD"])

P2 = P.copy()
#P2["FCRN"] = GP23["FCR"]
P2["2024"]=0


j = 0

ks = [0,0.01,0.02,0.05,0.075,0.1,0.2,0.3,0.4,0.5,1]#[0,1/5*C,2/5*C,3/5*C,4/5*C,C,10/5*C,15/5*C,20/5*C,25/5*C]
e10 = np.zeros(len(ks))
rt10 = np.zeros(len(ks))


for k in ks:
    E= 25
    C = 1/10
    eff = 0.825
    eff_l = 0.998
    bm.set_params(E*C,E,E*0,eff,1/eff,eff_l,0.5*E,lt_cl,lt_cpx,147.3,12.1)
    bm.set_k(k)
    res,e0 = bm.run_yr(P2,Act,start_d="2023-01-01",end_d="2023-12-31 12")
    
    tt,ttrc,ttrd,rr,cc = yr_rev_cyc_LT(bm, res, Act, P2,147.3,12.1)
    e10[j] = 100*np.sum((Act["aFRR_U"]*res["p_aFRR_U"] + Act["mFRR_U"]*res["p_mFRR_U"] + Act["FCRN_U"]*res["p_FCRN"] + Act["FCRD_U"]*res["p_FCRD_U"] + (Act["aFRR_D"]*res["p_aFRR_D"] +  Act["FCRN_D"]*res["p_FCRN"] + Act["FCRD_D"]*res["p_FCRD_D"]))[tt>0][tt<25] ) /np.sum(Act["aFRR_U"]*res["p_aFRR_U"] + Act["mFRR_U"]*res["p_mFRR_U"] + Act["FCRN_U"]*res["p_FCRN"] + Act["FCRD_U"]*res["p_FCRD_U"] + (Act["aFRR_D"]*res["p_aFRR_D"] +  Act["FCRN_D"]*res["p_FCRN"] + Act["FCRD_D"]*res["p_FCRD_D"]))
    #sum(tt<0) + sum(tt>25)
    rt10[j] = sum(np.sum(rr))
    print(j)
    print(e10[j])
    j+=1

#%% 8hr

OL = 8*24
f_disc = 0.08
f_tax = 0.22
bm = LT_Battery(OL)
bm.excl_markets(["FCRD"])

P2 = P.copy()
#P2["FCRN"] = GP23["FCR"]
P2["2024"]=0


j = 0

ks = [0,0.01,0.02,0.05,0.075,0.1,0.2,0.3,0.4,0.5,1]#[0,1/5*C,2/5*C,3/5*C,4/5*C,C,10/5*C,15/5*C,20/5*C,25/5*C]
e8 = np.zeros(len(ks))
rt8 = np.zeros(len(ks))


for k in ks:
    E= 25
    C = 1/8
    eff = 0.825
    eff_l = 0.998
    bm.set_params(E*C,E,E*0,eff,1/eff,eff_l,0.5*E,lt_cl,lt_cpx,147.3,12.1)
    bm.set_k(k)
    res,e0 = bm.run_yr(P2,Act,start_d="2023-01-01",end_d="2023-12-31 12")
    
    tt,ttrc,ttrd,rr,cc = yr_rev_cyc_LT(bm, res, Act, P2,147.3,12.1)
    e8[j] = 100*np.sum((Act["aFRR_U"]*res["p_aFRR_U"] + Act["mFRR_U"]*res["p_mFRR_U"] + Act["FCRN_U"]*res["p_FCRN"] + Act["FCRD_U"]*res["p_FCRD_U"] + (Act["aFRR_D"]*res["p_aFRR_D"] +  Act["FCRN_D"]*res["p_FCRN"] + Act["FCRD_D"]*res["p_FCRD_D"]))[tt>0][tt<25] ) /np.sum(Act["aFRR_U"]*res["p_aFRR_U"] + Act["mFRR_U"]*res["p_mFRR_U"] + Act["FCRN_U"]*res["p_FCRN"] + Act["FCRD_U"]*res["p_FCRD_U"] + (Act["aFRR_D"]*res["p_aFRR_D"] +  Act["FCRN_D"]*res["p_FCRN"] + Act["FCRD_D"]*res["p_FCRD_D"]))
    #sum(tt<0) + sum(tt>25)
    rt8[j] = sum(np.sum(rr))
    print(j)
    print(e8[j])
    j+=1





plt.figure()
plt.plot(ks,e5,label="5hr battery",color="tab:blue")
plt.axvline(0.2,linestyle=":",color="tab:blue")

plt.plot(ks,e8,label="8hr battery",color="tab:green")
plt.axvline(1/8,linestyle=":",color="tab:green")

plt.plot(ks,e10,label="10hr battery",color="tab:orange")
plt.axvline(0.1,linestyle=":",color="tab:orange")

plt.axvline(-3,linestyle=":",color="tab:gray",label="K scaled to C-rates")
plt.xlim(0,1)
plt.xlabel("Weighing of capacity reservation (value of K)")
plt.ylabel("Confidence of delivery [%]")
plt.legend()



plt.figure()
plt.plot(ks,rt5,label="5hr battery",color="tab:blue")
plt.plot(ks,rt8,label="8hr battery",color="tab:green")
plt.plot(ks,rt10,label="10hr battery",color="tab:orange")


plt.axhline(rt5[6],linestyle=":",color="tab:blue")
plt.axhline((rt8[5]*0.75+rt8[6]*0.25)/1,linestyle=":",color="tab:green")
plt.axhline(rt10[5],linestyle=":",color="tab:orange")

#plt.axvline(0.2,linestyle=":",color="tab:blue")
#plt.axvline(1/8,linestyle=":",color="tab:green")
#plt.axvline(0.1,linestyle=":",color="tab:orange")
plt.axvline(-3,linestyle=":",color="tab:gray",label="Revenue at K scaled to C-rates")
plt.xlim(0,1)

plt.xlabel("Weighing of capacity reservation (value of K)")
plt.legend()
plt.ylabel("Yearly revenue [€]")



plt.figure()
fig, ax = plt.subplots(2, sharex=True)
fig.tight_layout()

ax[0].plot(ks,e5,label="5hr battery",color="tab:blue")
ax[0].axvline(0.2,linestyle=":",color="tab:blue")

ax[0].plot(ks,e8,label="8hr battery",color="tab:green")
ax[0].axvline(1/8,linestyle=":",color="tab:green")

ax[0].plot(ks,e10,label="10hr battery",color="tab:orange")
ax[0].axvline(0.1,linestyle=":",color="tab:orange")

ax[0].axvline(-3,linestyle=":",color="tab:gray",label="K scaled to C-rates")
ax[0].set_xlim(0,1)
ax[0].xlabel("Weighing of capacity reservation (value of K)")
ax[0].ylabel("Confidence of delivery [%]")

ax[1].plot(ks,rt5,label="5hr battery",color="tab:blue")
plt.plot(ks,rt8,label="8hr battery",color="tab:green")
plt.plot(ks,rt10,label="10hr battery",color="tab:orange")


plt.axhline(rt5[6],linestyle=":",color="tab:blue")
plt.axhline((rt8[5]*0.75+rt8[6]*0.25)/1,linestyle=":",color="tab:green")
plt.axhline(rt10[5],linestyle=":",color="tab:orange")

#plt.axvline(0.2,linestyle=":",color="tab:blue")
#plt.axvline(1/8,linestyle=":",color="tab:green")
#plt.axvline(0.1,linestyle=":",color="tab:orange")
plt.axvline(-3,linestyle=":",color="tab:gray",label="Revenue at K scaled to C-rates")
plt.xlim(0,1)

plt.xlabel("Weighing of capacity reservation (value of K)")
plt.legend()
plt.ylabel("Yearly revenue [€]")


#[0.2 * 0.1 , 0.4 * 0.1, 0.6*0.1 , 0.8*0.1 , 1*0.1,0.2,0.3]




#(1-C)

#np.linspace(C,1,3)
#np.linspace(0.2,1,3)

#npv,irr,c = dcf(rr,30,0.08,0.22,0,-411*E*1000-416*E*C*1000)








#%% tests
#np.transpose(pd.DataFrame(np.sum(rr),columns=["lt"])).plot(kind="bar",stacked="True",width=1,edgecolor="white")

v = -(Act["aFRR_U"]*res["p_aFRR_U"] + Act["mFRR_U"]*res["p_mFRR_U"] + Act["FCRN_U"]*res["p_FCRN"] + Act["FCRD_U"]*res["p_FCRD_U"])*(1/0.825) + (Act["aFRR_D"]*res["p_aFRR_D"] +  Act["FCRN_D"]*res["p_FCRN"] + Act["FCRD_D"]*res["p_FCRD_D"])*0.825
plt.plot(tt[0:24*14])
plt.plot( (res["e_b"])[0:24*14] )

plt.plot(tt)
plt.plot( (res["e_b"]))

plt.plot(res["e_b"][6060:6100])

plt.plot(tt[6060:6100])
plt.plot(ttrc[6060:6100])
plt.plot(ttrd[6060:6100])
plt.plot(v[6060:6100],color="tab:grey")

plt.plot(res["e_d"][6060:6100])
plt.plot(res["e_c"][6060:6100])


v[6060:6100]


[np.where(tt<-10)[0]]

# E_mt = 25

# C_mt = 1/5
# mt_rs, mt_rv, mt_irr , mt_cyc = MT_run(OL =15*24, Esize = E_mt, Crate = C_mt, eff_u=0.825, eff_d=0.825, eff_lin=0.998, DoD=0, T=147.3,TP=12.1, price = P, activ = Act, cpx = np.multiply(-410.88,E_mt*10**3) - 414.68*10**3*E_mt*C_mt, opx=-12364*E_mt*C_mt,lft=20, excM=["FCRD"],fcr=False)


netA = (Act["aFRR_U"]*res["p_aFRR_U"] + Act["mFRR_U"]*res["p_mFRR_U"] + Act["FCRN_U"]*res["p_FCRN"] + Act["FCRD_U"]*res["p_FCRD_U"])/(1/eff) - (Act["aFRR_D"]*res["p_aFRR_D"] +  Act["FCRN_D"]*res["p_FCRN"] + Act["FCRD_D"]*res["p_FCRD_D"])*eff

#noR_soe = res["e_b"]+np.cumsum(netA)
#plt.plot(noR_soe[0:24*14])

R_SOE = res["e_b"].copy()*0
rec_d = R_SOE.copy()*0
rec_c = R_SOE.copy()*0

R_SOE[0] = res["e_b"][0] + netA[0]

max_d = - ((E_mt*C_mt)-(res["p_aFRR_U"]+res["p_mFRR_U"]+res["p_FCRN"]+res["p_FCRD_U"]+res["e_d"]))
max_c = (E_mt*C_mt)-(res["p_aFRR_D"]+res["p_FCRN"]+res["p_FCRD_D"]+res["e_c"])

for i in range(1,8759):
    R_SOE[i] = R_SOE[i-1] + netA[i] + res["e_c"][i]*eff - res["e_d"][i]/(1/eff) + rec_d[i]/eff + rec_c[i]*eff
    if R_SOE[i]/E_mt > 1-C:
        rec_d[i+1] = min( ( E_mt*(1-C) - R_SOE[i])*eff,max_d[i+1])
    if R_SOE[i]/E_mt < 0+C:
        rec_c[i+1] = min ( (E_mt*(0+C) - R_SOE[i])/eff,max_c[i+1])
    #R_SOE[i] += rec[i]
    







