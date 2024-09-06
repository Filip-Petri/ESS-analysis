from Load_data import *
#%% 

import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import os

import pyscipopt



class st_Battery:
    def __init__(self, OptLen):
        self.OptLen = OptLen
        self.P_max,self.E_max,self.E_min,self.Eta_up,self.Eta_down,self.Eta_lin,self.E_start = (cp.Parameter(),cp.Parameter(),cp.Parameter(),cp.Parameter(),cp.Parameter(),cp.Parameter(),cp.Parameter())
        #self.Eta_ud = cp.Parameter() #temp. implementation - only for dpp speed (runtime in sensitivity)
        #self.Eta_du = cp.Parameter() #temp. implementation - only for dpp speed (runtime in sensitivity)
        
        self.L = {"DA": cp.Parameter(self.OptLen),
                  "reg_U":cp.Parameter(self.OptLen),
                  "reg_D":cp.Parameter(self.OptLen),
                  "FCRN": cp.Parameter(self.OptLen),
                  "FCRD_U": cp.Parameter(self.OptLen),
                  "FCRD_D": cp.Parameter(self.OptLen),
                  "aFRR_U": cp.Parameter(self.OptLen),
                  "aFRR_D": cp.Parameter(self.OptLen),
                  "mFRR_U": cp.Parameter(self.OptLen)}
        
        self.p = {"DA": cp.Variable(self.OptLen),
                  "FCRN": cp.Variable(self.OptLen,nonneg=True),
                  "FCRD_U": cp.Variable(self.OptLen,nonneg=True),
                  "FCRD_D": cp.Variable(self.OptLen,nonneg=True),
                  "aFRR_U": cp.Variable(self.OptLen,nonneg=True),
                  "aFRR_D": cp.Variable(self.OptLen,nonneg=True),
                  "mFRR_U": cp.Variable(self.OptLen,nonneg=True)}
        self.e = {"b":cp.Variable(self.OptLen,nonneg=True),
                  "c":cp.Variable(self.OptLen,nonneg=True),
                  "d":cp.Variable(self.OptLen,nonneg=True)}
        
        
        self.E_act = {"mFRR_U":cp.Parameter(),
                    "aFRR_U":cp.Parameter(),    
                    "aFRR_D":cp.Parameter(),
                    "FCRN_D":cp.Parameter(),
                    "FCRN_U":cp.Parameter(),
                    "FCRD_D":cp.Parameter(),
                    "FCRD_U":cp.Parameter()}
        self.c_deg = cp.Variable(self.OptLen,nonneg=True)
        self.r_act = cp.Variable(self.OptLen)
        self.r_rec = cp.Variable(self.OptLen-1)
        
        self.A_U = cp.Variable(self.OptLen,nonneg=True)
        self.A_D = cp.Variable(self.OptLen,nonneg=True)
        #self.A_net_U = cp.Variable(self.OptLen,nonneg=True)
        #self.A_net_D = cp.Variable(self.OptLen,nonneg=True)
        self.capex = cp.Parameter()
        self.CycleLife = cp.Parameter()
        
        self.E_tariff = cp.Parameter()
        self.EP_tariff = cp.Parameter()
        
        # self.y = {"1":cp.Variable(self.OptLen,boolean=True),
        #           "2":cp.Variable(self.OptLen,boolean=True),
        #           "3":cp.Variable(self.OptLen,boolean=True),
        #           "4":cp.Variable(self.OptLen,boolean=True)}
        #self.M = 100
        self.obj = cp.Maximize(0)
                               #cycle cost
                               #+(cp.sum(self.e["d"] + self.e["c"] + 2*(self.E_act["mFRR_U"]*self.p["mFRR_U"] + self.E_act["aFRR_U"]*self.p["aFRR_U"] + self.E_act["aFRR_D"]*self.p["aFRR_D"] + self.E_act["FCRN_D"]*self.p["FCRN"] + self.E_act["FCRN_U"]*self.p["FCRN"] + self.E_act["FCRD_U"]*self.p["FCRD_U"] + self.E_act["FCRD_D"]*self.p["FCRD_D"]))*(CAPEX/8000)/(2*self.E_max)) )
                               #Activation cost
                               
                               

        #cp.sum(2*(self.E_act["mFRR_U"][:-1]*self.p["mFRR_U"][:-1]*self.L["reg_U"][:-1] + self.E_act["aFRR_U"][:-1]*self.p["aFRR_U"][:-1] + self.E_act["aFRR_D"][:-1]*self.p["aFRR_D"][:-1] + self.E_act["FCRN_D"][:-1]*self.p["FCRN"][:-1] + self.E_act["FCRN_U"][:-1]*self.p["FCRN"][:-1] + self.E_act["FCRD_U"][:-1]*self.p["FCRD_U"][:-1] + self.E_act["FCRD_D"][:-1]*self.p["FCRD_D"][:-1]))
                               
                               
                               
                               
        #+ cp.multiply(self.A_mFRR_U,self.p["mFRR_U"]) @ self.L_reg_U + cp.multiply(self.A_aFRR_U,self.p["aFRR_U"]) @ self.L_reg_U - cp.multiply(self.A_aFRR_D,self.p["aFRR_D"]) @ self.L_reg_D )#- sum((self.e["d"]+self.e["c"]+cp.multiply(self.A_aFRR_D,self.p["aFRR_D"])*self.Eta_up + cp.multiply(self.A_aFRR_U,self.p["aFRR_U"])*self.Eta_down + cp.multiply(self.A_mFRR_U,self.p["mFRR_U"])*self.Eta_down )/self.E_max/8000 *7.8*10**6 /2) )#+ 1*p_mFRR_U @ AP_mFRR_U + 1*p_aFRR_U @ AP_aFRR_U - 1*p_aFRR_D @ AP_aFRR_D)
        self.constraints = [self.c_deg == -( (self.e["c"] + self.A_D)*self.Eta_up + (self.e["d"] + self.A_U)*self.Eta_down)/(2*self.E_max) * (self.capex/self.CycleLife),#+ self.A_net_U + self.A_net_D)
                            #self.c_deg[0] == -(self.e["c"][0]*self.Eta_up + self.e["d"][0]*self.Eta_down + self.E_start*self.Eta_lin + self.A_U[0]*self.Eta_down + self.A_D[0]*self.Eta_up + self.A_net_U[0] + self.A_net_D[0])/(2*self.E_max) * self.capex/self.CycleLife,
                            #self.c_deg[1:] == -(self.e["c"][1:]*self.Eta_up + self.e["d"][1:]*self.Eta_down + self.e["b"][:-1]*self.Eta_lin + self.A_U[1:]*self.Eta_down + self.A_D[1:]*self.Eta_up + self.A_net_U[1:] + self.A_net_D[1:])/(2*self.E_max) * self.capex/self.CycleLife,
                            self.A_U == (self.E_act["mFRR_U"]*self.p["mFRR_U"] + self.E_act["aFRR_U"]*self.p["aFRR_U"] + self.E_act["FCRN_U"]*self.p["FCRN"] + self.E_act["FCRD_U"]*self.p["FCRD_U"]),
                            self.A_D == (self.E_act["aFRR_D"]*self.p["aFRR_D"] + self.E_act["FCRN_D"]*self.p["FCRN"] + self.E_act["FCRD_D"]*self.p["FCRD_D"]),
                            #self.A_net_U - self.A_net_D == self.A_U*self.Eta_down - self.A_D*self.Eta_up,
                            self.r_act == cp.multiply(self.A_U,self.L["reg_U"]-self.EP_tariff) - cp.multiply(self.A_D,self.L["reg_D"]+self.E_tariff),
                            self.r_rec == cp.multiply(self.A_D[:-1] * (self.Eta_up/self.Eta_down) - self.A_U[:-1] * (self.Eta_down/self.Eta_up), self.L["DA"][1:]) - self.A_U[:-1]*(self.Eta_down/self.Eta_up)*self.EP_tariff - self.A_D[:-1]*(self.Eta_up/self.Eta_down)*self.E_tariff,
                            
                            
                            #self.A_net_U <= self.A_U*self.Eta_down,
                            #self.A_net_D <= self.A_D*self.Eta_up,
                            #self.c_act == 0,
                            #self.c_act[:-1] == cp.multiply(self.A_D[:-1],self.L["reg_D"][:-1]) - cp.multiply(self.A_U[:-1],self.L["reg_U"][:-1]) + cp.multiply(self.A_net_U[:-1],self.L["reg_D"][1:])/self.Eta_up - cp.multiply(self.A_net_D[:-1],self.L["reg_U"][1:])/self.Eta_down,
                            #self.c_act[:-1] == cp.multiply(self.A_D[:-1],self.L["reg_D"][:-1] + self.E_tariff) - cp.multiply(self.A_U[:-1],self.L["reg_U"][:-1] - self.EP_tariff) + cp.multiply(self.A_net_U[:-1],self.L["DA"][1:]+self.E_tariff)/self.Eta_up - cp.multiply(self.A_net_D[:-1],self.L["DA"][1:] - self.EP_tariff)/self.Eta_down,
                           
                            #self.c_act[:-1] == cp.multiply(self.A_D[:-1],self.L["DA"][:-1]) - cp.multiply(self.A_U[:-1],self.L["DA"][:-1]) + cp.multiply(self.A_net_U[:-1],self.L["DA"][1:]) - cp.multiply(self.A_net_D[:-1],self.L["DA"][1:]),
                            #self.c_act[-1] == 0,

                            # self.A_net_U <= 50,
                            # self.A_net_D <= 50,
                            #Energy balance
                            self.e["b"][0]  == self.E_start*self.Eta_lin + self.e["c"][0]*self.Eta_up - self.e["d"][0]*self.Eta_down,#(b_d[0] + cp.multiply(p_mF_U[0],A_mFRR_U[0]))*Eta_down ,                 #Battery energy balance after first hour
                            self.e["b"][1:] == self.e["b"][0:-1]*self.Eta_lin + self.e["c"][1:]*self.Eta_up - self.e["d"][1:]*self.Eta_down,#(b_d[1:]+cp.multiply(p_mF_U[1:],A_mFRR_U[1:]))*Eta_down,    #Battery energy balance
                            self.e["b"][-1] == 0.5*self.E_max,#self.E_start,#,#self.E_start,#0.5*self.E_max,                                                  #SOC is restored to start value
                            #self.e["b"][23::24] == 0.5*self.E_max,
                            self.e["b"]     <= self.E_max,                                                        #Cannot hold more than max SOC
                            self.e["b"]     >= self.E_min,                                                  #Battery should never be under min soc
                            self.p["FCRN"][-1] + self.p["FCRD_U"][-1] + self.p["FCRD_D"][-1] + self.p["aFRR_U"][-1] + self.p["aFRR_D"][-1] + self.p["mFRR_U"][-1] == 0, 
                            
                            #Power capacity constraints (incl. LER)
                            self.e["c"][0] + 1.34*self.p["FCRN"][0] + self.p["FCRD_D"][0] + 0.2*self.p["FCRD_U"][0] + self.p["aFRR_D"][0] <= self.P_max*0.8,                                    #charging cannot exceed power cap. Some cap is reserved for FCR.
                            self.e["d"][0] + 1.34*self.p["FCRN"][0] + self.p["FCRD_U"][0] + 0.2*self.p["FCRD_D"][0] + self.p["aFRR_U"][0] + self.p["mFRR_U"][0] <= self.P_max*0.8,                           #discharging cannot exceed power cap. Some cap is reserved for FCR.
                            
                            (self.e["c"][1:] + 1.34*self.p["FCRN"][1:] + self.p["FCRD_D"][1:] + 0.2*self.p["FCRD_U"][1:] + self.p["aFRR_D"][1:]) + (1/3*self.p["FCRD_U"][:-1] + self.p["FCRN"][:-1] + self.p["aFRR_U"][:-1] + self.p["mFRR_U"][:-1])*self.Eta_down/self.Eta_up <= self.P_max*0.8,
                            (self.e["d"][1:] + 1.34*self.p["FCRN"][1:] + self.p["FCRD_U"][1:] + 0.2*self.p["FCRD_D"][1:] + self.p["aFRR_U"][1:] + self.p["mFRR_U"][1:]) + (1/3*self.p["FCRD_D"][:-1] + self.p["FCRN"][:-1] + self.p["aFRR_D"][:-1])*self.Eta_up/self.Eta_down <= self.P_max*0.8,
                            
                            #Energy capacity constraints
                            (self.p["FCRN"][0] + 1/3*self.p["FCRD_D"][0] + self.p["aFRR_D"][0] + self.e["c"][0])*self.Eta_up - self.e["d"][0]*self.Eta_down 
                            + (self.p["FCRN"][1] + 1/3*self.p["FCRD_D"][1] + self.p["aFRR_D"][1] + self.e["c"][1])*self.Eta_up -  self.e["d"][1]*self.Eta_down
                            <= self.E_max - self.E_start,
                            
                            (self.p["FCRN"][1:-1] + 1/3*self.p["FCRD_D"][1:-1] + self.p["aFRR_D"][1:-1] +self.e["c"][1:-1])*self.Eta_up - self.e["d"][1:-1]*self.Eta_down 
                            + (self.p["FCRN"][2:] + 1/3*self.p["FCRD_D"][2:] + self.p["aFRR_D"][2:] +self.e["c"][2:])*self.Eta_up - self.e["d"][2:]*self.Eta_down 
                            <= self.E_max - self.e["b"][0:-2],   #battery needs to be able to sustain min 1 hour of downreg and cannot bid full in 2 consecutive hours
                            
                            (self.p["FCRN"][0] + 1/3*self.p["FCRD_U"][0] + self.p["aFRR_U"][0] + self.p["mFRR_U"][0] + self.e["d"][0])*self.Eta_down - self.e["c"][0]*self.Eta_up
                            + (self.p["FCRN"][1] + 1/3*self.p["FCRD_U"][1] + self.p["aFRR_U"][1] + self.p["mFRR_U"][1] + self.e["d"][1])*self.Eta_down - self.e["c"][1]*self.Eta_up
                            <= self.E_start - self.E_min,
                            
                            (self.p["FCRN"][1:-1] + 1/3*self.p["FCRD_U"][1:-1] + self.p["aFRR_U"][1:-1] + self.p["mFRR_U"][1:-1] + self.e["d"][1:-1])*self.Eta_down - self.e["c"][1:-1]*self.Eta_up 
                            + (self.p["FCRN"][2:] + 1/3*self.p["FCRD_U"][2:] + self.p["aFRR_U"][2:] + self.p["mFRR_U"][2:] + self.e["d"][2:])*self.Eta_down - self.e["c"][2:]*self.Eta_up 
                            <= self.e["b"][0:-2] - self.E_min, #battery needs to be able to sustain min 1 hour of upreg and cannot bid full in 2 consecutive hours
                            
                            self.e["d"] - self.e["c"] == self.p["DA"], #+ 1* cp.multiply(self.p["mFRR_U"],self.A_mFRR_U) + 1*cp.multiply(self.p["aFRR_U"],self.A_aFRR_U) - 1*cp.multiply(self.p["aFRR_D"],self.A_aFRR_D),
                            
                            #self.p["FCRD_U"] <= 0,self.p["FCRD_D"] <= 0#,self.p["FCRN"] <= 0,self.p["aFRR_U"] <= 0,self.p["aFRR_D"] <=0,self.p["mFRR_U"]<=0,
                            ]
        self.prob = cp.Problem(self.obj, self.constraints)
        #FCRN: LER units/portfolios must have storage capacity of minimum 1 hour to handle long lasting frequency deviations. file:///C:/Users/filip/Downloads/ancillary-services-to-be-delivered-in-denmark-tender-conditions-1-2-2024%20(1).pdf
    
    def set_params(self,P_max,E_max,E_min,Eta_up,Eta_down,Eta_lin,E_start,cl,cpx,et,ept):
        self.P_max.value =  P_max
        self.E_max.value =  E_max
        self.E_min.value =  E_min
        self.Eta_up.value = Eta_up
        self.Eta_down.value=Eta_down
        self.Eta_lin.value= Eta_lin
        self.E_start.value= E_start
        #self.Eta_ud.value = Eta_up/Eta_down
        #self.Eta_du.value = Eta_down/Eta_up
        self.CycleLife.value = cl
        self.capex.value = cpx
        self.E_tariff.value = et #200 # NOTE check
        self.EP_tariff.value = ept
    def set_prices(self,Price):
        for k in self.L.keys():
            if k != "H":
                self.L[k].value = Price[k].values
    def set_avg_act(self,act):
        mm = np.mean(act)
        self.E_act["mFRR_U"].value = mm["mFRR_U"]
        self.E_act["aFRR_U"].value = mm["aFRR_U"]
        self.E_act["aFRR_D"].value = mm["aFRR_D"]
        self.E_act["FCRN_D"].value = mm["FCRN_D"]
        self.E_act["FCRN_U"].value = mm["FCRN_U"]
        self.E_act["FCRD_D"].value = mm["FCRD_D"]
        self.E_act["FCRD_U"].value = mm["FCRD_U"]
        
        self.obj =  cp.Maximize(#self.p["DA"]
                                self.e["d"] @ (self.L["DA"]-self.EP_tariff) - self.e["c"] @ (self.L["DA"]+self.E_tariff)
                               + self.p["FCRN"] @ self.L["FCRN"] + self.p["FCRD_U"] @ self.L["FCRD_U"] + self.p["FCRD_D"] @ self.L["FCRD_D"] 
                               + self.p["aFRR_U"] @ self.L["aFRR_U"] + self.p["aFRR_D"] @ self.L["aFRR_D"] + self.p["mFRR_U"] @ self.L["mFRR_U"]
                               + cp.sum(self.r_act) + cp.sum(self.r_rec) - cp.sum(self.c_deg)
                               )
        
        self.prob = cp.Problem(self.obj, self.constraints)

    def update_E_start(self,t):
        self.E_start.value= self.e["b"].value[t]

    def excl_markets(self,m):
        if "FCRD" in m:
            self.constraints.append(self.p["FCRD_U"] <= 0)
            self.constraints.append(self.p["FCRD_D"] <= 0)
        if "FCRN" in m:
            self.constraints.append(self.p["FCRN"] <= 0)
        if "aFRR" in m:
            self.constraints.append(self.p["aFRR_D"] <= 0)
            self.constraints.append(self.p["aFRR_U"] <= 0)
        if "mFRR" in m:
            self.constraints.append(self.p["mFRR_U"] <= 0)
        
        self.prob = cp.Problem(self.obj, self.constraints)
    
    def solve(self):
        #print("dpp?: ",self.prob.is_dcp(dpp=True))
        #print("dcp?: ",self.prob.is_dcp(dpp=False))
        self.prob.solve(solver=cp.SCIP,verbose=False)
        #print("ok")
        if self.prob.status != 'optimal':
            print(str(self.prob.status))
        return self.prob.value
    def res(self,hours):
        self.l = hours
        
        return pd.DataFrame({"p_DA":self.p["DA"].value[:self.l],
                "p_FCRN":    self.p["FCRN"].value[:self.l],
                "p_FCRD_U":  self.p["FCRD_U"].value[:self.l],
                "p_FCRD_D":  self.p["FCRD_D"].value[:self.l],
                "p_aFRR_U":  self.p["aFRR_U"].value[:self.l],
                "p_aFRR_D":  self.p["aFRR_D"].value[:self.l],
                "p_mFRR_U":  self.p["mFRR_U"].value[:self.l],
                "e_b":       self.e["b"].value[:self.l],
                "e_c":       self.e["c"].value[:self.l],
                "e_d":       self.e["d"].value[:self.l]})
    def return_params(self):
        return self.P_max.value,self.E_max.value,self.E_min.value,self.Eta_up.value,self.Eta_down.value,self.Eta_lin.value,self.E_start.value
    #def return_prices(self):
    #    return self.L["DA"].value
    
    def run_yr(self,yrP,yrA,start_d,end_d):        
        resLen = self.OptLen-24
        optStart = pd.date_range(start=start_d, end=end_d,freq=str(resLen)+"h")
        self.set_avg_act(yrA)
        yr_res = {"p_DA":np.zeros((len(optStart),resLen)),
           "p_FCRN":    np.zeros((len(optStart),resLen)),
           "p_FCRD_U":  np.zeros((len(optStart),resLen)),
           "p_FCRD_D":  np.zeros((len(optStart),resLen)),
           "p_aFRR_U":  np.zeros((len(optStart),resLen)),
           "p_aFRR_D":  np.zeros((len(optStart),resLen)),
           "p_mFRR_U":  np.zeros((len(optStart),resLen)),
           "e_b":       np.zeros((len(optStart),resLen)),
           "e_c":       np.zeros((len(optStart),resLen)),
           "e_d":       np.zeros((len(optStart),resLen))}
        self.all_E_start = np.zeros(len(optStart))
        
        for i in range(len(optStart)):
            #print(cp.sum(self.c_deg.value))
            #print(optStart[i],)
            date_r = pd.date_range(start=optStart[i],periods=self.OptLen,freq="h")
            
            PP = pd.DataFrame(yrP)[date_r[0]:date_r[-1]]
            self.set_prices(Price=PP)
            self.solve()
            
            # if sum( (np.array(self.A_net_D.value)>0.01) *(np.array(self.A_net_U.value)>0.01) ) > 0:
            #     print(self.A_net_D.value[(np.array(self.A_net_D.value)>0.1) *(np.array(self.A_net_U.value)>0.1)])
            #     print(self.A_net_U.value[(np.array(self.A_net_D.value)>0.1) *(np.array(self.A_net_U.value)>0.1)])
            
            
            Bat_res = self.res(resLen)
            Bat_res.index = date_r[:resLen]
            
            for k in yr_res.keys():
                yr_res[k][i,:] = Bat_res[k]
            
            self.all_E_start[i] = self.return_params()[6]
            #print(self.return_params()[6])
            
            self.update_E_start(resLen-1)
            #print(self.return_params()[6])
            #print()
            
        self.RES = pd.DataFrame({'p_DA':yr_res["p_DA"].flatten(), 
            'p_FCRN':yr_res["p_FCRN"].flatten(), 
            'p_FCRD_U':yr_res["p_FCRD_U"].flatten(), 
            'p_FCRD_D':yr_res["p_FCRD_D"].flatten(), 
            'p_aFRR_U':yr_res["p_aFRR_U"].flatten(), 
            'p_aFRR_D':yr_res["p_aFRR_D"].flatten(), 
            'p_mFRR_U':yr_res["p_mFRR_U"].flatten(), 
            'e_b':yr_res["e_b"].flatten() ,
            'e_c':yr_res["e_c"].flatten(), 
            'e_d':yr_res["e_d"].flatten()})
        self.RES.index = pd.date_range(start=start_d, periods=len(self.RES),freq="h")
        self.RES = self.RES[str(self.RES.index.year[0])]
        
        self.all_E_start = pd.DataFrame(self.all_E_start,index=optStart)[0]

        return self.RES,self.all_E_start

class FCR_st_Battery:
    def __init__(self, OptLen):
        self.OptLen = OptLen
        self.P_max,self.E_max,self.E_min,self.Eta_up,self.Eta_down,self.Eta_lin,self.E_start = (cp.Parameter(),cp.Parameter(),cp.Parameter(),cp.Parameter(),cp.Parameter(),cp.Parameter(),cp.Parameter())
        #self.Eta_ud = cp.Parameter() #temp. implementation - only for dpp speed (runtime in sensitivity)
        #self.Eta_du = cp.Parameter() #temp. implementation - only for dpp speed (runtime in sensitivity)
        
        self.L = {"DA": cp.Parameter(self.OptLen),
                  "reg_U":cp.Parameter(self.OptLen),
                  "reg_D":cp.Parameter(self.OptLen),
                  "FCRN": cp.Parameter(self.OptLen),
                  "FCRD_U": cp.Parameter(self.OptLen),
                  "FCRD_D": cp.Parameter(self.OptLen),
                  "aFRR_U": cp.Parameter(self.OptLen),
                  "aFRR_D": cp.Parameter(self.OptLen),
                  "mFRR_U": cp.Parameter(self.OptLen)}
        
        self.p = {"DA": cp.Variable(self.OptLen),
                  "FCRN": cp.Variable(self.OptLen,nonneg=True),
                  "FCRD_U": cp.Variable(self.OptLen,nonneg=True),
                  "FCRD_D": cp.Variable(self.OptLen,nonneg=True),
                  "aFRR_U": cp.Variable(self.OptLen,nonneg=True),
                  "aFRR_D": cp.Variable(self.OptLen,nonneg=True),
                  "mFRR_U": cp.Variable(self.OptLen,nonneg=True)}
        self.e = {"b":cp.Variable(self.OptLen,nonneg=True),
                  "c":cp.Variable(self.OptLen,nonneg=True),
                  "d":cp.Variable(self.OptLen,nonneg=True)}
        
        
        self.E_act = {"mFRR_U":cp.Parameter(),
                    "aFRR_U":cp.Parameter(),    
                    "aFRR_D":cp.Parameter(),
                    "FCRN_D":cp.Parameter(),
                    "FCRN_U":cp.Parameter(),
                    "FCRD_D":cp.Parameter(),
                    "FCRD_U":cp.Parameter()}
        self.c_deg = cp.Variable(self.OptLen,nonneg=True)
        self.r_act = cp.Variable(self.OptLen)
        self.r_rec = cp.Variable(self.OptLen-1)
        
        self.A_U = cp.Variable(self.OptLen,nonneg=True)
        self.A_D = cp.Variable(self.OptLen,nonneg=True)
        #self.A_net_U = cp.Variable(self.OptLen,nonneg=True)
        #self.A_net_D = cp.Variable(self.OptLen,nonneg=True)
        self.capex = cp.Parameter()
        self.CycleLife = cp.Parameter()
        
        self.E_tariff = cp.Parameter()
        self.EP_tariff = cp.Parameter()
        
        # self.y = {"1":cp.Variable(self.OptLen,boolean=True),
        #           "2":cp.Variable(self.OptLen,boolean=True),
        #           "3":cp.Variable(self.OptLen,boolean=True),
        #           "4":cp.Variable(self.OptLen,boolean=True)}
        #self.M = 100
        self.obj = cp.Maximize(0)
                               #cycle cost
                               #+(cp.sum(self.e["d"] + self.e["c"] + 2*(self.E_act["mFRR_U"]*self.p["mFRR_U"] + self.E_act["aFRR_U"]*self.p["aFRR_U"] + self.E_act["aFRR_D"]*self.p["aFRR_D"] + self.E_act["FCRN_D"]*self.p["FCRN"] + self.E_act["FCRN_U"]*self.p["FCRN"] + self.E_act["FCRD_U"]*self.p["FCRD_U"] + self.E_act["FCRD_D"]*self.p["FCRD_D"]))*(CAPEX/8000)/(2*self.E_max)) )
                               #Activation cost
                               
                               

        #cp.sum(2*(self.E_act["mFRR_U"][:-1]*self.p["mFRR_U"][:-1]*self.L["reg_U"][:-1] + self.E_act["aFRR_U"][:-1]*self.p["aFRR_U"][:-1] + self.E_act["aFRR_D"][:-1]*self.p["aFRR_D"][:-1] + self.E_act["FCRN_D"][:-1]*self.p["FCRN"][:-1] + self.E_act["FCRN_U"][:-1]*self.p["FCRN"][:-1] + self.E_act["FCRD_U"][:-1]*self.p["FCRD_U"][:-1] + self.E_act["FCRD_D"][:-1]*self.p["FCRD_D"][:-1]))
                               
                               
                               
                               
        #+ cp.multiply(self.A_mFRR_U,self.p["mFRR_U"]) @ self.L_reg_U + cp.multiply(self.A_aFRR_U,self.p["aFRR_U"]) @ self.L_reg_U - cp.multiply(self.A_aFRR_D,self.p["aFRR_D"]) @ self.L_reg_D )#- sum((self.e["d"]+self.e["c"]+cp.multiply(self.A_aFRR_D,self.p["aFRR_D"])*self.Eta_up + cp.multiply(self.A_aFRR_U,self.p["aFRR_U"])*self.Eta_down + cp.multiply(self.A_mFRR_U,self.p["mFRR_U"])*self.Eta_down )/self.E_max/8000 *7.8*10**6 /2) )#+ 1*p_mFRR_U @ AP_mFRR_U + 1*p_aFRR_U @ AP_aFRR_U - 1*p_aFRR_D @ AP_aFRR_D)
        self.constraints = [self.c_deg == -( (self.e["c"] + self.A_D)*self.Eta_up + (self.e["d"] + self.A_U)*self.Eta_down)/(2*self.E_max) * (self.capex/self.CycleLife),#+ self.A_net_U + self.A_net_D)
                            #self.c_deg[0] == -(self.e["c"][0]*self.Eta_up + self.e["d"][0]*self.Eta_down + self.E_start*self.Eta_lin + self.A_U[0]*self.Eta_down + self.A_D[0]*self.Eta_up + self.A_net_U[0] + self.A_net_D[0])/(2*self.E_max) * self.capex/self.CycleLife,
                            #self.c_deg[1:] == -(self.e["c"][1:]*self.Eta_up + self.e["d"][1:]*self.Eta_down + self.e["b"][:-1]*self.Eta_lin + self.A_U[1:]*self.Eta_down + self.A_D[1:]*self.Eta_up + self.A_net_U[1:] + self.A_net_D[1:])/(2*self.E_max) * self.capex/self.CycleLife,
                            self.A_U == (self.E_act["mFRR_U"]*self.p["mFRR_U"] + self.E_act["aFRR_U"]*self.p["aFRR_U"] + self.E_act["FCRN_U"]*self.p["FCRN"] + self.E_act["FCRD_U"]*self.p["FCRD_U"]),
                            self.A_D == (self.E_act["aFRR_D"]*self.p["aFRR_D"] + self.E_act["FCRN_D"]*self.p["FCRN"] + self.E_act["FCRD_D"]*self.p["FCRD_D"]),
                            #self.A_net_U - self.A_net_D == self.A_U*self.Eta_down - self.A_D*self.Eta_up,
                            self.r_act == cp.multiply(self.A_U,self.L["reg_U"]-self.EP_tariff) - cp.multiply(self.A_D,self.L["reg_D"]+self.E_tariff),
                            self.r_rec == cp.multiply(self.A_D[:-1] * (self.Eta_up/self.Eta_down) - self.A_U[:-1] * (self.Eta_down/self.Eta_up), self.L["DA"][1:]) - self.A_U[:-1]*(self.Eta_down/self.Eta_up)*self.EP_tariff - self.A_D[:-1]*(self.Eta_up/self.Eta_down)*self.E_tariff,
                            
                            
                            #self.A_net_U <= self.A_U*self.Eta_down,
                            #self.A_net_D <= self.A_D*self.Eta_up,
                            #self.c_act == 0,
                            #self.c_act[:-1] == cp.multiply(self.A_D[:-1],self.L["reg_D"][:-1]) - cp.multiply(self.A_U[:-1],self.L["reg_U"][:-1]) + cp.multiply(self.A_net_U[:-1],self.L["reg_D"][1:])/self.Eta_up - cp.multiply(self.A_net_D[:-1],self.L["reg_U"][1:])/self.Eta_down,
                            #self.c_act[:-1] == cp.multiply(self.A_D[:-1],self.L["reg_D"][:-1] + self.E_tariff) - cp.multiply(self.A_U[:-1],self.L["reg_U"][:-1] - self.EP_tariff) + cp.multiply(self.A_net_U[:-1],self.L["DA"][1:]+self.E_tariff)/self.Eta_up - cp.multiply(self.A_net_D[:-1],self.L["DA"][1:] - self.EP_tariff)/self.Eta_down,
                           
                            #self.c_act[:-1] == cp.multiply(self.A_D[:-1],self.L["DA"][:-1]) - cp.multiply(self.A_U[:-1],self.L["DA"][:-1]) + cp.multiply(self.A_net_U[:-1],self.L["DA"][1:]) - cp.multiply(self.A_net_D[:-1],self.L["DA"][1:]),
                            #self.c_act[-1] == 0,

                            # self.A_net_U <= 50,
                            # self.A_net_D <= 50,
                            #Energy balance
                            self.e["b"][0]  == self.E_start*self.Eta_lin + self.e["c"][0]*self.Eta_up - self.e["d"][0]*self.Eta_down,#(b_d[0] + cp.multiply(p_mF_U[0],A_mFRR_U[0]))*Eta_down ,                 #Battery energy balance after first hour
                            self.e["b"][1:] == self.e["b"][0:-1]*self.Eta_lin + self.e["c"][1:]*self.Eta_up - self.e["d"][1:]*self.Eta_down,#(b_d[1:]+cp.multiply(p_mF_U[1:],A_mFRR_U[1:]))*Eta_down,    #Battery energy balance
                            self.e["b"][-1] == 0.5*self.E_max,#self.E_start,#,#self.E_start,#0.5*self.E_max,                                                  #SOC is restored to start value
                            #self.e["b"][23::24] == 0.5*self.E_max,
                            self.e["b"]     <= self.E_max,                                                        #Cannot hold more than max SOC
                            self.e["b"]     >= self.E_min,                                                  #Battery should never be under min soc
                            self.p["FCRN"][-1] + self.p["FCRD_U"][-1] + self.p["FCRD_D"][-1] + self.p["aFRR_U"][-1] + self.p["aFRR_D"][-1] + self.p["mFRR_U"][-1] == 0, 
                            
                            #Power capacity constraints (incl. LER)
                            self.e["c"][0] + 1.25*self.p["FCRN"][0] + self.p["FCRD_D"][0] + 0.2*self.p["FCRD_U"][0] + self.p["aFRR_D"][0] <= self.P_max*0.8,                                    #charging cannot exceed power cap. Some cap is reserved for FCR.
                            self.e["d"][0] + 1.25*self.p["FCRN"][0] + self.p["FCRD_U"][0] + 0.2*self.p["FCRD_D"][0] + self.p["aFRR_U"][0] + self.p["mFRR_U"][0] <= self.P_max*0.8,                           #discharging cannot exceed power cap. Some cap is reserved for FCR.
                            
                            (self.e["c"][1:] + 1.25*self.p["FCRN"][1:] + self.p["FCRD_D"][1:] + 0.2*self.p["FCRD_U"][1:] + self.p["aFRR_D"][1:]) + (1/3*self.p["FCRD_U"][:-1] + 0.4*self.p["FCRN"][:-1] + self.p["aFRR_U"][:-1] + self.p["mFRR_U"][:-1])*self.Eta_down/self.Eta_up <= self.P_max*0.8,
                            (self.e["d"][1:] + 1.25*self.p["FCRN"][1:] + self.p["FCRD_U"][1:] + 0.2*self.p["FCRD_D"][1:] + self.p["aFRR_U"][1:] + self.p["mFRR_U"][1:]) + (1/3*self.p["FCRD_D"][:-1] + 0.4*self.p["FCRN"][:-1] + self.p["aFRR_D"][:-1])*self.Eta_up/self.Eta_down <= self.P_max*0.8,
                            
                            #Energy capacity constraints
                            (0.4*self.p["FCRN"][0] + 1/3*self.p["FCRD_D"][0] + self.p["aFRR_D"][0] + self.e["c"][0])*self.Eta_up - self.e["d"][0]*self.Eta_down 
                            + (0.4*self.p["FCRN"][1] + 1/3*self.p["FCRD_D"][1] + self.p["aFRR_D"][1] + self.e["c"][1])*self.Eta_up -  self.e["d"][1]*self.Eta_down
                            <= self.E_max - self.E_start,
                            
                            (0.4*self.p["FCRN"][1:-1] + 1/3*self.p["FCRD_D"][1:-1] + self.p["aFRR_D"][1:-1] +self.e["c"][1:-1])*self.Eta_up - self.e["d"][1:-1]*self.Eta_down 
                            + (0.4*self.p["FCRN"][2:] + 1/3*self.p["FCRD_D"][2:] + self.p["aFRR_D"][2:] +self.e["c"][2:])*self.Eta_up - self.e["d"][2:]*self.Eta_down 
                            <= self.E_max - self.e["b"][0:-2],   #battery needs to be able to sustain min 1 hour of downreg and cannot bid full in 2 consecutive hours
                            
                            (0.4*self.p["FCRN"][0] + 1/3*self.p["FCRD_U"][0] + self.p["aFRR_U"][0] + self.p["mFRR_U"][0] + self.e["d"][0])*self.Eta_down - self.e["c"][0]*self.Eta_up
                            + (0.4*self.p["FCRN"][1] + 1/3*self.p["FCRD_U"][1] + self.p["aFRR_U"][1] + self.p["mFRR_U"][1] + self.e["d"][1])*self.Eta_down - self.e["c"][1]*self.Eta_up
                            <= self.E_start - self.E_min,
                            
                            (0.4*self.p["FCRN"][1:-1] + 1/3*self.p["FCRD_U"][1:-1] + self.p["aFRR_U"][1:-1] + self.p["mFRR_U"][1:-1] + self.e["d"][1:-1])*self.Eta_down - self.e["c"][1:-1]*self.Eta_up 
                            + (0.4*self.p["FCRN"][2:] + 1/3*self.p["FCRD_U"][2:] + self.p["aFRR_U"][2:] + self.p["mFRR_U"][2:] + self.e["d"][2:])*self.Eta_down - self.e["c"][2:]*self.Eta_up 
                            <= self.e["b"][0:-2] - self.E_min, #battery needs to be able to sustain min 1 hour of upreg and cannot bid full in 2 consecutive hours
                            
                            self.e["d"] - self.e["c"] == self.p["DA"], #+ 1* cp.multiply(self.p["mFRR_U"],self.A_mFRR_U) + 1*cp.multiply(self.p["aFRR_U"],self.A_aFRR_U) - 1*cp.multiply(self.p["aFRR_D"],self.A_aFRR_D),
                            
                            #self.p["FCRD_U"] <= 0,self.p["FCRD_D"] <= 0#,self.p["FCRN"] <= 0,self.p["aFRR_U"] <= 0,self.p["aFRR_D"] <=0,self.p["mFRR_U"]<=0,
                            ]
        self.prob = cp.Problem(self.obj, self.constraints)
        #FCRN: LER units/portfolios must have storage capacity of minimum 1 hour to handle long lasting frequency deviations. file:///C:/Users/filip/Downloads/ancillary-services-to-be-delivered-in-denmark-tender-conditions-1-2-2024%20(1).pdf
    
    def set_params(self,P_max,E_max,E_min,Eta_up,Eta_down,Eta_lin,E_start,cl,cpx,et,ept):
        self.P_max.value =  P_max
        self.E_max.value =  E_max
        self.E_min.value =  E_min
        self.Eta_up.value = Eta_up
        self.Eta_down.value=Eta_down
        self.Eta_lin.value= Eta_lin
        self.E_start.value= E_start
        #self.Eta_ud.value = Eta_up/Eta_down
        #self.Eta_du.value = Eta_down/Eta_up
        self.CycleLife.value = cl
        self.capex.value = cpx
        self.E_tariff.value = et #200 # NOTE check
        self.EP_tariff.value = ept
    def set_prices(self,Price):
        for k in self.L.keys():
            if k != "H":
                self.L[k].value = Price[k].values
    def set_avg_act(self,act):
        mm = np.mean(act)
        self.E_act["mFRR_U"].value = mm["mFRR_U"]
        self.E_act["aFRR_U"].value = mm["aFRR_U"]
        self.E_act["aFRR_D"].value = mm["aFRR_D"]
        self.E_act["FCRN_D"].value = mm["FCRN_D"]
        self.E_act["FCRN_U"].value = mm["FCRN_U"]
        self.E_act["FCRD_D"].value = mm["FCRD_D"]
        self.E_act["FCRD_U"].value = mm["FCRD_U"]
        
        self.obj =  cp.Maximize(#self.p["DA"]
                                self.e["d"] @ (self.L["DA"]-self.EP_tariff) - self.e["c"] @ (self.L["DA"]+self.E_tariff)
                               + self.p["FCRN"] @ self.L["FCRN"] + self.p["FCRD_U"] @ self.L["FCRD_U"] + self.p["FCRD_D"] @ self.L["FCRD_D"] 
                               + self.p["aFRR_U"] @ self.L["aFRR_U"] + self.p["aFRR_D"] @ self.L["aFRR_D"] + self.p["mFRR_U"] @ self.L["mFRR_U"]
                               + cp.sum(self.r_act) + cp.sum(self.r_rec) - cp.sum(self.c_deg)
                               )
        
        self.prob = cp.Problem(self.obj, self.constraints)

    def update_E_start(self,t):
        self.E_start.value= self.e["b"].value[t]

    def excl_markets(self,m):
        if "FCRD" in m:
            self.constraints.append(self.p["FCRD_U"] <= 0)
            self.constraints.append(self.p["FCRD_D"] <= 0)
        if "FCRN" in m:
            self.constraints.append(self.p["FCRN"] <= 0)
        if "aFRR" in m:
            self.constraints.append(self.p["aFRR_D"] <= 0)
            self.constraints.append(self.p["aFRR_U"] <= 0)
        if "mFRR" in m:
            self.constraints.append(self.p["mFRR_U"] <= 0)
        
        self.prob = cp.Problem(self.obj, self.constraints)
    
    def solve(self):
        #print("dpp?: ",self.prob.is_dcp(dpp=True))
        #print("dcp?: ",self.prob.is_dcp(dpp=False))
        self.prob.solve(solver=cp.SCIP,verbose=False)
        #print("ok")
        if self.prob.status != 'optimal':
            print(str(self.prob.status))
        return self.prob.value
    def res(self,hours):
        self.l = hours
        
        return pd.DataFrame({"p_DA":self.p["DA"].value[:self.l],
                "p_FCRN":    self.p["FCRN"].value[:self.l],
                "p_FCRD_U":  self.p["FCRD_U"].value[:self.l],
                "p_FCRD_D":  self.p["FCRD_D"].value[:self.l],
                "p_aFRR_U":  self.p["aFRR_U"].value[:self.l],
                "p_aFRR_D":  self.p["aFRR_D"].value[:self.l],
                "p_mFRR_U":  self.p["mFRR_U"].value[:self.l],
                "e_b":       self.e["b"].value[:self.l],
                "e_c":       self.e["c"].value[:self.l],
                "e_d":       self.e["d"].value[:self.l]})
    def return_params(self):
        return self.P_max.value,self.E_max.value,self.E_min.value,self.Eta_up.value,self.Eta_down.value,self.Eta_lin.value,self.E_start.value
    #def return_prices(self):
    #    return self.L["DA"].value
    
    def run_yr(self,yrP,yrA,start_d,end_d):        
        resLen = self.OptLen-24
        optStart = pd.date_range(start=start_d, end=end_d,freq=str(resLen)+"h")
        self.set_avg_act(yrA)
        yr_res = {"p_DA":np.zeros((len(optStart),resLen)),
           "p_FCRN":    np.zeros((len(optStart),resLen)),
           "p_FCRD_U":  np.zeros((len(optStart),resLen)),
           "p_FCRD_D":  np.zeros((len(optStart),resLen)),
           "p_aFRR_U":  np.zeros((len(optStart),resLen)),
           "p_aFRR_D":  np.zeros((len(optStart),resLen)),
           "p_mFRR_U":  np.zeros((len(optStart),resLen)),
           "e_b":       np.zeros((len(optStart),resLen)),
           "e_c":       np.zeros((len(optStart),resLen)),
           "e_d":       np.zeros((len(optStart),resLen))}
        self.all_E_start = np.zeros(len(optStart))
        
        for i in range(len(optStart)):
            #print(cp.sum(self.c_deg.value))
            #print(optStart[i],)
            date_r = pd.date_range(start=optStart[i],periods=self.OptLen,freq="h")
            
            PP = pd.DataFrame(yrP)[date_r[0]:date_r[-1]]
            self.set_prices(Price=PP)
            self.solve()
            
            # if sum( (np.array(self.A_net_D.value)>0.01) *(np.array(self.A_net_U.value)>0.01) ) > 0:
            #     print(self.A_net_D.value[(np.array(self.A_net_D.value)>0.1) *(np.array(self.A_net_U.value)>0.1)])
            #     print(self.A_net_U.value[(np.array(self.A_net_D.value)>0.1) *(np.array(self.A_net_U.value)>0.1)])
            
            
            Bat_res = self.res(resLen)
            Bat_res.index = date_r[:resLen]
            
            for k in yr_res.keys():
                yr_res[k][i,:] = Bat_res[k]
            
            self.all_E_start[i] = self.return_params()[6]
            #print(self.return_params()[6])
            
            self.update_E_start(resLen-1)
            #print(self.return_params()[6])
            #print()
            
        self.RES = pd.DataFrame({'p_DA':yr_res["p_DA"].flatten(), 
            'p_FCRN':yr_res["p_FCRN"].flatten(), 
            'p_FCRD_U':yr_res["p_FCRD_U"].flatten(), 
            'p_FCRD_D':yr_res["p_FCRD_D"].flatten(), 
            'p_aFRR_U':yr_res["p_aFRR_U"].flatten(), 
            'p_aFRR_D':yr_res["p_aFRR_D"].flatten(), 
            'p_mFRR_U':yr_res["p_mFRR_U"].flatten(), 
            'e_b':yr_res["e_b"].flatten() ,
            'e_c':yr_res["e_c"].flatten(), 
            'e_d':yr_res["e_d"].flatten()})
        self.RES.index = pd.date_range(start=start_d, periods=len(self.RES),freq="h")
        self.RES = self.RES[str(self.RES.index.year[0])]
        
        self.all_E_start = pd.DataFrame(self.all_E_start,index=optStart)[0]

        return self.RES,self.all_E_start



def yr_rev_cyc_ST(bat,res,yrA,yrP,TC,TP):
    yrP = yrP["2023"]
    P,E,E_min,Eta_up,Eta_down,Eta_lin = bat.return_params()[:-1]
    C = P/E
    
    Real_SOE = res["e_b"].copy()*0
    rec_d = Real_SOE.copy()*0
    rec_c = Real_SOE.copy()*0
    max_d = - ((P)-(res["p_aFRR_U"]+res["p_mFRR_U"]+res["p_FCRN"]+res["p_FCRD_U"]+res["e_d"]))
    max_c = (P)-(res["p_aFRR_D"]+res["p_FCRN"]+res["p_FCRD_D"]+res["e_c"])
    
    r = pd.DataFrame({"DA":np.zeros(len(res)),
           "FCRN":np.zeros(len(res)),
           "FCRD_U":np.zeros(len(res)),
           "FCRD_D":np.zeros(len(res)),
           "aFRR_U":np.zeros(len(res)),
           "aFRR_D":np.zeros(len(res)),
           "mFRR_U":np.zeros(len(res)),
           "Act":np.zeros(len(res)),
           "Recovery":np.zeros(len(res)),
           "Tariffs":np.zeros(len(res))
           },index=res["2023"].index)
    #all trading and capacity pure revenue (no tariffs)
    for k in r.columns[0:7]:
        r[k] = res["p_"+k] * yrP[k]
    
    #activation revenue
    au= (yrA["aFRR_U"]*res["p_aFRR_U"] + yrA["mFRR_U"]*res["p_mFRR_U"] + yrA["FCRN_U"]*res["p_FCRN"] + yrA["FCRD_U"]*res["p_FCRD_U"])
    ad = (yrA["aFRR_D"]*res["p_aFRR_D"] +  yrA["FCRN_D"]*res["p_FCRN"] + yrA["FCRD_D"]*res["p_FCRD_D"])
    r["Act"][(au-ad)>0] = ((au-ad)*yrP["reg_U"])[(au-ad)>0]
    r["Act"][(au-ad)<=0] = ((au-ad)*yrP["reg_D"])[(au-ad)<=0]
    
    
    #True soe calc
    netA = -(yrA["aFRR_U"]*res["p_aFRR_U"] + yrA["mFRR_U"]*res["p_mFRR_U"] + yrA["FCRN_U"]*res["p_FCRN"] + yrA["FCRD_U"]*res["p_FCRD_U"])*Eta_down + (yrA["aFRR_D"]*res["p_aFRR_D"] +  yrA["FCRN_D"]*res["p_FCRN"] + yrA["FCRD_D"]*res["p_FCRD_D"])*Eta_up
    
    Real_SOE[0] = (E*0.5)*Eta_lin + netA[0] + res["e_c"][0]*Eta_up - res["e_d"][0]*Eta_down 

    for i in range(1,len(Real_SOE)-1):
        Real_SOE[i] = Real_SOE[i-1]*Eta_lin + netA[i] + res["e_c"][i]*Eta_up - res["e_d"][i]*Eta_down + rec_d[i]*Eta_down + rec_c[i]*Eta_up
        
        #if 1:#m > 2*1/C:
        if res["e_b"][i] - Real_SOE[i] < 0: 
            rec_d[i+1] = max( ( res["e_b"][i] - Real_SOE[i])*(1/Eta_down),max_d[i+1])
        elif res["e_b"][i] - Real_SOE[i] > 0: 
            rec_c[i+1] = min ( (res["e_b"][i] - Real_SOE[i])*(1/Eta_up),max_c[i+1])
        # if abs(res["e_b"][i] - Real_SOE[i]) < C*E: 
        #     m=0
        
        # #if Real_SOE[i]/E > 1-C*Eta_up or Real_SOE[i]/E - res["e_b"][i]/E > (2*C*Eta_up):
        # elif Real_SOE[i]/E > 1-C*Eta_up:
        #     rec_d[i+1] = max( ( E*(1-C) - Real_SOE[i])*Eta_up,max_d[i+1])
        #     #rec_d[i+1] = max( ( res["e_b"][i] - Real_SOE[i])*Eta_up,max_d[i+1])
        # #if Real_SOE[i]/E < C*Eta_down or res["e_b"][i]/E - Real_SOE[i]/E > (2*C*Eta_up):
        # elif Real_SOE[i]/E < C*Eta_down:
        #     rec_c[i+1] = min ( (E*(0+C) - Real_SOE[i])*Eta_down,max_c[i+1])
        #     #rec_c[i+1] = min ( (res["e_b"][i] - Real_SOE[i])*Eta_down,max_c[i+1])
        # m+=1
    Real_SOE[-1] = Real_SOE[-2]*Eta_lin + netA[-1] + res["e_c"][-1]*Eta_up - res["e_d"][-1]*Eta_down + rec_d[-1]*Eta_down + rec_c[-1]*Eta_up
    
    #print(sum(rec_c))
    #print(sum(rec_d))
    #print( sum((-rec_c -rec_d)*yrP["DA"]) )
    r["Recovery"] = (-(rec_c + rec_d))*yrP["DA"]
    
    netGrid = (res["p_DA"] + au - ad + rec_d - rec_c)
    r["Tariffs"][netGrid<0] = (netGrid*TC)[netGrid<0]
    r["Tariffs"][netGrid>=0] = - (netGrid*TP)[netGrid>=0]
    cyc = np.sum(abs(np.diff(np.concatenate(([0.5*E],Real_SOE))))/(2*E))
    return Real_SOE,rec_c,rec_d,r,cyc
    
    
def dcf(rev,LT,r_DC,r_T,OPEX,CAPEX): #yrRev,inflation
    #DCF analysis assumes investment cost in the end of year 0
    #operation starts from year 1 and lasts until the end of the lifetime
    
    LT = int(LT)
    inflation = np.ones(LT+1)*0.02#temp
    
    #OPEX = 570*self.P_max.value + 2.13*self.E_max.value #needs inflation correction
    Operation_year = np.arange(0,LT+1)
    inf0 = np.zeros(LT+1)
    inf0[0] = 1
    for i in range(LT):
        inf0[i+1] = inf0[i]*(1+inflation[i])
    
    
    DF = pd.DataFrame({
                       "Revenue(Real)":np.zeros(LT+1),
                          "Revenue(Nominal)":np.zeros(LT+1),
                          "OPEX(Real)":np.zeros(LT+1),
                          "OPEX(Nominal)":np.zeros(LT+1),
                          "EBITDA":np.zeros(LT+1),
                          "CAPEX":np.zeros(LT+1),
                          "Depreciation[%]":np.zeros(LT+1),
                          "Depreciation[EUR]":np.zeros(LT+1),
                          "EBIT":np.zeros(LT+1),
                          "Payable Tax":np.zeros(LT+1),
                          "CF(Nominal)":np.zeros(LT+1),
                          "DC index":np.zeros(LT+1),
                          "DCF":np.zeros(LT+1)},index=np.arange(LT+1))
    
    DF["Revenue(Real)"][1:]=np.ones(LT)*sum(np.sum(rev))
    DF["Revenue(Nominal)"]=DF["Revenue(Real)"]*inf0      
    DF["OPEX(Real)"][1:] = np.ones(LT)*OPEX
    DF["OPEX(Nominal)"] = DF["OPEX(Real)"]*inf0
    DF["EBITDA"] = DF["OPEX(Nominal)"]+DF["Revenue(Nominal)"]
    DF["CAPEX"][0] = CAPEX*inf0[0]
    DF["Depreciation[%]"][1:] = 1/LT
    DF["Depreciation[EUR"]=DF["Depreciation[%]"]*CAPEX
    DF["EBIT"] = DF["EBITDA"]+DF["Depreciation[EUR"]
    DF["Payable Tax"] = -DF["EBIT"]*r_T
    DF["CF(Nominal)"] = DF["Revenue(Nominal)"] + DF["OPEX(Nominal)"] + DF["CAPEX"] + DF["Payable Tax"]
    DF["DC index"] = 1/(1+r_DC)**Operation_year
    DF["DCF"] = DF["CF(Nominal)"]*DF["DC index"]
    
    NPV = np.sum(DF["DCF"])
    #print(DF["CF(Nominal)"])
    IRR = npf.irr(DF["CF(Nominal)"])
    

    return NPV,IRR,DF

#%%
# OL = 8*24
# f_disc = 0.08
# f_tax = 0.22
# bs = st_Battery(OL)
# #bs.excl_markets(["FCRD"])

# P2 = P.copy()
# #P2["FCRN"] = GP23["FCR"]
# P2["2024"]=0


# E= 10
# C = 1
# eff = 0.922
# eff_l = 0.999
# bs.set_params(E*C,E,E*0.2,eff,1/eff,eff_l,0.5*E,0,0,147.3,12.1)
# res,e0 = bs.run_yr(P2,Act,start_d="2023-01-01",end_d="2023-12-31 12")

# tt,ttrc,ttrd,rr,cc = yr_rev_cyc_ST(bs, res, Act, P2,147.3,12.1)


# npv,irr,c = dcf(rr,15,0.08,0.22,-9219.1*E*C,np.multiply(-394,E*10**3) - 368.76*10**3*E*C)






