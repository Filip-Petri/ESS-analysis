
from Load_data import *
#%%
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import cvxopt
import os
import time
import pyscipopt
class H_storage_L:
    def __init__(self, OptLen):
        self.OptLen = OptLen
        self.P_max,self.E_max,self.E_start = (cp.Parameter(),cp.Parameter(),cp.Parameter())
        self.Dmax = cp.Parameter()
        self.capex = cp.Parameter()
        self.CycleLife = cp.Parameter()


        self.L = {"DA": cp.Parameter(self.OptLen),
                  "reg_U":cp.Parameter(self.OptLen),
                  "reg_D":cp.Parameter(self.OptLen),
                  "FCRN": cp.Parameter(self.OptLen),
                  "FCRD_U": cp.Parameter(self.OptLen),
                  "FCRD_D": cp.Parameter(self.OptLen),
                  "aFRR_U": cp.Parameter(self.OptLen),
                  "aFRR_D": cp.Parameter(self.OptLen),
                  "mFRR_U": cp.Parameter(self.OptLen),
                  "H": cp.Parameter(self.OptLen)}
        
        self.p = {"DA": cp.Variable(self.OptLen,nonpos=True),
                  "H": cp.Variable(self.OptLen,nonneg=True),
                  "FCRN": cp.Variable(self.OptLen,nonneg=True),
                  "FCRD_U": cp.Variable(self.OptLen,nonneg=True),
                  "FCRD_D": cp.Variable(self.OptLen,nonneg=True),
                  "aFRR_U": cp.Variable(self.OptLen,nonneg=True),
                  "aFRR_D": cp.Variable(self.OptLen,nonneg=True),
                  "mFRR_U": cp.Variable(self.OptLen,nonneg=True)}
        
        self.e = {"b":cp.Variable(self.OptLen,nonneg=True)}
        

        
        # self.A_U = cp.Variable(self.OptLen,nonneg=True)
        # self.A_D = cp.Variable(self.OptLen,nonneg=True)
        # self.A_net_U = cp.Variable(self.OptLen,nonneg=True)
        # self.A_net_D = cp.Variable(self.OptLen,nonneg=True)

        
        self.Tariff = cp.Parameter() # NOTE check
        
        self.obj = cp.Maximize(self.p["H"]@ self.L["H"] + self.p["DA"]@ (self.L["DA"] + self.Tariff)
                  + self.p["FCRN"]@self.L["FCRN"] + self.p["FCRD_U"] @ self.L["FCRD_U"] + self.p["FCRD_D"] @ self.L["FCRD_D"]
                  + self.p["aFRR_U"] @ self.L["aFRR_U"] + self.p["aFRR_D"] @ self.L["aFRR_D"] + self.p["mFRR_U"] @ self.L["mFRR_U"]
                  )
                               
                               

        #cp.sum(2*(self.E_act["mFRR_U"][:-1]*self.p["mFRR_U"][:-1]*self.L["reg_U"][:-1] + self.E_act["aFRR_U"][:-1]*self.p["aFRR_U"][:-1] + self.E_act["aFRR_D"][:-1]*self.p["aFRR_D"][:-1] + self.E_act["FCRN_D"][:-1]*self.p["FCRN"][:-1] + self.E_act["FCRN_U"][:-1]*self.p["FCRN"][:-1] + self.E_act["FCRD_U"][:-1]*self.p["FCRD_U"][:-1] + self.E_act["FCRD_D"][:-1]*self.p["FCRD_D"][:-1]))
                               
                               
        self.pc = {"e":cp.Variable(self.OptLen,nonneg=True),
                   "c":cp.Variable(self.OptLen,nonneg=True)}
        
        #self.tpe = cp.Variable(self.OptLen,nonneg=True)
        
        #self.z = {"on":cp.Variable(self.OptLen,boolean=True),
        #          "sb":cp.Variable(self.OptLen,boolean=True)}
        
        self.h_p = cp.Variable(OptLen,nonneg=True)
        self.hp_max = cp.Variable(OptLen,nonneg=True)
        self.hp_min = cp.Variable(OptLen,nonneg=True)
        

        #Psb = 0.05
        Pmin = 0.16
        self.A = 1/33.33*1000*0.6
        B = 0#-self.A*Pmin #*Pmax
        Ce = 1.67/1000 #MWh/kg
        
        #+ cp.multiply(self.A_mFRR_U,self.p["mFRR_U"]) @ self.L_reg_U + cp.multiply(self.A_aFRR_U,self.p["aFRR_U"]) @ self.L_reg_U - cp.multiply(self.A_aFRR_D,self.p["aFRR_D"]) @ self.L_reg_D )#- sum((self.e["d"]+self.e["c"]+cp.multiply(self.A_aFRR_D,self.p["aFRR_D"])*self.Eta_up + cp.multiply(self.A_aFRR_U,self.p["aFRR_U"])*self.Eta_down + cp.multiply(self.A_mFRR_U,self.p["mFRR_U"])*self.Eta_down )/self.E_max/8000 *7.8*10**6 /2) )#+ 1*p_mFRR_U @ AP_mFRR_U + 1*p_aFRR_U @ AP_aFRR_U - 1*p_aFRR_D @ AP_aFRR_D)
        self.constraints = [
            #Day-ahead market relation to power consumption
            #self.z["on"][0]==0, #what
            #self.z["on"][3]==1,
            
            self.pc["e"] + self.pc["c"] == -self.p["DA"],
            
            #limits for electrolyzer power consumption
            Pmin*self.P_max <= self.pc["e"], #Pmin = 0.16*self.P_max #Psb = 0.05*P_max
            self.pc["e"] <= self.P_max, #Psb = 0.05*P_max
            
            #Pmin * z["on"]  + Psb * z["sb"] <= pc["e"] - (p["FCRN"] + p["FCRD_D"] + p["aFRR_D"]),
            #pc["e"] + (p["FCRN"] + p["FCRD_U"] + p["aFRR_U"] + p["mFRR_U"]) <= Pmax * z["on"]+ Psb * z["sb"],
            
            #Hydrogen production is a function of electrolyzer power
            self.h_p == self.A*self.pc["e"] + self.P_max*B,
            
            #power range for electrolyzer in on-mode
            #Pmin*self.P_max <= self.tpe,
            #self.tpe <= self.P_max,
            
            #total electrolyzer power
            #self.pc["e"] == self.tpe,
            #compressor power
            self.pc["c"] == Ce*self.h_p,
            
            #max hydrogen demand
            self.p["H"] <= self.Dmax,
            
            #cp.sum(self.p["H"]) >= self.P_max * A * 0.5,
            
            #min demand for all hours in period
            #sum(self.p["H"]) >= self.Dmax*len(self.Dmax)/2,
            #sum(self.p["H"]) >= self.Dmax*self.OptLen/3,
            
            #State of energy in hydrogen tank
            self.e["b"][0] == self.E_start + self.h_p[0] - self.p["H"][0],
            self.e["b"][1:] == self.e["b"][:-1] + self.h_p[1:] - self.p["H"][1:],
            self.e["b"][-1] == self.E_max/2,
            
            
            #Ancillary service hydrogen tank limits (for potential activation)      
            self.e["b"] + self.hp_max <= self.E_max,    
            self.e["b"] - self.hp_min >= 0, 
            
            #min/max hydrogen production upon activation
            self.hp_max == self.A*(self.pc["e"] +(self.p["FCRN"] + 1/3*self.p["FCRD_D"] + self.p["aFRR_D"]) ) + self.P_max*B,
            self.hp_min == self.A*(self.pc["e"] -(self.p["FCRN"] + 1/3*self.p["FCRD_U"] + self.p["aFRR_U"] + self.p["mFRR_U"])) + self.P_max*B,
            
            #All ancillary services are only active during on-mode
            #(self.p["FCRN"] + self.p["FCRD_D"] + self.p["aFRR_D"]) + (self.p["FCRN"] + self.p["FCRD_U"] + self.p["aFRR_U"] + self.p["mFRR_U"]) <= 2*self.P_max,
            #Last hour cannot have ancillary services
            (self.p["FCRN"][-1] + self.p["FCRD_D"][-1] + self.p["aFRR_D"][-1]) + (self.p["FCRN"][-1] + self.p["FCRD_U"][-1] + self.p["aFRR_U"][-1] + self.p["mFRR_U"][-1]) == 0,
            
            #ancillary service power limits
            (self.p["FCRN"] + self.p["FCRD_D"] + self.p["aFRR_D"]) <=  self.P_max - self.pc["e"],
            (self.p["FCRN"] + self.p["FCRD_U"] + self.p["aFRR_U"] + self.p["mFRR_U"]) <= self.pc["e"] - Pmin*self.P_max,
            
            #reserve tank capacity for recovery (on activation)
            #self.p["H"][1:] <= self.Dmax - (self.hp_max[:-1]-self.h_p[:-1]),
            #self.p["H"][1:] >= (self.h_p[:-1] - self.hp_min[:-1])
            self.p["H"] <= self.Dmax - (self.hp_max-self.h_p),
            self.p["H"] >= (self.h_p - self.hp_min)
            ]
        self.prob = cp.Problem(self.obj, self.constraints)
        
        #needs to reserve electrolyzer capacity in the following hour to recover from the excess/deficit of activation hydrogen 
        #and remember to include calcs in rev_year
    def set_params(self,P_max,E_max,E_start,Dmax,cl,cpx,et):
        self.P_max.value =  P_max
        self.E_max.value =  E_max
        self.E_start.value= E_start
        self.Dmax.value = Dmax
        
        self.CycleLife.value = cl
        self.capex.value = cpx
        self.Tariff.value = et
    def set_prices(self,Price):
        for k in self.L.keys():
            self.L[k].value = Price[k].values
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
        #self.prob.solve()
        #self.prob.solve(cp.HIGHS)
        self.prob.solve(cp.SCIP)#scipy_options={'limits/gap': 1e-3}
        
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
                "p_H":       self.p["H"].value[:self.l],
                "pc_e":       self.pc["e"].value[:self.l]
                })

    def return_params(self):
        return self.P_max.value,self.E_max.value,self.E_start.value,self.Dmax.value, self.A
    def set_avg_act(self,act):
        mm = np.mean(act)
        self.E_act = {"mFRR_U":cp.Parameter(),
            "aFRR_U":cp.Parameter(),    
            "aFRR_D":cp.Parameter(),
            "FCRN_D":cp.Parameter(),
            "FCRN_U":cp.Parameter(),
            "FCRD_D":cp.Parameter(),
            "FCRD_U":cp.Parameter()}
        
        
        
        self.E_act["mFRR_U"].value = mm["mFRR_U"]
        self.E_act["aFRR_U"].value = mm["aFRR_U"]
        self.E_act["aFRR_D"].value = mm["aFRR_D"]
        self.E_act["FCRN_D"].value = mm["FCRN_D"]
        self.E_act["FCRN_U"].value = mm["FCRN_U"]
        self.E_act["FCRD_D"].value = mm["FCRD_D"]
        self.E_act["FCRD_U"].value = mm["FCRD_U"]
        
        
        self.r_act  = cp.Variable(self.OptLen)
        self.A_U = cp.Variable(self.OptLen,nonneg=True)
        self.A_D = cp.Variable(self.OptLen,nonneg=True)
        
        self.obj = cp.Maximize(self.p["H"]@ self.L["H"] + self.p["DA"]@ (self.L["DA"] + self.Tariff)
                  + self.p["FCRN"]@self.L["FCRN"] + self.p["FCRD_U"] @ self.L["FCRD_U"] + self.p["FCRD_D"] @ self.L["FCRD_D"]
                  + self.p["aFRR_U"] @ self.L["aFRR_U"] + self.p["aFRR_D"] @ self.L["aFRR_D"] + self.p["mFRR_U"] @ self.L["mFRR_U"]
                  + cp.sum(self.r_act) -cp.sum(self.e["b"]*0.001)
                  )
        
        self.constraints.append(self.r_act == cp.multiply(self.A_U,self.L["reg_U"]+self.Tariff)-cp.multiply(self.A_D,self.L["reg_D"] + self.Tariff) )
        self.constraints.append(self.r_act[-1]==0)
        self.constraints.append(self.A_U == (self.E_act["mFRR_U"]*self.p["mFRR_U"] + self.E_act["aFRR_U"]*self.p["aFRR_U"] + self.E_act["FCRN_U"]*self.p["FCRN"] + self.E_act["FCRD_U"]*self.p["FCRD_U"]))
        self.constraints.append(self.A_D == (self.E_act["aFRR_D"]*self.p["aFRR_D"] + self.E_act["FCRN_D"]*self.p["FCRN"] + self.E_act["FCRD_D"]*self.p["FCRD_D"]))
        
        
        #cycle cost - note 
        #)#+cp.sum(self.e["d"] + self.e["c"] + 2*( (self.E_act["mFRR_U"]*self.p["mFRR_U"] + self.E_act["aFRR_U"]*self.p["aFRR_U"] + self.E_act["FCRN_U"]*self.p["FCRN"] + self.E_act["FCRD_U"]*self.p["FCRD_U"])*self.Eta_down.value + (self.E_act["aFRR_D"]*self.p["aFRR_D"] + self.E_act["FCRN_D"]*self.p["FCRN"]  + self.E_act["FCRD_D"]*self.p["FCRD_D"])*self.Eta_up.value))*np.round(CAPEX / (2*self.E_max.value)/100000,2 ) )
        
        self.prob = cp.Problem(self.obj, self.constraints)
    
    def run_yr(self,yrP,yrA,start_d,end_d):   
        start_time = time.time()
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
            "p_H":       np.zeros((len(optStart),resLen)),
            "pc_e":      np.zeros((len(optStart),resLen))}
        self.all_E_start = np.zeros(len(optStart))
        
        for i in range(len(optStart)):
            
            #print(optStart[i],)
            date_r = pd.date_range(start=optStart[i],periods=self.OptLen,freq="h")
            
            PP = pd.DataFrame(yrP)[date_r[0]:date_r[-1]]
            self.set_prices(Price=PP)
            self.solve()
            print("solve: "+str(i)+", elapsed time: "+str(time.time() - start_time))

            # if sum( (np.array(self.A_net_D.value)>0.01) *(np.array(self.A_net_U.value)>0.01) ) > 0:
            #     print(self.A_net_D.value[(np.array(self.A_net_D.value)>0.1) *(np.array(self.A_net_U.value)>0.1)])
            #     print(self.A_net_U.value[(np.array(self.A_net_D.value)>0.1) *(np.array(self.A_net_U.value)>0.1)])
            
            
            Bat_res = self.res(resLen)
            Bat_res.index = date_r[:resLen]
            
            for k in yr_res.keys():
                yr_res[k][i,:] = Bat_res[k]
            
            self.all_E_start[i] = self.return_params()[2]
            self.update_E_start(resLen-1)
            
        self.RES = pd.DataFrame({'p_DA':yr_res["p_DA"].flatten(), 
            'p_FCRN':yr_res["p_FCRN"].flatten(), 
            'p_FCRD_U':yr_res["p_FCRD_U"].flatten(), 
            'p_FCRD_D':yr_res["p_FCRD_D"].flatten(), 
            'p_aFRR_U':yr_res["p_aFRR_U"].flatten(), 
            'p_aFRR_D':yr_res["p_aFRR_D"].flatten(), 
            'p_mFRR_U':yr_res["p_mFRR_U"].flatten(), 
            'e_b':yr_res["e_b"].flatten(),
            'p_H':yr_res["p_H"].flatten(),
            'pc_e':yr_res["pc_e"].flatten()})
        self.RES.index = pd.date_range(start=start_d, periods=len(self.RES),freq="h")
        self.RES = self.RES[str(self.RES.index.year[0])]
        
        self.all_E_start = pd.DataFrame(self.all_E_start,index=optStart)[0]

        return self.RES,self.all_E_start
class FCR_H_storage_L:
    def __init__(self, OptLen):
        self.OptLen = OptLen
        self.P_max,self.E_max,self.E_start = (cp.Parameter(),cp.Parameter(),cp.Parameter())
        self.Dmax = cp.Parameter()
        self.capex = cp.Parameter()
        self.CycleLife = cp.Parameter()


        self.L = {"DA": cp.Parameter(self.OptLen),
                  "reg_U":cp.Parameter(self.OptLen),
                  "reg_D":cp.Parameter(self.OptLen),
                  "FCRN": cp.Parameter(self.OptLen),
                  "FCRD_U": cp.Parameter(self.OptLen),
                  "FCRD_D": cp.Parameter(self.OptLen),
                  "aFRR_U": cp.Parameter(self.OptLen),
                  "aFRR_D": cp.Parameter(self.OptLen),
                  "mFRR_U": cp.Parameter(self.OptLen),
                  "H": cp.Parameter(self.OptLen)}
        
        self.p = {"DA": cp.Variable(self.OptLen,nonpos=True),
                  "H": cp.Variable(self.OptLen,nonneg=True),
                  "FCRN": cp.Variable(self.OptLen,nonneg=True),
                  "FCRD_U": cp.Variable(self.OptLen,nonneg=True),
                  "FCRD_D": cp.Variable(self.OptLen,nonneg=True),
                  "aFRR_U": cp.Variable(self.OptLen,nonneg=True),
                  "aFRR_D": cp.Variable(self.OptLen,nonneg=True),
                  "mFRR_U": cp.Variable(self.OptLen,nonneg=True)}
        
        self.e = {"b":cp.Variable(self.OptLen,nonneg=True)}
        

        
        # self.A_U = cp.Variable(self.OptLen,nonneg=True)
        # self.A_D = cp.Variable(self.OptLen,nonneg=True)
        # self.A_net_U = cp.Variable(self.OptLen,nonneg=True)
        # self.A_net_D = cp.Variable(self.OptLen,nonneg=True)

        
        self.Tariff = cp.Parameter() # NOTE check
        
        self.obj = cp.Maximize(self.p["H"]@ self.L["H"] + self.p["DA"]@ (self.L["DA"] + self.Tariff)
                  + self.p["FCRN"]@self.L["FCRN"] + self.p["FCRD_U"] @ self.L["FCRD_U"] + self.p["FCRD_D"] @ self.L["FCRD_D"]
                  + self.p["aFRR_U"] @ self.L["aFRR_U"] + self.p["aFRR_D"] @ self.L["aFRR_D"] + self.p["mFRR_U"] @ self.L["mFRR_U"]
                  )
                               
                               

        #cp.sum(2*(self.E_act["mFRR_U"][:-1]*self.p["mFRR_U"][:-1]*self.L["reg_U"][:-1] + self.E_act["aFRR_U"][:-1]*self.p["aFRR_U"][:-1] + self.E_act["aFRR_D"][:-1]*self.p["aFRR_D"][:-1] + self.E_act["FCRN_D"][:-1]*self.p["FCRN"][:-1] + self.E_act["FCRN_U"][:-1]*self.p["FCRN"][:-1] + self.E_act["FCRD_U"][:-1]*self.p["FCRD_U"][:-1] + self.E_act["FCRD_D"][:-1]*self.p["FCRD_D"][:-1]))
                               
                               
        self.pc = {"e":cp.Variable(self.OptLen,nonneg=True),
                   "c":cp.Variable(self.OptLen,nonneg=True)}
        
        #self.tpe = cp.Variable(self.OptLen,nonneg=True)
        
        #self.z = {"on":cp.Variable(self.OptLen,boolean=True),
        #          "sb":cp.Variable(self.OptLen,boolean=True)}
        
        self.h_p = cp.Variable(OptLen,nonneg=True)
        self.hp_max = cp.Variable(OptLen,nonneg=True)
        self.hp_min = cp.Variable(OptLen,nonneg=True)
        

        #Psb = 0.05
        Pmin = 0.16
        self.A = 1/33.33*1000*0.6
        B = 0#-self.A*Pmin #*Pmax
        Ce = 1.67/1000 #MWh/kg
        
        #+ cp.multiply(self.A_mFRR_U,self.p["mFRR_U"]) @ self.L_reg_U + cp.multiply(self.A_aFRR_U,self.p["aFRR_U"]) @ self.L_reg_U - cp.multiply(self.A_aFRR_D,self.p["aFRR_D"]) @ self.L_reg_D )#- sum((self.e["d"]+self.e["c"]+cp.multiply(self.A_aFRR_D,self.p["aFRR_D"])*self.Eta_up + cp.multiply(self.A_aFRR_U,self.p["aFRR_U"])*self.Eta_down + cp.multiply(self.A_mFRR_U,self.p["mFRR_U"])*self.Eta_down )/self.E_max/8000 *7.8*10**6 /2) )#+ 1*p_mFRR_U @ AP_mFRR_U + 1*p_aFRR_U @ AP_aFRR_U - 1*p_aFRR_D @ AP_aFRR_D)
        self.constraints = [
            #Day-ahead market relation to power consumption
            #self.z["on"][0]==0, #what
            #self.z["on"][3]==1,
            
            self.pc["e"] + self.pc["c"] == -self.p["DA"],
            
            #limits for electrolyzer power consumption
            Pmin*self.P_max <= self.pc["e"], #Pmin = 0.16*self.P_max #Psb = 0.05*P_max
            self.pc["e"] <= self.P_max, #Psb = 0.05*P_max
            
            #Pmin * z["on"]  + Psb * z["sb"] <= pc["e"] - (p["FCRN"] + p["FCRD_D"] + p["aFRR_D"]),
            #pc["e"] + (p["FCRN"] + p["FCRD_U"] + p["aFRR_U"] + p["mFRR_U"]) <= Pmax * z["on"]+ Psb * z["sb"],
            
            #Hydrogen production is a function of electrolyzer power
            self.h_p == self.A*self.pc["e"] + self.P_max*B,
            
            #power range for electrolyzer in on-mode
            #Pmin*self.P_max <= self.tpe,
            #self.tpe <= self.P_max,
            
            #total electrolyzer power
            #self.pc["e"] == self.tpe,
            #compressor power
            self.pc["c"] == Ce*self.h_p,
            
            #max hydrogen demand
            self.p["H"] <= self.Dmax,
            
            #cp.sum(self.p["H"]) >= self.P_max * A * 0.5,
            
            #min demand for all hours in period
            #sum(self.p["H"]) >= self.Dmax*len(self.Dmax)/2,
            #sum(self.p["H"]) >= self.Dmax*self.OptLen/3,
            
            #State of energy in hydrogen tank
            self.e["b"][0] == self.E_start + self.h_p[0] - self.p["H"][0],
            self.e["b"][1:] == self.e["b"][:-1] + self.h_p[1:] - self.p["H"][1:],
            self.e["b"][-1] == self.E_max/2,
            
            
            #Ancillary service hydrogen tank limits (for potential activation)      
            self.e["b"] + self.hp_max <= self.E_max,    
            self.e["b"] - self.hp_min >= 0, 
            
            #min/max hydrogen production upon activation
            self.hp_max == self.A*(self.pc["e"] +(0.4*self.p["FCRN"] + 1/3*self.p["FCRD_D"] + self.p["aFRR_D"]) ) + self.P_max*B,
            self.hp_min == self.A*(self.pc["e"] -(0.4*self.p["FCRN"] + 1/3*self.p["FCRD_U"] + self.p["aFRR_U"] + self.p["mFRR_U"])) + self.P_max*B,
            
            #All ancillary services are only active during on-mode
            #(self.p["FCRN"] + self.p["FCRD_D"] + self.p["aFRR_D"]) + (self.p["FCRN"] + self.p["FCRD_U"] + self.p["aFRR_U"] + self.p["mFRR_U"]) <= 2*self.P_max,
            #Last hour cannot have ancillary services
            (self.p["FCRN"][-1] + self.p["FCRD_D"][-1] + self.p["aFRR_D"][-1]) + (self.p["FCRN"][-1] + self.p["FCRD_U"][-1] + self.p["aFRR_U"][-1] + self.p["mFRR_U"][-1]) == 0,
            
            #ancillary service power limits
            (self.p["FCRN"] + self.p["FCRD_D"] + self.p["aFRR_D"]) <=  self.P_max - self.pc["e"],
            (self.p["FCRN"] + self.p["FCRD_U"] + self.p["aFRR_U"] + self.p["mFRR_U"]) <= self.pc["e"] - Pmin*self.P_max,
            
            #reserve tank capacity for recovery (on activation)
            #self.p["H"][1:] <= self.Dmax - (self.hp_max[:-1]-self.h_p[:-1]),
            #self.p["H"][1:] >= (self.h_p[:-1] - self.hp_min[:-1])
            self.p["H"] <= self.Dmax - (self.hp_max-self.h_p),
            self.p["H"] >= (self.h_p - self.hp_min)
            ]
        self.prob = cp.Problem(self.obj, self.constraints)
        
        #needs to reserve electrolyzer capacity in the following hour to recover from the excess/deficit of activation hydrogen 
        #and remember to include calcs in rev_year
    def set_params(self,P_max,E_max,E_start,Dmax,cl,cpx,et):
        self.P_max.value =  P_max
        self.E_max.value =  E_max
        self.E_start.value= E_start
        self.Dmax.value = Dmax
        
        self.CycleLife.value = cl
        self.capex.value = cpx
        self.Tariff.value = et
    def set_prices(self,Price):
        for k in self.L.keys():
            self.L[k].value = Price[k].values
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
        #self.prob.solve()
        #self.prob.solve(cp.HIGHS)
        self.prob.solve(cp.SCIP)#scipy_options={'limits/gap': 1e-3}
        
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
                "p_H":       self.p["H"].value[:self.l],
                "pc_e":       self.pc["e"].value[:self.l]
                })

    def return_params(self):
        return self.P_max.value,self.E_max.value,self.E_start.value,self.Dmax.value, self.A
    def set_avg_act(self,act):
        mm = np.mean(act)
        self.E_act = {"mFRR_U":cp.Parameter(),
            "aFRR_U":cp.Parameter(),    
            "aFRR_D":cp.Parameter(),
            "FCRN_D":cp.Parameter(),
            "FCRN_U":cp.Parameter(),
            "FCRD_D":cp.Parameter(),
            "FCRD_U":cp.Parameter()}
        
        
        
        self.E_act["mFRR_U"].value = mm["mFRR_U"]
        self.E_act["aFRR_U"].value = mm["aFRR_U"]
        self.E_act["aFRR_D"].value = mm["aFRR_D"]
        self.E_act["FCRN_D"].value = mm["FCRN_D"]
        self.E_act["FCRN_U"].value = mm["FCRN_U"]
        self.E_act["FCRD_D"].value = mm["FCRD_D"]
        self.E_act["FCRD_U"].value = mm["FCRD_U"]
        
        
        self.r_act  = cp.Variable(self.OptLen)
        self.A_U = cp.Variable(self.OptLen,nonneg=True)
        self.A_D = cp.Variable(self.OptLen,nonneg=True)
        
        self.obj = cp.Maximize(self.p["H"]@ self.L["H"] + self.p["DA"]@ (self.L["DA"] + self.Tariff)
                  + self.p["FCRN"]@self.L["FCRN"] + self.p["FCRD_U"] @ self.L["FCRD_U"] + self.p["FCRD_D"] @ self.L["FCRD_D"]
                  + self.p["aFRR_U"] @ self.L["aFRR_U"] + self.p["aFRR_D"] @ self.L["aFRR_D"] + self.p["mFRR_U"] @ self.L["mFRR_U"]
                  + cp.sum(self.r_act) -cp.sum(self.e["b"]*0.001)
                  )
        
        self.constraints.append(self.r_act == cp.multiply(self.A_U,self.L["reg_U"]+self.Tariff)-cp.multiply(self.A_D,self.L["reg_D"] + self.Tariff) )
        self.constraints.append(self.r_act[-1]==0)
        self.constraints.append(self.A_U == (self.E_act["mFRR_U"]*self.p["mFRR_U"] + self.E_act["aFRR_U"]*self.p["aFRR_U"] + self.E_act["FCRN_U"]*self.p["FCRN"] + self.E_act["FCRD_U"]*self.p["FCRD_U"]))
        self.constraints.append(self.A_D == (self.E_act["aFRR_D"]*self.p["aFRR_D"] + self.E_act["FCRN_D"]*self.p["FCRN"] + self.E_act["FCRD_D"]*self.p["FCRD_D"]))
        
        
        #cycle cost - note 
        #)#+cp.sum(self.e["d"] + self.e["c"] + 2*( (self.E_act["mFRR_U"]*self.p["mFRR_U"] + self.E_act["aFRR_U"]*self.p["aFRR_U"] + self.E_act["FCRN_U"]*self.p["FCRN"] + self.E_act["FCRD_U"]*self.p["FCRD_U"])*self.Eta_down.value + (self.E_act["aFRR_D"]*self.p["aFRR_D"] + self.E_act["FCRN_D"]*self.p["FCRN"]  + self.E_act["FCRD_D"]*self.p["FCRD_D"])*self.Eta_up.value))*np.round(CAPEX / (2*self.E_max.value)/100000,2 ) )
        
        self.prob = cp.Problem(self.obj, self.constraints)
    
    def run_yr(self,yrP,yrA,start_d,end_d):   
        start_time = time.time()
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
            "p_H":       np.zeros((len(optStart),resLen)),
            "pc_e":      np.zeros((len(optStart),resLen))}
        self.all_E_start = np.zeros(len(optStart))
        
        for i in range(len(optStart)):
            
            #print(optStart[i],)
            date_r = pd.date_range(start=optStart[i],periods=self.OptLen,freq="h")
            
            PP = pd.DataFrame(yrP)[date_r[0]:date_r[-1]]
            self.set_prices(Price=PP)
            self.solve()
            print("solve: "+str(i)+", elapsed time: "+str(time.time() - start_time))

            # if sum( (np.array(self.A_net_D.value)>0.01) *(np.array(self.A_net_U.value)>0.01) ) > 0:
            #     print(self.A_net_D.value[(np.array(self.A_net_D.value)>0.1) *(np.array(self.A_net_U.value)>0.1)])
            #     print(self.A_net_U.value[(np.array(self.A_net_D.value)>0.1) *(np.array(self.A_net_U.value)>0.1)])
            
            
            Bat_res = self.res(resLen)
            Bat_res.index = date_r[:resLen]
            
            for k in yr_res.keys():
                yr_res[k][i,:] = Bat_res[k]
            
            self.all_E_start[i] = self.return_params()[2]
            self.update_E_start(resLen-1)
            
        self.RES = pd.DataFrame({'p_DA':yr_res["p_DA"].flatten(), 
            'p_FCRN':yr_res["p_FCRN"].flatten(), 
            'p_FCRD_U':yr_res["p_FCRD_U"].flatten(), 
            'p_FCRD_D':yr_res["p_FCRD_D"].flatten(), 
            'p_aFRR_U':yr_res["p_aFRR_U"].flatten(), 
            'p_aFRR_D':yr_res["p_aFRR_D"].flatten(), 
            'p_mFRR_U':yr_res["p_mFRR_U"].flatten(), 
            'e_b':yr_res["e_b"].flatten(),
            'p_H':yr_res["p_H"].flatten(),
            'pc_e':yr_res["pc_e"].flatten()})
        self.RES.index = pd.date_range(start=start_d, periods=len(self.RES),freq="h")
        self.RES = self.RES[str(self.RES.index.year[0])]
        
        self.all_E_start = pd.DataFrame(self.all_E_start,index=optStart)[0]

        return self.RES,self.all_E_start


def yr_rev_cyc_H(hb,res,yrA,yrP,TC):
    yrP = yrP["2023"]
    P,E,dm,PH = np.array(hb.return_params())[[0,1,3,4]]
    
    Real_SOE = res["e_b"].copy()*0
    deH = Real_SOE.copy()*0
    #rec_c = Real_SOE.copy()*0
    #max_d = - ((P)-(res["p_aFRR_U"]+res["p_mFRR_U"]+res["p_FCRN"]+res["p_FCRD_U"]))
    #max_c = (P)-(res["p_aFRR_D"]+res["p_FCRN"]+res["p_FCRD_D"]+res["e_c"])
    
    r = pd.DataFrame({"DA":np.zeros(len(res)),
           "FCRN":np.zeros(len(res)),
           "FCRD_U":np.zeros(len(res)),
           "FCRD_D":np.zeros(len(res)),
           "aFRR_U":np.zeros(len(res)),
           "aFRR_D":np.zeros(len(res)),
           "mFRR_U":np.zeros(len(res)),
           "H":np.zeros(len(res)),
           "Act":np.zeros(len(res)),
           "Recovery":np.zeros(len(res)),
           "Tariffs":np.zeros(len(res))
           },index=res["2023"].index)
    #all trading and capacity pure revenue (no tariffs)
    for k in r.columns[0:8]:
        r[k] = res["p_"+k] * yrP[k]
    
    #activation revenue
    au= (yrA["aFRR_U"]*res["p_aFRR_U"] + yrA["mFRR_U"]*res["p_mFRR_U"] + yrA["FCRN_U"]*res["p_FCRN"] + yrA["FCRD_U"]*res["p_FCRD_U"])
    ad = (yrA["aFRR_D"]*res["p_aFRR_D"] +  yrA["FCRN_D"]*res["p_FCRN"] + yrA["FCRD_D"]*res["p_FCRD_D"])
    r["Act"][(au-ad)>0] = ((au-ad)*yrP["reg_U"])[(au-ad)>0]
    r["Act"][(au-ad)<=0] = ((au-ad)*yrP["reg_D"])[(au-ad)<=0]
    
    
    #True soe calc
    netA = -(yrA["aFRR_U"]*res["p_aFRR_U"] + yrA["mFRR_U"]*res["p_mFRR_U"] + yrA["FCRN_U"]*res["p_FCRN"] + yrA["FCRD_U"]*res["p_FCRD_U"])*PH + (yrA["aFRR_D"]*res["p_aFRR_D"] +  yrA["FCRN_D"]*res["p_FCRN"] + yrA["FCRD_D"]*res["p_FCRD_D"])*PH
    
    #Real_SOE[0] = res["e_b"][0] + netA[0]
    Real_SOE[0] =  E*0.5 + netA[0] - res["p_DA"][0]*PH - res["p_H"][0]
    for i in range(1,len(Real_SOE)):
        #Real_SOE[i] = Real_SOE[i-1] + netA[i] - res["p_DA"][i]*PH - res["p_H"][i]
        deH[i] = res["e_b"][i] - ( Real_SOE[i-1] + netA[i] - res["p_DA"][i]*PH - res["p_H"][i])
        Real_SOE[i] = Real_SOE[i-1] + netA[i] - res["p_DA"][i]*PH - res["p_H"][i] + deH[i]
    
    r["Recovery"] = deH*yrP["H"]
    netGrid = (res["p_DA"] + au - ad)
    r["Tariffs"] = (netGrid*TC)
    #r["Tariffs"][netGrid>=0] = - (netGrid*TP)[netGrid>=0]
    #cyc = np.sum(abs(np.diff(np.concatenate(([0.5*E],Real_SOE))))/(2*E))
    return Real_SOE,r,deH



