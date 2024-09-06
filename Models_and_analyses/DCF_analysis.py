#%% DISCOUNTED CASH FLOW METHOD
import numpy as np
import numpy_financial as npf
import pandas as pd


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


def dcf_de(rev,LT,r_DC,r_T,OPEX,CAPEX,rev2,BT): #yrRev,inflation
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
    
    m = np.logspace(1,0,BT)
    DF["Revenue(Real)"][1:BT+1] = sum(np.sum(rev))*m + sum(np.sum(rev2))*(1-m)
    DF["Revenue(Real)"][BT+1:]= sum(np.sum(rev2))

    DF["Revenue(Nominal)"]=DF["Revenue(Real)"]*inf0      
    DF["OPEX(Real)"][1:] = np.ones(LT)*OPEX
    DF["OPEX(Nominal)"] = DF["OPEX(Real)"]*inf0
    DF["EBITDA"] = DF["OPEX(Nominal)"]+DF["Revenue(Nominal)"]
    DF["CAPEX"][0] = CAPEX*inf0[0]
    DF["Depreciation[%]"][1:] = 1/LT
    DF["Depreciation[EUR"]= DF["Depreciation[%]"]*CAPEX
    DF["EBIT"] = DF["EBITDA"]+DF["Depreciation[EUR"]
    DF["Payable Tax"] = -DF["EBIT"]*r_T
    DF["CF(Nominal)"] = DF["Revenue(Nominal)"] + DF["OPEX(Nominal)"] + DF["CAPEX"] + DF["Payable Tax"]
    DF["DC index"] = 1/(1+r_DC)**Operation_year
    DF["DCF"] = DF["CF(Nominal)"]*DF["DC index"]
    
    NPV = np.sum(DF["DCF"])
    #print(DF["CF(Nominal)"])
    IRR = npf.irr(DF["CF(Nominal)"])
    

    return NPV,IRR,DF


#m = np.linspace(0,10,10)
#i=1



#t1,t2,t3 = dcf_de(rr,15,0.08,0.22,-9219.1*E*C,np.multiply(-394,E*10**3) - 368.76*10**3*E*C,rr*0.5,10)
#t1,t2,t3 = dcf(rr,15,0.08,0.22,-9219.1*E*C,np.multiply(-394,E*10**3) - 368.76*10**3*E*C)



