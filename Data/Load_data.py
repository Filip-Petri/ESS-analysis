import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import cvxopt
import os
import time

from data_retriever import *

#%% Data
dates = ["2023","2024-02-01"]

#dateIdx = pd.date_range(start=dates[0], end=dates[1],freq='h',tz="utc")[:-1]

#Importing all (raw) data (indexed by UTC datetime)
DF_DA = DA_data(dates[0],dates[1],"DK2")
DF_bal = balance_data(dates[0],dates[1],"DK2")
DF_bal_dk1 = balance_data(dates[0],dates[1],"DK1")
DF_aFRR = aFRR_data(dates[0],dates[1],"DK2")
DF_aFRR_act = aFRR_act_data(dates[0],dates[1],"DK1")
DF_aFRR_act2 = aFRR_act_data(dates[0],dates[1],"DK2")
DF_mFRR = mFRR_data(dates[0],dates[1],"DK2")

DF_mFRR_DK1 = mFRR_data(dates[0],dates[1],"DK1")
DF_bal_DK1 = balance_data(dates[0],dates[1],"DK1")

DF_FCRN,DF_FCRD_u,DF_FCRD_d = FCR_data(dates[0],dates[1],"DK2","total")

#DF_FCRN,DF_FCRD_u,DF_FCRD_d = FCR_data("2020","2024-02-01","DK2","total")

#DF_FCRN,DF_FCRD_d,DF_FCRD_u = FCR_data("2022","2023-01-01","DK2","total")

aa = (((DF_bal_DK1["mFRRUpActBal"] + DF_bal["mFRRUpActBal"])/(DF_mFRR_DK1["mFRR_UpPurchased"] + DF_mFRR["mFRR_UpPurchased"])))
aa[aa>1] = 1


#Filling missing values (based on UTC time)
dfs = {"DA":DF_DA,
       "bal":DF_bal,
       "aFRR":DF_aFRR,
       "aFRR_act":DF_aFRR_act,
       "mFRR":DF_mFRR,
       "FCRN":DF_FCRN,
       "FCRDD":DF_FCRD_d,
       "FCRDU":DF_FCRD_u}

for a in dfs.keys():
    #missingDates = pd.date_range(start=(pd.to_datetime(dates[0])-pd.Timedelta("1h")), end=pd.to_datetime(dates[1])-pd.Timedelta("2h"),freq="1h").difference(dfs[a].index)
    missingDates = pd.date_range(start=(dfs[a].index[0]), end=pd.to_datetime(dates[1])-pd.Timedelta("2h"),freq="1h").difference(dfs[a].index)

    #if (len(missingDates) > 0):
    dfs[a] = dfs[a].resample("h").mean()
    dfs[a] = dfs[a].interpolate()
    
    #set index back to danish time:
    #dfs[a] = dfs[a].set_index(dateIdx)
    #print("Dataset "+a+" in period (UTC): "+str(dfs[a].index[0])+" to "+str(dfs[a].index[-1]))
    
    print("interpolated values for: "+a+" for the following dates (UTC):")
    print(missingDates)
    print()
    
P = {"DA":      dfs["DA"]["SpotPriceEUR"],
       "IMB": dfs["bal"]["ImbalancePriceEUR"],
       "reg_U": dfs["bal"]["BalancingPowerPriceUpEUR"],
       "reg_D": dfs["bal"]["BalancingPowerPriceDownEUR"],
       "FCRN":  dfs["FCRN"]["PriceTotalEUR"],
       "aFRR_U":dfs["aFRR"]['aFRR_UpCapPriceEUR'],
       "aFRR_D":dfs["aFRR"]['aFRR_DownCapPriceEUR'],
       "mFRR_U":dfs["mFRR"]["mFRR_UpPriceEUR"],
       "FCRD_U":dfs["FCRDU"]["PriceTotalEUR"],
       "FCRD_D":dfs["FCRDD"]["PriceTotalEUR"]}

P = pd.DataFrame(P)
#FCR frequency data
freqData = pd.read_csv(r"C:\Users\filip\OneDrive\Documents\DTU_kandidat\Twig_energy_project\Data\fingrid\Frequency.csv")
freqData.index = pd.to_datetime(freqData['End time UTC'])
ff = freqData["Frequency - real time data"].resample("min").interpolate()["2023"]
ff.index = ff.index.tz_convert(tz=None)

fn_d = ff.copy()*0
fn_u = ff.copy()*0
fd_d = ff.copy()*0
fd_u = ff.copy()*0

fn_u[ff<49.9] = 1
fn_u[(49.9<ff)*(ff<50)] = -(ff[(49.9<ff)*(ff<50)] - 50)/0.1
fn_d[ff>50.1] = 1
fn_d[(50.1>ff)*(ff>50)] = (ff[(50.1>ff)*(ff>50)] - 50)/0.1

fd_u[ff < 49.5] = 1
fd_u[(49.5<ff)*(ff<49.9)] = -(ff[(49.5<ff)*(ff<49.9)] - 49.9)/0.4

fd_d[ff > 50.5] = 1
fd_d[(50.5>ff)*(ff>50.1)] = (ff[(50.5>ff)*(ff>50.1)] - 50.1) /0.4


#maybe do fcr in 49.8-50.2
fcr_u = ff.copy()*0
fcr_d = ff.copy()*0
fcr_u[ff<49.8] = 1
fcr_u[(49.8<ff)*(ff<50)] = -(ff[(49.8<ff)*(ff<50)] - 50)/0.2
fcr_d[ff>50.2] = 1
fcr_d[(50.2>ff)*(ff>50)] = (ff[(50.2>ff)*(ff>50)] - 50)/0.2

fcr_u = fcr_u.resample("h").sum()/60
fcr_d = fcr_d.resample("h").sum()/60

#dfs["mFRR"]["mFRR_UpPurchased"]
#All activation    
Act = pd.DataFrame({"mFRR_U":(dfs["bal"]["mFRRUpActBal"]/max(dfs["bal"]["mFRRUpActBal"])),
                   "aFRR_U":dfs["aFRR_act"]["aFRR_UpActivated"],
                   "aFRR_D":dfs["aFRR_act"]["aFRR_DownActivated"],
                   "FCRN_D":fn_d.resample("h").sum()/60,
                   "FCRN_U":fn_u.resample("h").sum()/60,
                   "FCRD_D":fd_d.resample("h").sum()/60,
                   "FCRD_U":fd_u.resample("h").sum()/60
                   })["2023"]
#dk1 afrr act purchase
#https://transparency.entsoe.eu/balancing/r2/activationAndActivatedBalancingReserves/show?name=&defaultValue=false&viewType=TABLE&areaType=MBA&atch=false&dateTime.dateTime=21.12.2023+00:00|UTC|DAYTIMERANGE&dateTime.endDateTime=21.12.2023+00:00|UTC|DAYTIMERANGE&reserveType.values=A96&marketArea.values=CTY|10Y1001A1001A65H!MBA|10YDK-1--------W&dateTime.timezone=UTC&dateTime.timezone_input=UTC 


Act["aFRR_U"][:"2023-12-21 22"] = Act["aFRR_U"][:"2023-12-21 22"]/100
Act["aFRR_U"]["2023-12-21 23":] = Act["aFRR_U"]["2023-12-21 23":]/110
Act["aFRR_D"][:"2023-12-21 22"] = Act["aFRR_D"][:"2023-12-21 22"]/100
Act["aFRR_D"]["2023-12-21 23":] = Act["aFRR_D"]["2023-12-21 23":]/110


# "FCRN_D":(ff>50.01).resample("h").sum()/60,
# "FCRN_U":(ff<49.99).resample("h").sum()/60,
# "FCRD_D":(ff>50.1).resample("h").sum()/60,
# "FCRD_U":(ff<49.9).resample("h").sum()/60




#%% mFRR data scaling

#mFRR activation is not realistic (for a common range, see appendix of https://en.energinet.dk/media/gieparrh/outlook-for-ancillary-services-2023-2040.pdf)
#Using the 2022 data from the figure the mean activation is approximately:
#(20/44*0.05) + (8/44 * 0.15) + (9/44 * 0.3) +(4/44*0.5)+(3/44*0.8) #=21.114%

#The dk2 activation data is considered a reasonable indicator of when activation happens
#NOTE that this is not a real activation curve - but an attemt to simulate the dynamics
#mFRR is individual for plants, and dk2 activation can also include other price zones


#The data is sampled on a daily mean to avoid the many hours of inactivity
d_mean = dfs["bal"]["mFRRUpActBal"].resample("d").mean()

#an hourly interpolation is made over the daily mean
#Since the total purchased mFRR amount is known, a fraction can be calculated:
itp_frac = (d_mean.resample("h").interpolate()/dfs["mFRR"]["mFRR_UpPurchased"])["2023"]

#since the mean of the data is 2.78% activation, it is scaled to correspond with Energinets figure
scaled_mFRR = itp_frac*(21.1/2.78)

#As the scaling makes the activation exceed 100% in a few hours, these are set to 100%
scaled_mFRR[scaled_mFRR>1] = 1

#The figure can now be examined
plt.hist(scaled_mFRR*10,bins=10,density=True,edgecolor="white")
plt.ylabel("Fraction of hours in the year")
plt.xticks([0,1,2,3,4,5,6,7,8,9,10],[0,10,20,30,40,50,60,70,80,90,100])
plt.xlabel("Activation [%]")
np.mean(scaled_mFRR)

#The data is reasonably close to the figure of Energinet.
Act["mFRR_U"] = scaled_mFRR

#%% German Data
f1 = r"C:\Users\filip\OneDrive\Documents\DTU_kandidat\Twig_energy_project\Data\European_FCR.xlsx"
gp = pd.read_excel(f1)[['TENDER_NUMBER','GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]']]
#https://www.regelleistung.net/apps/datacenter/tenders/?productTypes=SRL,MRL,PRL&markets=BALANCING_CAPACITY,BALANCING_ENERGY&date=2023-01-01&tenderTab=PRL$CAPACITY$1
gp = gp[gp["TENDER_NUMBER"]==1]
GP23 = pd.DataFrame({"FCR":((gp['GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]']).replace("-","0")).astype(float)})
GP23.index=pd.date_range("2023","2023-12-31 23",freq="4h")
GP23 = GP23.resample("h").ffill()

GP23 = pd.concat([GP23, pd.DataFrame({"FCR":np.ones(3)*GP23["FCR"]["2023-12-31 20"]},index=pd.date_range("2023-12-31 21","2023-12-31 23",freq="h"))] )



#%% Plots and histograms
AAct = Act.copy()
AAct["FCR_D (DE)"] = fcr_d
AAct["FCR_U (DE)"] = fcr_u

fig, ax = plt.subplots(3,3)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
i,j=0,0
for n in AAct.columns:
    ax[i,j].set_title(n)
    ax[i,j].hist(AAct[n],edgecolor="white")
    i+=1
    if i ==3:
        i=0
        j+=1


PPrice = P["2023"].copy()
PPrice["FCR"] = GP23["FCR"]

fig, ax = plt.subplots(3,3)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
i,j=0,0
for n in PPrice.columns[[0,1,4,5,6,7,8,9]]:
    ax[i,j].set_title(n)
    ax[i,j].set_xlabel("Price")
    ax[i,j].set_xlabel("Total hours")
    ax[i,j].hist(PPrice[n],edgecolor="white")
    i+=1
    if i ==3:
        i=0
        j+=1
ax[2,2].set_visible(False)

#description prices
priP = pd.DataFrame(np.round(PPrice[:].describe(),1)).transpose()
priP.to_clipboard()

#description activation
priA = pd.DataFrame(np.round(AAct[:].describe(),2)).transpose()
priA.to_clipboard()

#description frequency
priF = freqData["Frequency - real time data"]["2023"]
freqData["Frequency - real time data"]["2023"].describe()

