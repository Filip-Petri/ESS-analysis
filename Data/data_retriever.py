
import requests
import json
import pandas as pd
import numpy as np


"""ELPRISER API"""
def DA_data(date_start,date_end,price_area):
    #valid formats 
    #dates: "yyyy" or "yyyy-MM-dd" or "yyyy-MM-ddTHH:mm"
    #price_area: "DK1" or ""DK2"
    
    date_start = (pd.to_datetime(date_start)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    date_end = (pd.to_datetime(date_end)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    
    url='https://api.energidataservice.dk/dataset/Elspotprices?start='+date_start+'&end='+date_end+'&filter={"PriceArea":["'+price_area+'"]}'
    data_json =requests.get(url).json()
    
    df = pd.DataFrame.from_records(data_json["records"])
    p = df.set_index(pd.to_datetime(df['HourUTC']))#['SpotPriceEUR']
    return p[::-1]


#da=DA_data("2023","2024","DK2")



#print(pd.date_range(start="2023", end="2024",freq="1h")[0:-1].difference(da.index))


def balance_data(date_start,date_end,price_area):
    #valid formats 
    #dates: "yyyy" or "yyyy-MM-dd" or "yyyy-MM-ddTHH:mm"
    #price_area: "DK1" or ""DK2"
    
    date_start = (pd.to_datetime(date_start)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    date_end = (pd.to_datetime(date_end)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    url='https://api.energidataservice.dk/dataset/RegulatingBalancePowerdata?start='+date_start+'&end='+date_end+'&filter={"PriceArea":["'+price_area+'"]}'
    data_json =requests.get(url).json()
    

    df = pd.DataFrame.from_records(data_json["records"])
    p = df.set_index(pd.to_datetime(df['HourUTC']))#['SpotPriceEUR']
    return p[::-1]

#balance_data("2023","2024","DK2")





def aFRR_data(date_start,date_end,price_area):
    #valid formats 
    #dates: "yyyy" or "yyyy-MM-dd" or "yyyy-MM-ddTHH:mm"
    #price_area: ""DK2"
    
    date_start = (pd.to_datetime(date_start)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    date_end = (pd.to_datetime(date_end)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    url='https://api.energidataservice.dk/dataset/AfrrReservesNordic?start='+date_start+'&end='+date_end+'&filter={"PriceArea":["'+price_area+'"]}'
    data_json =requests.get(url).json()
    
    
    #df.index = df.index.strftime('%Y-%m-%d T%H:%M:%ss')
    
    #data = pd.DataFrame.from_records(data_json["records"])
    #data.index = data["HourUTC"]
    
    
    #df = pd.DataFrame(index=pd.date_range(date_start,date_end,freq="h"),columns=[data.columns])
    #df.loc[pd.to_datetime(data.index)] = data[:]
    
    #p = df.replace([np.nan, np.inf], 0)
    
    df = pd.DataFrame.from_records(data_json["records"])
    p = df.set_index(pd.to_datetime(df['HourUTC']))#['SpotPriceEUR']
    return p#[::-1]
#t_af=aFRR_data("2020","2025","DK2")

def aFRR_act_data(date_start,date_end,price_area):
    #valid formats 
    #dates: "yyyy" or "yyyy-MM-dd" or "yyyy-MM-ddTHH:mm"
    #price_area: "DK1" or ""DK2"
    date_start = (pd.to_datetime(date_start)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    date_end = (pd.to_datetime(date_end)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    url='https://api.energidataservice.dk/dataset/AfrrActivatedAutomatic?start='+date_start+'&end='+date_end+'&filter={"PriceArea":["'+price_area+'"]}'
    data_json =requests.get(url).json()
    
    df = pd.DataFrame.from_records(data_json["records"])
    p = df.set_index(pd.to_datetime(df['HourUTC']))#['SpotPriceEUR']
    return p[::-1]

#aFRR_act_data(date_start,date_end,"DK2")
#check for missing data
#print(pd.date_range(start="2022-12-08 01:00:00", end="2024-01-21").difference(t_af.index))



def mFRR_data(date_start,date_end,price_area):
    #valid formats 
    #dates: "yyyy" or "yyyy-MM-dd" or "yyyy-MM-ddTHH:mm"
    #price_area: "DK1" or ""DK2"
    #note: datasets have missing data (shown as NaN)
    date_start = (pd.to_datetime(date_start)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    date_end = (pd.to_datetime(date_end)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    url1='https://api.energidataservice.dk/dataset/mFRRCapacityMarket?start='+date_start+'&end='+date_end+'&filter={"PriceArea":["'+price_area+'"]}'
    url2='https://api.energidataservice.dk/dataset/MfrrReserves'+price_area+'?start='+date_start+'&end='+date_end
    
    data_json1 =requests.get(url1).json()
    data_json2 =requests.get(url2).json()
    

    df1 = pd.DataFrame.from_records(data_json1["records"])
    df2 = pd.DataFrame.from_records(data_json2["records"])
    
    if df1.empty or df2.empty:
        columns = ['mFRR_DownPurchased', 'mFRR_DownPriceEUR', 'mFRR_UpPurchased',
       'mFRR_DownPriceDKK', 'mFRR_UpPriceEUR', 'mFRR_UpPriceDKK', 'HourDK',
       'HourUTC']
        if df1.empty:
            df = df2[columns]
        elif df2.empty:
            df = df1[columns]
        else:
            print("Error: No data selected")

    else:
        columns = list(set(df1.columns) & set(df2.columns))
        df = pd.concat([df1[columns],df2[columns]])
    
    p = df.set_index(pd.to_datetime(df['HourUTC']))#.resample("h").mean()#['SpotPriceEUR']

    return p[::-1]




def FCR_data(date_start,date_end,price_area,auction_type):
    #valid formats 
    #dates: "yyyy" or "yyyy-MM-dd" or "yyyy-MM-ddTHH:mm"
    #price_area: "DK2" or "SE1" or "SE2" or "SE3" or "SE4"
    #auction_type: "D-1 early" or "D-1 late" or "Total"
    
    date_start = (pd.to_datetime(date_start)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    date_end = (pd.to_datetime(date_end)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    url='https://api.energidataservice.dk/dataset/FcrNdDK2?start='+date_start+'&end='+date_end+'&filter={"PriceArea":["'+price_area+'"],'+'"AuctionType":["'+auction_type+'"]}'
    data_json =requests.get(url).json()
    
    df = pd.DataFrame.from_records(data_json["records"])
    
    
    product_names = df["ProductName"].unique()
    
    dfs = {product: df[df['ProductName'] == product] for product in product_names}
    
    
    df_FCRN = dfs['FCR-N']
    df_FCRD_u = dfs['FCR-D upp']
    df_FCRD_d = dfs['FCR-D ned']

    p_FCRN = df_FCRN.set_index(pd.to_datetime(df_FCRN['HourUTC']))
    p_FCRD_u = df_FCRD_u.set_index(pd.to_datetime(df_FCRD_u['HourUTC']))
    p_FCRD_d = df_FCRD_d.set_index(pd.to_datetime(df_FCRD_d['HourUTC']))
    
    #Returns dataframes FCRN,FCRD_d,FCRD_u
    return p_FCRN[::-1],p_FCRD_u[::-1],p_FCRD_d[::-1]


def FCR_data_dk1(date_start,date_end):
        #valid formats 
    #dates: "yyyy" or "yyyy-MM-dd" or "yyyy-MM-ddTHH:mm"
    #price area: "DK1"
    date_start = (pd.to_datetime(date_start)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    date_end = (pd.to_datetime(date_end)+pd.Timedelta("1h")).strftime('%Y-%m-%dT%H:%M')
    url='https://api.energidataservice.dk/dataset/FcrDK1?start='+date_start+'&end='+date_end
    data_json =requests.get(url).json()
    
    df = pd.DataFrame.from_records(data_json["records"])
    p = df.set_index(pd.to_datetime(df['HourUTC']))#['SpotPriceEUR']
    return p[::-1]

#FCRN,FCRD_d,FCRD_u = FCR_data("2022","2024","DK2","total")



#remember to resample and interpolate/fill
# def patch_and_interpolate(data):
#     df.interpolate()







