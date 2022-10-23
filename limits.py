import pickle as pkl
import os
import csv
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import RobustScaler, StandardScaler
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std
def create(n,l=1,h=1,std=True,quant = True,iqh=None,iql =None,keyt=True,vl=None):
    if vl == None:
        vl = "n"
    if keyt == True:
        key = random.randint(1,10000)
    else:
        key = 0
    scaler = None
    try:
        with open('C:/Users/Ian/Documents/Financial testing/model//backtestpreds_{}.pkl'.format(vl), 'rb') as handle:
            preds = pkl.load(handle)["score"]
    except:
        with open('C:/Users/Ian/Documents/Financial testing/model//backtestpreds_{}.pkl'.format(vl), 'rb') as handle:
            preds = pkl.load(handle)
    # idv = preds.index.drop('BTCBUSD',level=1)
    # idvt = idv.drop('ETHBUSD',level=1)
    
    # preds = preds.reindex(idvt)
    #with open("model/testset.pkl",'rb') as hop:
    #    preds = pkl.load(hop)[1]
    #df2 = RobustScaler().fit_transform(preds.values.reshape(-1, 1))
    #preds = pd.Series(np.squeeze(df2),index=preds.index)
    #preds = preds.rank(pct=True)
    
    ay = preds.axes[0].levshape[1]
    a = preds.shape[0]//ay
    a = int(a*n)
    
    
    
    
    
    
    
    if std == True:
        
        
        scaler = RobustScaler()
        
        scaler.fit(preds.values.reshape(-1, 1))
        
        if os.path.exists("C:/Users/Ian/Documents/Financial testing/model/scaler.pkl"):
            os.remove("C:/Users/Ian/Documents/Financial testing/model/scaler.pkl")
        with open("C:/Users/Ian/Documents/Financial testing/model/scaler{}.pkl".format(key), 'wb') as handle:
            pkl.dump(scaler,handle)
        df = pd.DataFrame(scaler.transform(preds.values.reshape(-1, 1))).sort_values(by=0,ascending=False)
        # newdf = []
        
        # newdf = np.where(df.values[:,0] != np.nan,(sum((df.values[:,e]*x) for e,x in enumerate(nmods))/sum(nmods)),0)
        
        # df = pd.DataFrame(newdf).sort_values(by=0,ascending=False)
        if quant == True:
            Q1 = df.quantile(iqh)
            Q3 = df.quantile(iql)
            IQR = Q3 - Q1
            lowr = (Q1 - 1.5 * IQR)
            highr = (Q3 + 1.5 * IQR)
            z_preds = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)].to_numpy()
            delb =  len(df) - len(z_preds)
            
            delv = 0
            bb = int((a*h)+delv)
            b = z_preds[bb]
            if l != None:
                cc = int((-a*l)-delv)
                c = z_preds[cc]
            else:
                c = 0
            
        else:
            z_preds = preds.to_numpy()
            b = z_preds[int(a*h)]
            if l != None:
                c = z_preds[int(-a*l)]
            else:
                c = 0
            highr = 10000
            lowr = -10000
        params = scaler.get_params()
       # z_preds = zs#z_score(pd.DataFrame(preds[0]))

        
    else:
        prd = preds.to_numpy()
        prd = np.sort(prd)[::-1]
        
        b=prd[int(a*h)]
        if l != None:
            c=prd[int(a*l)]
        else:
            c= 0
        highr = 10000
        lowr = -10000
    if os.path.exists("C:/Users/Ian/Documents/Financial testing/stats_utils/stats.csv"):
        os.remove("C:/Users/Ian/Documents/Financial testing/stats_utils/stats.csv")
    with open("C:/Users/Ian/Documents/Financial testing/stats_utils/stats{}.csv".format(key),'w') as f:
        writer = csv.writer(f)
        
        writer.writerow([np.squeeze(b),np.squeeze(c)])
    if os.path.exists("C:/Users/Ian/Documents/Financial testing/stats_utils/hl.csv"):
        os.remove("C:/Users/Ian/Documents/Financial testing/stats_utils/hl.csv")
    with open("C:/Users/Ian/Documents/Financial testing/stats_utils/hl{}.csv".format(key),'w') as f:
        writer = csv.writer(f)
        writer.writerow([highr,lowr]) 
    if keyt == True: 
        return scaler, [highr,lowr],key
    else:
        return scaler, [highr,lowr]
if __name__ == "__main__":
    create(1)
# create(1)
# print('')