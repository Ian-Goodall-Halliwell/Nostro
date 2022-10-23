
import yaml
import os
from qlib.utils import init_instance_by_config
import pandas as pd
import dateparser
import pickle as pkl
from qlib import init
from qlib.config import REG_CRY
from qlib.data.dataset.handler import DataHandlerLP
import lightgbm as lgb
import numpy as np
import gc
import optuna
from datetime import datetime,timedelta
import pytz
import numpy as np
import d_e_model
import queue
import sys
import threading
from sklearn.preprocessing import RobustScaler
def run(ver, newver=False, labeln=None,sm=True):
    if sm == True:
        segs = {"train":[pd.Timestamp(dateparser.parse("2022-05-01 00:00:00")),pd.Timestamp(dateparser.parse("2022-05-25 00:00:00"))],
                                "valid":[pd.Timestamp(dateparser.parse("2022-05-25 00:05:00")),pd.Timestamp(dateparser.parse("2022-05-28 00:00:00"))],
                                "test":[pd.Timestamp(dateparser.parse("2022-05-28 00:05:00")),pd.Timestamp(dateparser.parse("2022-05-30 12:00:00"))],
                                "all":[pd.Timestamp(dateparser.parse("2022-05-01 00:00:00")),pd.Timestamp(dateparser.parse("2022-05-30 12:00:00"))]
        }
    else:
        segs = {"train":[pd.Timestamp(dateparser.parse("2022-01-01 00:00:00")),pd.Timestamp(dateparser.parse("2022-04-01 00:00:00"))],
                            "valid":[pd.Timestamp(dateparser.parse("2022-04-01 00:05:00")),pd.Timestamp(dateparser.parse("2022-05-01 00:00:00"))],
                            "test":[pd.Timestamp(dateparser.parse("2022-05-01 00:05:00")),pd.Timestamp(dateparser.parse("2022-05-30 12:00:00"))],
                                "all":[pd.Timestamp(dateparser.parse("2022-01-01 00:00:00")),pd.Timestamp(dateparser.parse("2022-05-30 12:00:00"))]
                                }
   
            
    with open("C:/Users/Ian/Documents/Financial testing/model/model_config.yaml") as fil:
        cf = yaml.safe_load(fil)
    if os.path.exists("F:/datasets/handlertemp_{}.pkl".format(ver)):
        with open("F:/datasets/handlertemp_{}.pkl".format(ver),"rb") as f:
            handler = pkl.load(f)
            
    else:    
        if sm == True:
            cf['task']['dataset']['kwargs']['segments']['train'][0] = pd.Timestamp(dateparser.parse("2022-05-01 00:00:00"))
            cf['task']['dataset']['kwargs']['segments']['train'][1] = pd.Timestamp(dateparser.parse("2022-05-25 00:00:00"))

            cf['task']['dataset']['kwargs']['segments']['valid'][0] = pd.Timestamp(dateparser.parse("2022-05-25 00:05:00"))
            cf['task']['dataset']['kwargs']['segments']['valid'][1] = pd.Timestamp(dateparser.parse("2022-05-28 00:00:00"))

            cf['task']['dataset']['kwargs']['segments']['test'][0] = pd.Timestamp(dateparser.parse("2022-05-28 00:05:00"))
            cf['task']['dataset']['kwargs']['segments']['test'][1] = pd.Timestamp(dateparser.parse("2022-05-30 12:00:00"))
            cf['task']['dataset']['kwargs']['segments'].update({"all":[pd.Timestamp(dateparser.parse("2022-05-01 00:00:00")),pd.Timestamp(dateparser.parse("2022-05-30 12:00:00"))]})
        else:        
            cf['task']['dataset']['kwargs']['segments']['train'][0] = pd.Timestamp(dateparser.parse("2022-01-01 00:00:00"))
            cf['task']['dataset']['kwargs']['segments']['train'][1] = pd.Timestamp(dateparser.parse("2022-04-01 00:00:00"))

            cf['task']['dataset']['kwargs']['segments']['valid'][0] = pd.Timestamp(dateparser.parse("2022-04-01 00:05:00"))
            cf['task']['dataset']['kwargs']['segments']['valid'][1] = pd.Timestamp(dateparser.parse("2022-05-01 00:00:00"))

            cf['task']['dataset']['kwargs']['segments']['test'][0] = pd.Timestamp(dateparser.parse("2022-05-01 00:05:00"))
            cf['task']['dataset']['kwargs']['segments']['test'][1] = pd.Timestamp(dateparser.parse("2022-05-30 12:00:00"))

            cf['task']['dataset']['kwargs']['segments'].update({"all":[pd.Timestamp(dateparser.parse("2022-01-01 00:00:00")),pd.Timestamp(dateparser.parse("2022-05-30 12:00:00"))]})
            
        
     

        

        
        
      

        
        if ver == "train":
            cf['data_handler_config']['start_time'] = cf['task']['dataset']['kwargs']['segments']['train'][0]
            cf['data_handler_config']['fit_start_time'] = cf['task']['dataset']['kwargs']['segments']['train'][0]
            cf['data_handler_config']['fit_end_time'] = cf['task']['dataset']['kwargs']['segments']['train'][1]
            cf['data_handler_config']['end_time'] = cf['task']['dataset']['kwargs']['segments']['train'][1]
        if ver == "valid":
            cf['data_handler_config']['start_time'] = cf['task']['dataset']['kwargs']['segments']['valid'][0]
            cf['data_handler_config']['fit_start_time'] = cf['task']['dataset']['kwargs']['segments']['valid'][0]
            cf['data_handler_config']['fit_end_time'] = cf['task']['dataset']['kwargs']['segments']['valid'][0]
            cf['data_handler_config']['end_time'] = cf['task']['dataset']['kwargs']['segments']['valid'][1]
            
        if ver == "test":
            cf['data_handler_config']['start_time'] = cf['task']['dataset']['kwargs']['segments']['test'][0]
            cf['data_handler_config']['fit_start_time'] = cf['task']['dataset']['kwargs']['segments']['test'][0]
            cf['data_handler_config']['fit_end_time'] = cf['task']['dataset']['kwargs']['segments']['test'][0]
            cf['data_handler_config']['end_time'] = cf['task']['dataset']['kwargs']['segments']['test'][1]
        if ver == "all":
            cf['data_handler_config']['start_time'] = cf['task']['dataset']['kwargs']['segments']['train'][0]
            cf['data_handler_config']['fit_start_time'] = cf['task']['dataset']['kwargs']['segments']['train'][0]
            cf['data_handler_config']['fit_end_time'] = cf['task']['dataset']['kwargs']['segments']['train'][1]
            cf['data_handler_config']['end_time'] = cf['task']['dataset']['kwargs']['segments']['test'][1]
        
        if os.path.exists("F:/datasets/handlertemp_{}.pkl".format(ver)):
            with open("F:/datasets/handlertemp_{}.pkl".format(ver),"rb") as f:
                handler = pkl.load(f)
        else:
            cf['task']['dataset']['kwargs']['handler']['kwargs'].update({"drop_raw":False})
            handler = init_instance_by_config(cf['task']['dataset']['kwargs']['handler'])
            
            print("Creating dataset")    
            cf['task']['dataset']['kwargs']['handler'] = handler
            handler.to_pickle("F:/datasets/handlertemp_{}.pkl".format(ver),dump_all=True)
    if isinstance(labeln, pd.Series): 
        labeln = labeln.reindex_like(handler._learn["label"])
        del handler._learn
        del handler._infer
        
        
        
        handler._data = handler._data.drop(labels="label",axis=1)
        for file in os.listdir("C:/Users/Ian/Documents/Financial testing/model/models"):
            fp = os.path.join("C:/Users/Ian/Documents/Financial testing/model/models",file)
            with open(fp,'rb') as f:
                labelv = pkl.load(f)
                labelv = labelv.reindex(handler._data["feature"].index)
                labelv = labelv.drop(labels="label",axis=1)
                namedict = {}
                for col in labelv.columns:
                    namedict.update({col:file.split(".")[0].split("_")[1] +"_"+ col})
                    
                labelv.rename(columns = namedict,inplace = True)    
                tmpt = handler._data["feature"].join(labelv)
                
                del handler._data
                
                
                handler._data={"feature":tmpt}
            print('e')
        #
        
        handler._data.update({"label":labeln})
        
        ind = handler._data['feature'].columns
        btc =labeln.xs("BTCUSDT",axis=0,level=1)
        idx = [("feature",x) for x in ind]
        idx.append(("label","label"))
        indexc = pd.MultiIndex.from_tuples(idx)
        
        feat = handler._data["feature"]
        label = handler._data["label"]
        handler._data = pd.DataFrame(feat,columns=indexc)
        handler._data['feature'] = feat
        handler._data["label"] = labeln
        handler.fit_process_data()
        df2 = RobustScaler().fit_transform(labeln.values.reshape(-1, 1))
        dfl = pd.Series(np.squeeze(df2),index=labeln.index)
        
        handler._learn["label"]=dfl
        handler._infer["label"]=dfl
        handler._data["label"]=dfl
        
        dfx = dfl.xs("1INCHUSDT",axis=0,level=1)
        print("e")
    if newver==True:
        
        return handler
    cf['task']['dataset']['kwargs']['handler'] = handler
    cf['task']['dataset']['kwargs']['segments'] = segs
    
    dataset = init_instance_by_config(cf['task']['dataset'])
    
    features = dataset.prepare(
                [ver], col_set=["feature"], data_key=DataHandlerLP.DK_R
            )
    
    labels = dataset.prepare(
                [ver], col_set=["label"], data_key=DataHandlerLP.DK_R
            )
    
    return features, labels      
    
def label_algorithm(all_labels,tradefee):
    new_labels = pd.Series().reindex_like(all_labels) 
    length = new_labels.axes[0] 
    for timestamp in length:
        current_label = all_labels[timestamp]
        if current_label < -tradefee*2:
            new_labels[timestamp] = current_label #- tradefee*2
        else:
            run_score = current_label
            newtimestamp = timestamp 
            breakcond = False
            while True:
                if breakcond == True:
                    new_labels[timestamp] = run_score #- tradefee*2 
                    break
                newtimestamp = newtimestamp + timedelta(minutes=5) 
                try:
                    next_label = all_labels[newtimestamp]
                except:
                    next_label = -0.1
                if next_label >= 0:
                    run_score += next_label
                elif next_label < 0:
                    #breakcond = True
                    #break
                    
                    if next_label <= -tradefee:
                        new_labels[timestamp] = run_score #- tradefee*2
                        break
                    elif next_label > -tradefee:
                        newesttimestamp = newtimestamp
                        loss_score = next_label
                        cnt =0
                        while True:
                            cnt+=1
                            if cnt > 72:
                                breakcond =True
                                break
                            try:
                                newesttimestamp = newesttimestamp + timedelta(minutes=5) 
                                newestlabel = all_labels[newesttimestamp]
                            except:
                                breakcond = True
                                break
                            loss_score += newestlabel
                            if loss_score <= -tradefee:
                                breakcond = True
                                break
                            elif loss_score > 0:
                                run_score += next_label
                                break
                            else:
                                continue
    return new_labels
                                
                            
            
            
    
    
def multifunc(datadict,q,resdict):
    try:
        i = q.get()
    except:
        i = q
    try:
        all_labels = datadict.xs(i,axis=0,level=1)["LABEL0"] *100
        
    except:
        
        return
    
    
    
    
    tradefee = 0.75
    
    new_labels = label_algorithm(all_labels,tradefee)
    
        
    #qd.put({i:new_labels})        
    resdict.update({i:new_labels})
    print("done {}".format(i))
    print(len(new_labels))
    ppd = gc.collect()
      
def runnoset(trial, retlabel=False,funked=False,makebinaries=True,sm=True):
    
    if os.path.exists("C:/Users/Ian/Documents/Financial testing/model/{}set_test.pkl".format(trial)):
        with open("C:/Users/Ian/Documents/Financial testing/model/{}set_test.pkl".format(trial),'rb') as fr:
            datadict = pkl.load(fr)
            pass
    
    else:
        valid_feats,valid_labels = run(trial,sm=sm)
        try:
            valid_feats,valid_labels = valid_feats[0],valid_labels[0] 
        except:  
            valid_feats,valid_labels = valid_feats,valid_labels
        
        datadict = pd.concat([valid_feats['feature'],valid_labels['label']],axis=1)  
       
        
        
    
        with open("C:/Users/Ian/Documents/Financial testing/model/{}set_test.pkl".format(trial),'wb') as f:
            pkl.dump(datadict,f)
    
        
    if os.path.exists("C:/Users/Ian/Documents/Financial testing/model/{}bckup.pkl".format(trial)):     
        with open("C:/Users/Ian/Documents/Financial testing/model/{}bckup.pkl".format(trial),'rb') as f:
            resdict = pkl.load(f) 
            
            print("e")
            
    else:
        
        length = datadict.xs("BTCUSDT",axis=0,level=1).axes[0]
        
        resdict = {"datetime":length.values}
        q = queue.SimpleQueue()
        qd = queue.SimpleQueue()
        e = list(datadict.axes[0].levels[1].values)
        for a in e:
            multifunc(datadict,a,resdict)
        # for qi in e:
        #     q.put(qi)
        # size = q.qsize()
        
        # for r in range(size//8):
        #     thrdlist = []
        #     for thrd in range(16):
        #         thrdlist.append(threading.Thread(target=multifunc, args=(datadict,q,resdict,)))
        #     for thread in thrdlist:
        #         thread.start()
        #     for thread in thrdlist:
        #         thread.join()
            
        
        with open("C:/Users/Ian/Documents/Financial testing/model/{}bckup.pkl".format(trial),'wb') as f:
            pkl.dump(resdict,f) 
        with open("C:/Users/Ian/Documents/Financial testing/model/{}set_test.pkl".format(trial),'wb') as fr:
            pkl.dump(datadict,fr)
         
    
    if os.path.exists("model/{}set.pkl".format(trial)):
        with open("model/{}set.pkl".format(trial), "rb") as f:
            valid,df = pkl.load(f) 
            #ox = df.xs("BTCUSDT",axis=0,level=1)
            #xtr = ox["LABEL0"]
            print('e')
            
            
    else:
        idx = resdict.pop("datetime")
        df = pd.DataFrame(resdict, index=idx)
        with open("C:/Users/Ian/Documents/Financial testing/model/allbckup.pkl",'rb') as f:
            alld = pkl.load(f) 
            idx = alld.pop("datetime")
            alld = pd.DataFrame(alld, index=idx)
        df = alld.reindex_like(df)
        df = df.stack()
        with open("C:/Users/Ian/Documents/Financial testing/model/{}set_test.pkl".format(trial),'rb') as f:
            valid = pkl.load(f)
        with open("model/{}set.pkl".format(trial), "wb") as f:
            valid = valid.reindex(df.index)
            pkl.dump([valid,df],f)  
    
    if makebinaries == True:
        
        if funked == True:
            version = "preds"
        else:
            version = "nopreds"
        
        
        if trial == "valid":
            if funked == True:
                with open("model/yearlypreds_valid.pkl", "rb") as f:
                    predlabels = pkl.load(f)
                lablis = []
                for a in valid.columns:
                    if "LABEL" in a:
                        lablis.append(a)
                        
                
                valid.drop(lablis, axis=1, inplace=True)
                for a in lablis:
                    sp = a.split("LABEL")[1]
                    if len(sp) == 1:
                        sp = "0" + sp
                        a = "LABEL" + sp
                    valid[a] = predlabels[a]
                print("")
            valid = valid.to_numpy(dtype="float32")
            df = df.to_numpy(dtype="float32")
            old = lgb.Dataset("model/train_labels_{}.bin".format(version),params={"max_bin":15},free_raw_data=False)
            new = lgb.Dataset(np.squeeze(valid), label=np.squeeze(df),params={"max_bin":15},free_raw_data=False,reference=old)
            new.save_binary("model/{}_labels_{}.bin".format(trial,version))
        elif trial == "train":
            if funked == True:
                with open("model/yearlypreds_train.pkl", "rb") as f:
                    predlabels = pkl.load(f)
                lablis = []
                #predlabelsnp = predlabels.to_numpy()
                for a in valid.columns:
                    if "LABEL" in a:
                        lablis.append(a)
                        
                
                valid.drop(lablis, axis=1, inplace=True)
                for a in lablis:
                    sp = a.split("LABEL")[1]
                    if len(sp) == 1:
                        sp = "0" + sp
                        a = "LABEL" + sp
                    valid[a] = predlabels[a]
                print("")
            valid = valid.to_numpy(dtype="float32")
            df = df.to_numpy(dtype="float32")
            new = lgb.Dataset(np.squeeze(valid), label=np.squeeze(df),params={"max_bin":15},free_raw_data=False)
            new.save_binary("model/{}_labels_{}.bin".format(trial,version))
        else:
            pass
    if retlabel == True:
        return df
    
import random   

def long_short_eval(pred,label):
    
    precret = calc_long_short_prec_concise(pred,label.label)
    return "LSLoss", precret, False

def long_short_loss(pred,label):
    
   
    grad, hess = calc_long_short_prec_fd_sd(pred,label.label)
    return grad,hess

def calc_long_short_prec_concise(pred: pd.Series, label: pd.Series):

    df = pd.DataFrame({"label":label,"pred":pred})
    dfoutput = np.where(df["label"] - df["pred"] >= 0,((df["label"] - df["pred"])**2),(((df["label"] - df["pred"])*2)**2))
    dfsum = dfoutput.sum()
    avg = np.sqrt(dfsum/len(dfoutput))

    return avg

def calc_long_short_prec(
    pred: pd.Series, label: pd.Series):

   
    #df = pd.DataFrame({"label":label,"pred":pred})
    dfoutput = np.where(label - pred >= 0,(label - pred)**2,(((label - pred)*2)**2))
    dfsum = dfoutput.sum()
    avg = np.sqrt(dfsum/len(dfoutput))

    return avg

def calc_long_short_prec_fd_sd(pred: pd.Series, label: pd.Series):

    
    #df = pd.DataFrame({"label":label,"pred":pred},).squeeze().reset_index()
    dfoutput1d = np.where(label - pred >= 0,(-(label - pred)*2),(-((label - pred)*4)))
    
    dfoutput2d = np.where(label - pred >= 0,(np.full_like(label, 2)),(np.full_like(label, 4)))
    

    return dfoutput1d, dfoutput2d
def objective(trial):
    timed = datetime.now()
    
    task = {
        
            
                "boosting":trial.suggest_categorical("boosting",["goss","gbdt","dart"]),
                
                
                "feature_fraction_seed":random.randint(0,100000),
                "linear_lambda":trial.suggest_loguniform("linear_lambda", 1e-8, 1e4),
                
                
                "feature_fraction_bynode":trial.suggest_uniform("feature_fraction_bynode", 0.4, 1.0),
                "min_sum_hessian_in_leaf":trial.suggest_loguniform("min_sum_hessian_in_leaf", 1e-8, 1e4),
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                "learning_rate": trial.suggest_uniform("learning_rate", 0, 0.4),
                "subsample": trial.suggest_uniform("subsample", 0.4, 1),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e5),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e5),
                "max_depth": trial.suggest_int("max_depth", 1, 30),
                "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
                
                
                "verbosity":-1,
                "max_bin" : 15,
                "nthread": 16,
                
                "device":"gpu",
                "gpu_use_dp":True,
                "early_stopping_rounds":20
            }
        
    
    if task["boosting"] == "goss":
        task.update({
            "top_rate" : trial.suggest_uniform("top_rate", 0,0.5),
            "other_rate" : trial.suggest_uniform("other_rate", 0,0.5)
        })
    if task["boosting"] != "goss":
        task.update({
            "bagging_seed":random.randint(0,100000),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            
            
            })
    if task["boosting"] != "dart":
        task.update({
            "early_stopping_rounds":20
        })
    if task["boosting"] == "dart":
        task.update({
            "drop_rate":trial.suggest_uniform("drop_rate", 0, 1),
            "max_drop": trial.suggest_int("max_drop", 0, 200),
            "skip_drop":trial.suggest_uniform("skip_drop", 0, 1.0),
            "drop_seed":random.randint(0,100000),
        })
    task.update({"objective":long_short_loss})
    train = lgb.Dataset("model/train_labels_preds.bin")
    valid = lgb.Dataset("model/valid_labels_preds.bin")
    model = lgb.train(params=task,train_set=train,valid_sets=[train,valid],valid_names=["train","valid"],num_boost_round=50,feval=long_short_eval)
    
    runtime = datetime.now() - timed
    print("Training took... {}".format(runtime))
    return model.best_score['valid']['LSLoss']
def  trainmodel(trial,optim=False):
    train = lgb.Dataset("model/train_labels_{}.bin".format(trial))
    valid = lgb.Dataset("model/valid_labels_{}.bin".format(trial))
    study_name="LGBM_Trading_stratetgy_{}".format(trial)
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,storage=storage_name,load_if_exists=True)
    if optim == True:
        study.optimize(objective,n_trials=100, n_jobs=1)
    bestparams = study.best_params
    
    bestparams.update({
                "verbosity":-1,
                "max_bin" : 15,
                "nthread": 16,
                
                "device":"gpu",
                "gpu_use_dp":True,
                #"early_stopping_rounds":20
                })
    bestparams.update({"objective":long_short_loss})
    model = lgb.train(params=bestparams,train_set=train,valid_sets=[train,valid],valid_names=["train","valid"],num_boost_round=50,feval=long_short_eval)
    model.save_model("model/decisionmodel_{}.bin".format(trial))
    print("e")
def trainde():
    study_name="LGBM_Trading_stratetgy"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,storage=storage_name,load_if_exists=True)
    #study.optimize(objective,n_trials=5, n_jobs=1)
    bestparams = study.best_params
    
    bestparams.update({
                "verbosity":-1,
                "max_bin" : 15,
                "nthread": 16,
                
                "device":"gpu",
                "gpu_use_dp":True,
                #"early_stopping_rounds":20
                })
    #bestparams.update({"objective":long_short_loss})
    model = d_e_model.DEnsembleModel(
            base_model= "gbm",
            loss= "mse",
            num_models= 6,
            enable_sr= True,
            enable_fs= True,
            alpha1= 1,
            alpha2= 1,
            bins_sr= 10,
            bins_fs= 5,
            decay= 0.5,
            sample_ratios=[0.8,0.7,0.6,0.5,0.4],
            sub_weights=[1,0.2,0.2,0.2,0.2,0.2],
            epochs= 28,
            kwargs=bestparams)
    model.fit()
    with open('model/demod.pkl',"wb") as f:
        pkl.dump(model,f)
def pred(trial,ver,funked=True):
    with open("model/{}set.pkl".format(trial), "rb") as f:
            valid,df = pkl.load(f) 
    
   
        
    if funked == True:
        with open("model/yearlypreds_{}.pkl".format(trial),"rb") as f:
            predlabels = pkl.load(f)
            predlabels = predlabels.reindex(valid.index)
        lablis = []
        for a in valid.columns:
            if "LABEL" in a:
                lablis.append(a)
                
        #labels = features[['LABEL{}'.format(meta_input)]]
        valid.drop(lablis, axis=1, inplace=True)
        for a in lablis:
            sp = a.split("LABEL")[1]
            if len(sp) == 1:
                sp = "0" + sp
                a = "LABEL" + sp
            valid[a] = predlabels[a]
        print("")
    model = lgb.Booster(model_file="model/decisionmodel_{}.bin".format(ver))
    preds = pd.Series(model.predict(valid.values), index=valid.index)
    btc = preds.xs("BTCUSDT",axis=0,level=1)
    btctrue = df.xs("BTCUSDT",axis=0,level=1)
    with open("model/backtestpreds.pkl", "wb") as f:
        pkl.dump(preds, f)
    print('e')
def pl():
    # with open("F:\linear_alpha158_handler_horizon144.pkl",'rb') as fl:
    #     dh = pkl.load(fl)
    # f = dh._infer
    # fh = f.xs("ADABUSD",axis=0,level=1)
    # for en,a in enumerate(fh.axes[0]):
    #     if en > 2000:
    #         fff = fh.xs(a).to_dict()
    #         for adf in fff:
    #             ad = fff[adf]
    #             if ad == np.nan:
                    
    #                 print(adf)
    #             if ad == np.inf:
    #                 print(adf)
    #             if ad == -np.inf:
    #                 print(adf)
    with open("model/backtestpreds 0.12.pkl", "rb") as f:
        preds = pkl.load(f)
    cor = np.corrcoef(preds['label'].values,preds['score'].values,rowvar=False)
    print('e')
    
if __name__ == '__main__':
    provider_uri = "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/5m-qlib" 
    init(provider_uri={"5min":provider_uri}, region=REG_CRY)
    #label = runnoset("test", retlabel=True, makebinaries=False)
    # with open ("F:/mlruns/2/c1d7af629875465890cc0f75017c0431/artifacts/task",'rb') as f:
    #     ln = pkl.load(f)
    # with open("F:/mlruns/2/9f31c7214f2840f18511618370ddd5fd/artifacts/task", "rb") as f:
    #     lrd = pkl.load(f)
    # ln['dataset']['kwargs']['handler'] = lrd['dataset']['kwargs']['handler']
    
    # with open ("F:/mlruns/2/c1d7af629875465890cc0f75017c0431/artifacts/task",'wb') as f:
    #     pkl.dump(ln,f)
    with open("F:\dataset.pkl",'rb') as f:
        dataset = pkl.load(f)
    del dataset._data    
    with open("F:\dataset.pkl",'wb') as f:
        pkl.dump(dataset,f)
    pl()
    #runnoset("test",funked=False,makebinaries=False)
    #runnoset("valid",funked=False,makebinaries=False)
    #runnoset("test",funked=False,makebinaries=False)
    #trainmodel("preds",optim=False,makebinaries=False)
    
    #pred("valid","preds")
    #trainde()
    