#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
import gc
import numpy as np
from qlib.config import REG_CRY
import mrmr
import matplotlib as mp
import plotly.express as px
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    roc_auc_score,
    fbeta_score,
)
import lightgbm as lgb
import qlib.contrib.report.analysis_position as qcr
from qlib.utils import init_instance_by_config
from sklearn.preprocessing import MinMaxScaler
from qlib.data.dataset import TSDatasetH
from pickle import UnpicklingError
from optuna.trial import TrialState
import yaml
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategyNew
from qlib.backtest.exchange import Exchange
from sklearn.metrics import classification_report
from qlib.backtest.utils import LevelInfrastructure, CommonInfrastructure
from qlib.backtest import create_account_instance
from datetime import datetime, timedelta, timezone
import pandas as pd

import dateparser
from qlib.data.dataset import DataHandlerLP
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
import numpy as np
import limits
import os
import pytz
import pickle as pkl

from qlib import init
import optuna
import random
import time
import line_profiler


def sortsegs(taskdict, segs):
    taskdict["market"] = "temp"
    taskdict["benchmark"] = "temp"
    taskdict["data_handler_config"]["instruments"] = "temp"
    taskdict["data_handler_config"]["start_time"] = segs[0]
    taskdict["data_handler_config"]["end_time"] = segs[5]
    taskdict["data_handler_config"]["fit_start_time"] = segs[0]
    taskdict["data_handler_config"]["fit_end_time"] = segs[1]
    taskdict["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = segs[0]
    taskdict["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = segs[5]
    taskdict["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["fit_start_time"] = segs[
        0
    ]
    taskdict["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["fit_end_time"] = segs[1]

    taskdict["task"]["dataset"]["kwargs"]["segments"]["train"][0] = segs[0]
    taskdict["task"]["dataset"]["kwargs"]["segments"]["train"][1] = segs[1]
    taskdict["task"]["dataset"]["kwargs"]["segments"]["valid"][0] = segs[2]
    taskdict["task"]["dataset"]["kwargs"]["segments"]["valid"][1] = segs[3]
    taskdict["task"]["dataset"]["kwargs"]["segments"]["test"][0] = segs[4]
    taskdict["task"]["dataset"]["kwargs"]["segments"]["test"][1] = segs[5]
    return taskdict


def fixdates(datestart, dateend):
    start = dateparser.parse(datestart)
    end = dateparser.parse(dateend)
    d1_ts = time.mktime(start.timetuple())
    d2_ts = time.mktime(end.timetuple())

    # They are now in seconds, subtract and then divide by 60 to get minutes.
    totaldifference = int(d2_ts - d1_ts) // 60

    teststart = dateparser.parse("July 01 2022")
    testend = end
    d1_ts = time.mktime(teststart.timetuple())
    d2_ts = time.mktime(testend.timetuple())

    # They are now in seconds, subtract and then divide by 60 to get minutes.
    testdifference = int(d2_ts - d1_ts) // 60
    trainvalidperiod = totaldifference - testdifference
    trainperiod = int(trainvalidperiod * 0.8)
    validperiod = int(trainvalidperiod * 0.2)
    trainend = start + timedelta(minutes=trainperiod)
    validstart = trainend + timedelta(minutes=1)
    validend = validstart + timedelta(minutes=validperiod)

    # Final adjustment to ensure no overlap
    teststart = teststart + timedelta(minutes=2)
    return [
        start.strftime("%d %B, %Y, %H:%M:%S"),
        trainend.strftime("%d %B, %Y, %H:%M:%S"),
        validstart.strftime("%d %B, %Y, %H:%M:%S"),
        validend.strftime("%d %B, %Y, %H:%M:%S"),
        teststart.strftime("%d %B, %Y, %H:%M:%S"),
        testend.strftime("%d %B, %Y, %H:%M:%S"),
    ]


def objective(trial):

    shuffle = trial.suggest_categorical("shuffle", [True, False])
    # shuffle = True
    global dataset_t
    timed = datetime.now()
    try:
        os.remove("F:/train_main.bin")
        os.remove("F:/valid_main.bin")
    except:
        pass
    try:
        os.remove("F:/train_test.bin")
        os.remove("F:/valid_test.bin")
    except:
        pass
    with open(
        "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-qlib/instruments/all.txt",
        "r",
    ) as f:
        lines = f.readlines()
        stocks = {}
        for e, line in enumerate(lines):
            stocks.update(
                {
                    line.strip().split("\t")[0]: [
                        line.strip().split("\t")[1],
                        line.strip().split("\t")[2],
                    ]
                }
            )
    keylist = list(stocks.keys())

    try:
        stock = globals()["stock_current"]
    except:
        stock = trial.suggest_categorical("stock", keylist)
    # stock = "ADABUSD"
    segs = fixdates(line.strip().split("\t")[1], line.strip().split("\t")[2])
    with open(
        "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-qlib/instruments/temp.txt",
        "w",
    ) as f:
        strings = stock + "\t" + stocks[stock][0] + "\t" + stocks[stock][1] + "\n"
        f.write(strings)
    init(provider_uri={"1min": provider_uri}, region=REG_CRY, clear_mem_cache=True)
    vl = 0
    # newfeats = trial.suggest_categorical("newfeats", [True, False])
    newfeats = False
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "boosting": "goss",
                "loss": "binary",
                # "loss": trial.suggest_categorical(
                #     "loss", ["binary", "xentropy", "focal"]
                # ),
                "feature_fraction_seed": 0,
                # "linear_lambda":trial.suggest_loguniform("linear_lambda", 1e-8, 1e4),
                # "feature_fraction_bynode": trial.suggest_uniform(
                #     "feature_fraction_bynode", 0.4, 1.0
                # ),
                # "min_sum_hessian_in_leaf": trial.suggest_loguniform(
                #     "min_sum_hessian_in_leaf", 1e-8, 1e4
                # ),
                # "min_sum_hessian_in_leaf": 1,
                # "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                # "learning_rate": trial.suggest_uniform("learning_rate", 0.05, 0.4),
                "learning_rate": 0.2,
                # "subsample": trial.suggest_uniform("subsample", 0.4, 1),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e5),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e5),
                "max_depth": trial.suggest_int("max_depth", 1, 30),
                "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
                # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                # "feature_pre_filter":False,
                "verbosity": -1,
                "max_bin": 255,
                "nthread": 6,
                "device": "gpu",
                # "gpu_use_dp": True,
                # "force_row_wise": True,
                # "deterministic": True,
                "seed": 10,
            },
        },
    }
    if task["model"]["kwargs"]["boosting"] == "goss":
        task["model"]["kwargs"].update(
            {
                "top_rate": trial.suggest_uniform("top_rate", 0, 0.5),
                "other_rate": trial.suggest_uniform("other_rate", 0, 0.5),
            }
        )
    if task["model"]["kwargs"]["boosting"] != "goss":
        task["model"]["kwargs"].update(
            {
                "bagging_seed": 0,
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            }
        )
    if task["model"]["kwargs"]["boosting"] != "dart":
        task["model"]["kwargs"].update(
            {
                "early_stopping_rounds": 200,
                "num_boost_round": 1000,
            }
        )
    if task["model"]["kwargs"]["boosting"] == "dart":
        task["model"]["kwargs"].update(
            {
                "num_boost_round": 50,
                "drop_rate": trial.suggest_uniform("drop_rate", 0, 1),
                "max_drop": trial.suggest_int("max_drop", 0, 200),
                "skip_drop": trial.suggest_uniform("skip_drop", 0, 1.0),
                "drop_seed": 0,
            }
        )
    model = init_instance_by_config(task["model"])
    if task["model"]["kwargs"]["loss"] == "focal":
        alpha = trial.suggest_categorical("alpha", [None, 0, 0.25, 0.5, 0.75, 1])
        gamma = trial.suggest_categorical("gamma", [0, 0.25, 0.5, 0.75, 1])
        print(alpha, gamma)
    else:
        alpha, gamma = None, None
    evals_result = dict()

    with open("model/model_config.yaml", "r") as f:
        mdl = yaml.safe_load(f)
        mdl = sortsegs(mdl, segs)
    if not os.path.exists("F:/datasets/dataset_{}.pkl".format(stock)):

        dataset_t = init_instance_by_config(mdl["task"]["dataset"]["kwargs"]["handler"])
        try:

            dataset_t.to_pickle(
                "F:/datasets/dataset_{}.pkl".format(stock), dump_all=True
            )
        except:
            pass
        mdl["task"]["dataset"]["kwargs"]["handler"] = dataset_t
        dataset_t = init_instance_by_config(mdl["task"]["dataset"])
        del mdl["task"]["dataset"]["kwargs"]["handler"]
        with open("model/named_{}.pkl".format(stock), "wb") as f:

            df = dataset_t.prepare(
                segments=slice(None),
                col_set="feature",
                data_key=DataHandlerLP.DK_L,
            )
            pkl.dump(df.columns, f)
        dataset_t = dataset_t.handler
    else:
        with open("F:/datasets/dataset_{}.pkl".format(stock), "rb") as f:
            dataset_t = pkl.load(f)

    def balancedata(df, ver):
        if ver == "learn":

            mdl["task"]["dataset"]["kwargs"]["handler"] = df
            dataset_t = init_instance_by_config(mdl["task"]["dataset"])
            _df = dataset_t.prepare(
                ["train", "valid", "test"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )

        else:
            mdl["task"]["dataset"]["kwargs"]["handler"] = df
            dataset_t = init_instance_by_config(mdl["task"]["dataset"])
            _df = dataset_t.prepare(
                ["train", "valid", "test"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_I,
            )
        for e, df in enumerate(_df):
            if e == 2:
                continue
            true_df = df[df["label"]["LABEL{}".format(vl)] == 1]
            truelen = len(true_df)

            false_df = df[df["label"]["LABEL{}".format(vl)] == 0]
            falselen = len(false_df)

            if truelen > falselen:
                smallerlen = falselen
                true_df = true_df.sample(n=smallerlen, random_state=1)
            else:
                smallerlen = truelen
                false_df = false_df.sample(n=smallerlen, random_state=1)
            print(smallerlen)
            df = pd.concat([false_df, true_df])
            df = df.sort_index()
            _df[e] = df
        try:
            df = pd.concat([_df[0], _df[1], _df[2]], verify_integrity=True)
        except Exception as e:
            print(e)
            df = pd.concat([_df[0], _df[1]], verify_integrity=True)

        df = df.sort_index()
        dfind = df.index.drop_duplicates()
        df = df.reindex(dfind)
        return df

    balance = trial.suggest_categorical("balance", [True, False])

    def func_labels(data):

        # INFER
        allabels = data._infer["label"]
        datalabel = allabels["LABEL{}".format(vl)]

        def func(x):
            if x > 0.5:
                return 1
            else:
                return 0

        datalabel = datalabel.apply(func)

        allabels["LABEL{}".format(vl)] = datalabel
        data._infer["label"] = allabels

        if balance == True:
            data._infer = balancedata(data, "infer")
        if shuffle == True:
            data._infer = data._infer

        print("e")

        # LEARN

        allabels = data._learn["label"]
        datalabel = allabels["LABEL{}".format(vl)]

        def func(x):
            if x > 0.5:
                return 1
            else:
                return 0

        datalabel = datalabel.apply(func)
        print("e")

        allabels["LABEL{}".format(vl)] = datalabel
        data._learn["label"] = allabels
        if balance == True:
            data._learn = balancedata(data, "learn")
        return data

    dataset_t = func_labels(dataset_t)

    with open("model/model_config.yaml", "r") as f:
        mdl = yaml.safe_load(f)
        mdl = sortsegs(mdl, segs)
    topk = trial.suggest_categorical("topkd", [30, 50, 100, 200, 300, 400, 500])
    # topk = 200
    # topk = 50
    try:

        mdl["task"]["dataset"]["kwargs"]["handler"] = dataset_t.handler
    except:
        mdl["task"]["dataset"]["kwargs"]["handler"] = dataset_t
    dataset_t = init_instance_by_config(mdl["task"]["dataset"])
    mk = False
    if mk == True:
        with open("model/named_{}.pkl".format(stock), "wb") as f:
            df = dataset_t.prepare(
                segments=slice(None), col_set="feature", data_key=DataHandlerLP.DK_L
            )
            pkl.dump(df, f)

    del mdl
    calcfeats = True
    if calcfeats == True:
        if newfeats == False:
            loadold = True
            if loadold == True:
                try:
                    with open("model/fm_{}.pkl".format(stock), "rb") as f:
                        model = pkl.load(f)
                except:

                    model.fit(
                        dataset_t,
                        meta_input=vl,
                        evals_result=evals_result,
                        testval="main",
                        alpha=alpha,
                        gamma=gamma,
                        shuffle=shuffle,
                    )
                    with open("model/fm_{}.pkl".format(stock), "wb") as f:
                        pkl.dump(model, f)
            else:
                model.fit(
                    dataset_t,
                    meta_input=vl,
                    evals_result=evals_result,
                    testval="main",
                    alpha=alpha,
                    gamma=gamma,
                    shuffle=shuffle,
                )
            fint = model.get_feature_importance()

            # with open("model/named_{}.pkl".format(stock), "rb") as f:
            #     df = pkl.load(f)
            #     try:
            #         cols = df.columns
            #         with open("model/named_{}.pkl".format(stock), "wb") as f:
            #             pkl.dump(cols, f)
            #         del df
            #     except:
            #         cols = df
            #         del df

            fi_named = fint.to_dict()

            fint = pd.Series(fi_named)

            # trial.suggest_categorical(
            #     "feats", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            # )

            # topk = topk[0]
            #
            col_selected = fint.nlargest(topk)
            col_selected.to_csv("model/cols.csv")
            col_selected = col_selected.index

        try:
            os.remove("model/feats_selected_{}_{}.pkl".format(topk, stock))
        except:
            pass
        if newfeats == True:

            df = dataset_t.prepare(
                "train",
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )

            df = df.astype("float32")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            df = df.astype("float32")
            print(df.isnull().any().any())
            if df.empty:
                raise ValueError(
                    "Empty data from dataset, please check your dataset config."
                )
            x, y = df["feature"], df["label"]["LABEL{}".format(vl)]
            selected_features = mrmr.mrmr_regression(
                X=x, y=y, K=topk, n_jobs=8, return_scores=True
            )
            with open("model/feats_selected_{}_{}.pkl".format(topk, stock), "wb") as f:
                pkl.dump(selected_features, f)
            col_selected = selected_features[0]
            print(selected_features)
            print(col_selected)

        prep_ds = dataset_t.prepare(
            slice(None),
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )

        feature_df = prep_ds["feature"]
        label_df = prep_ds["label"]["LABEL{}".format(vl)]

        feature_selected = feature_df.loc[:, col_selected]

        df_all = {
            "label": label_df.reindex(feature_selected.index),
            "feature": feature_selected,
        }
        df_all = pd.concat(df_all, axis=1)
        if os.path.exists("model/fea_label_df.pkl"):
            os.remove("model/fea_label_df.pkl")
        df_all.to_pickle("model/fea_label_df.pkl")
        handler = DataHandlerLP(
            data_loader={
                "class": "qlib.data.dataset.loader.StaticDataLoader",
                "kwargs": {"config": "model/fea_label_df.pkl"},
            }
        )

        with open("model/model_config.yaml", "r") as f:
            mdl = yaml.safe_load(f)
            mdl = sortsegs(mdl, segs)
        mdl["task"]["dataset"]["kwargs"]["handler"] = handler
        dataset_t = init_instance_by_config(mdl["task"]["dataset"])

    model = init_instance_by_config(task["model"])
    model.fit(
        dataset_t,
        meta_input=vl,
        evals_result=evals_result,
        testval="test",
        alpha=alpha,
        gamma=gamma,
        shuffle=shuffle,
    )
    global preds
    preds = model.predict(dataset_t)

    def objectivebinary_return():
        def conditions(x):
            if x > 0.5:
                return 1
            elif x <= 0.5:
                return 0

        func = np.vectorize(conditions)
        _preds = globals()["preds"]
        predvals = _preds.values
        predvals = func(predvals)
        _preds = pd.Series(np.squeeze(predvals), index=_preds.index)

        labl = globals()["dataset_t"].handler._infer.label["LABEL{}".format(vl)]
        labelvals = labl.reindex(_preds.index).values
        labelvals = func(labelvals)
        labl = pd.Series(np.squeeze(labelvals), index=_preds.index)

        print(_preds.head(5))
        print(labl.head(5))
        # predv = np.nan_to_num(preds.values)
        # lablv = np.nan_to_num(labl.values)
        # fig = px.histogram(labl, nbins=10)
        # fig.show()
        # fig = px.histogram(_preds, nbins=10)
        # fig.show()
        f1 = precision_score(labl, _preds)
        f2 = fbeta_score(labl, _preds, beta=0.25)
        # report = classification_report(labl, _preds, output_dict=True)
        # f1 = report["weighted avg"]["f1-score"]
        return f1, f2, labl, _preds

    f1, f2, labl, preds = objectivebinary_return()
    if not os.path.exists("model/models/stockpreds/{}".format(stock)):
        os.mkdir("model/models/stockpreds/{}".format(stock))
    if not os.path.exists("model/models/stockmodels/{}".format(stock)):
        os.mkdir("model/models/stockmodels/{}".format(stock))
    runtime = datetime.now() - timed
    print("Training took... {}".format(runtime))
    print("IC: {}".format(f2))
    with open(
        "model/models/stockpreds/{}/{}_{}_backtestpreds_{}_{}.pkl".format(
            stock, "{:.4f}".format(f1), "{:.4f}".format(f2), stock, newfeats
        ),
        "wb",
    ) as f:
        pkl.dump({"label": labl, "score": preds}, f)
        print(len(labl), len(preds))
    with open(
        "model/models/stockmodels/{}/{}_{}_model_{}_{}.pkl".format(
            stock, "{:.4f}".format(f1), "{:.4f}".format(f2), stock, newfeats
        ),
        "wb",
    ) as f:
        pkl.dump(model, f)
    # _df = dataset_t.prepare(
    #     ["train", "valid"],
    #     col_set=["label"],
    #     data_key=DataHandlerLP.DK_I,
    # )
    # _df = pd.concat(_df)["label"]

    # fig = px.histogram(_df, x="LABEL0", nbins=100)
    # fig.show()
    # plt = lgb.plot_tree(model.model, show_info=["data_percentage", "leaf_weight"])
    # mp.pyplot.show()
    # mp.pyplot.close()
    # plt[0].show()
    # plt.close()
    return f2


def backtestoptim2(trial, df=None, model=None, dataset=None):
    # vl = trial.suggest_int("label",0,4)
    vl = 0
    with open("model//backtestpreds_{}.pkl".format(vl), "rb") as handle:
        df = pkl.load(handle)["score"]

    a = trial.suggest_uniform("index modifier variable", 0.01, 5)

    b = trial.suggest_uniform("low modifier variable", 0.01, 5)

    c = trial.suggest_uniform("high modifier variable", 0.01, 5)

    # std=False
    std = trial.suggest_categorical("std", [True, False])

    if std == True:

        quant = trial.suggest_categorical("Use Quant", [True, False])
        if quant == True:
            iqh = trial.suggest_uniform("high quantile modifier variable", 0.001, 0.3)
            # iqh = 0
            iql = trial.suggest_uniform("low quantile modifier variable", 0.7, 1)
        else:
            iql = 0
            iqh = 0

    else:
        quant = False
        iqh = None
        iql = None

    # iqh = 0
    # iql = 0
    # quant = False
    # std = False
    topk = trial.suggest_int("topk", 3, 20)
    # topk= 5

    __, _, key = limits.create(
        a, l=b, h=c, std=std, quant=quant, iqh=iqh, iql=iql, vl=None
    )
    CSI300_BENCH = "BTCBUSD"
    FREQ = "1min"
    STRATEGY_CONFIG = {
        "topk": topk,
        "n_drop": 2,
        "signal": df,
        "model": model,
        "dataset": dataset,
        "trial": trial,
        "key": key,
        "vl": vl,
    }
    EXECUTOR_CONFIG = {
        "time_per_step": "1min",
        "generate_portfolio_metrics": True,
    }

    backtest_config = {
        # "start_time": pd.Timestamp(dateparser.parse("2022-0:1-01 00:10:00")),
        # "end_time": pd.Timestamp(dateparser.parse("2022-05-01 00:00:00")),
        "start_time": pd.Timestamp(dateparser.parse("2022-07-18 00:05:00")),
        "end_time": pd.Timestamp(dateparser.parse("2022-07-26 00:00:00")),
        "account": 5000,
        "benchmark": CSI300_BENCH,
        "exchange_kwargs": {
            "freq": FREQ,
            "deal_price": "close",
            # "open_cost": 0.00075,
            # "close_cost": 0.00075,
            "open_cost": 0.000000000075,
            "close_cost": 0.00000000075,
            "impact_cost": 0.000000001,
            "min_cost": 0,
        },
    }

    strategy_obj = TopkDropoutStrategyNew(**STRATEGY_CONFIG)

    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
    # backtest
    portfolio_metric_dict, indicator_dict = backtest(
        executor=executor_obj, strategy=strategy_obj, **backtest_config
    )
    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
    if portfolio_metric_dict == None:
        return
    # backtest info
    report_normal, positions_normal = portfolio_metric_dict.get("1min")
    fig = qcr.report.report_graph(report_normal, show_notebook=False)
    fig[0].write_html("figure1.html")
    # analysis
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"], freq=analysis_freq
    )
    analysis["excess_return_with_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"] - report_normal["cost"],
        freq=analysis_freq,
    )

    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    # log metrics
    analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())
    # print out results
    print(f"The following are analysis results of benchmark return({analysis_freq}).")
    print(risk_analysis(report_normal["bench"], freq=analysis_freq))
    print(
        f"The following are analysis results of the excess return without cost({analysis_freq})."
    )
    print(analysis["excess_return_without_cost"])
    print(
        f"The following are analysis results of the excess return with cost({analysis_freq})."
    )
    print(analysis["excess_return_with_cost"])
    actv = report_normal.account.values[-1]  # - report_normal.total_cost.values[-1]
    print(actv)
    # return actv
    if os.path.exists(
        "C:/Users/Ian/Documents/Financial testing/model/scaler{}.pkl".format(key)
    ):
        os.remove(
            "C:/Users/Ian/Documents/Financial testing/model/scaler{}.pkl".format(key)
        )
    return actv  # analysis["excess_return_with_cost"]["risk"]["annualized_return"]


def rounded_to_the_last_epoch_1m(now):
    rounded = now - (now - datetime.min.replace(tzinfo=timezone.utc)) % timedelta(
        minutes=1
    )
    return rounded.replace(tzinfo=None)


def run(d, train):
    global stock_current
    global model
    dmax = rounded_to_the_last_epoch_1m(d)

    with open(
        "C:/Users/Ian/Documents/Financial testing/model/model_config.yaml"
    ) as fil:
        cf = yaml.safe_load(fil)

    if not train == True:
        dlo = 1

        cf["task"]["dataset"]["kwargs"]["segments"]["train"][0] = pd.Timestamp(
            dmax - timedelta(minutes=dlo)
        )
        cf["task"]["dataset"]["kwargs"]["segments"]["test"][1] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["segments"]["train"][1] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["segments"]["valid"][0] = pd.Timestamp(
            dmax - timedelta(minutes=dlo)
        )
        cf["task"]["dataset"]["kwargs"]["segments"]["valid"][1] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["segments"]["test"][0] = pd.Timestamp(
            dmax - timedelta(minutes=dlo)
        )
        cf["data_handler_config"]["start_time"] = pd.Timestamp(
            dmax - timedelta(minutes=dlo)
        )
        cf["data_handler_config"]["fit_start_time"] = pd.Timestamp(
            dmax - timedelta(minutes=dlo)
        )
        cf["data_handler_config"]["fit_end_time"] = pd.Timestamp(dmax)
        cf["data_handler_config"]["end_time"] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["handler"]["kwargs"][
            "start_time"
        ] = pd.Timestamp(dmax - timedelta(minutes=dlo))
        cf["task"]["dataset"]["kwargs"]["handler"]["kwargs"][
            "fit_start_time"
        ] = pd.Timestamp(dmax - timedelta(minutes=dlo))
        cf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = pd.Timestamp(
            dmax
        )
        cf["task"]["dataset"]["kwargs"]["handler"]["kwargs"][
            "fit_end_time"
        ] = pd.Timestamp(dmax)

        cf["task"]["dataset"]["kwargs"]["segments"]["train"][0] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["segments"]["test"][1] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["segments"]["train"][1] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["segments"]["valid"][0] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["segments"]["valid"][1] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["segments"]["test"][0] = pd.Timestamp(dmax)
        cf["data_handler_config"]["start_time"] = pd.Timestamp(dmax)
        cf["data_handler_config"]["fit_start_time"] = pd.Timestamp(dmax)
        cf["data_handler_config"]["fit_end_time"] = pd.Timestamp(dmax)
        cf["data_handler_config"]["end_time"] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["handler"]["kwargs"][
            "start_time"
        ] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["handler"]["kwargs"][
            "fit_start_time"
        ] = pd.Timestamp(dmax)
        cf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = pd.Timestamp(
            dmax
        )
        cf["task"]["dataset"]["kwargs"]["handler"]["kwargs"][
            "fit_end_time"
        ] = pd.Timestamp(dmax)

        dataset = init_instance_by_config(cf["task"]["dataset"])

    # else:
    #     dataset = init_instance_by_config(cf['task']['dataset'])
    if train == True:
        retrain = True
        if retrain == True:

            with open("model/model_config.yaml", "r") as f:
                mdl = yaml.safe_load(f)

            study_namei = "ADABUSD"
            storage_namei = "sqlite:///LGBM_0.db"
            studyi = optuna.create_study(
                study_name=study_namei,
                storage=storage_namei,
                load_if_exists=True,
                direction="maximize",
            )
            besttrialinit = studyi.best_params
            optim = True
            if optim == True:
                pass
                study_name = "LGBM_{}_FULL".format(0)
                storage_name = "sqlite:///{}.db".format(study_name)
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_name,
                    load_if_exists=True,
                    direction="maximize",
                )
                with open(
                    "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-qlib/instruments/all.txt",
                    "r",
                ) as f:
                    lines = f.readlines()
                    stocks = {}
                    for e, line in enumerate(lines):
                        stocks.update(
                            {
                                line.strip().split("\t")[0]: [
                                    line.strip().split("\t")[1],
                                    line.strip().split("\t")[2],
                                ]
                            }
                        )
                keylist = list(stocks.keys())
                goodtrials = [x for x in study.trials if x.state == TrialState.COMPLETE]
                completedtrials = len(goodtrials)
                runtrials = 0
                for stock in keylist:
                    # stock = "BNBBUSD"
                    for feat in [False]:
                        if runtrials < completedtrials:
                            runtrials += 1
                            continue
                        # for subtrial in [0, 0.25, 0.5, 0.75, 1]:
                        #     besttrialinit["gamma"] = subtrial
                        besttrialinit["stock"] = stock
                        besttrialinit["newfeats"] = feat
                        # study.enqueue_trial(besttrialinit)
                        try:
                            try:
                                os.remove("model/fm_{}.pkl".format(stock))
                            except:
                                pass
                            study_names = "{}".format(stock)
                            storage_names = "sqlite:///stockdbs/{}.db".format(
                                study_names
                            )
                            studystock = optuna.create_study(
                                study_name=study_names,
                                storage=storage_names,
                                load_if_exists=True,
                                direction="maximize",
                            )
                            stock_current = stock
                            studystock.enqueue_trial(besttrialinit)
                            studystock.optimize(
                                objective,
                                n_trials=50,
                                n_jobs=1,
                                catch=(
                                    AssertionError,
                                    RuntimeError,
                                    UnboundLocalError,
                                    UnpicklingError,
                                ),
                            )
                            besttrialinit2 = studystock.best_params
                            besttrialinit2["stock"] = stock
                            besttrialinit2["newfeats"] = feat
                            study.enqueue_trial(besttrialinit2)

                            study.optimize(
                                objective,
                                n_trials=1,
                                n_jobs=1,
                                catch=(
                                    AssertionError,
                                    RuntimeError,
                                    UnboundLocalError,
                                    UnpicklingError,
                                ),
                            )
                        except Exception as e:
                            print(e)

                    gc.collect()

        makepreds = False
        if makepreds == True:

            with open("model/models//model_full.pkl", "rb") as f:
                model = pkl.load(f)
            with open("model/model_config.yaml", "r") as f:
                mdl = yaml.safe_load(f)

            with open("F:/dataset.pkl", "rb") as f:
                dataset = pkl.load(f)
            mdl["task"]["dataset"]["kwargs"]["handler"] = dataset
            dataset = init_instance_by_config(mdl["task"]["dataset"])
            del mdl

            preds = model.predict(dataset, segment="valid")

            if os.path.exists("model//backtestpreds_n.pkl"):
                os.remove("model//backtestpreds_n.pkl")
            with open("model//backtestpreds_n.pkl", "wb") as handle:
                pkl.dump({"score": preds}, handle)
        backtestt = False
        if backtestt == True:

            study_name2 = "backtest"
            storage_name2 = "sqlite:///backtest2.db".format(study_name2)
            study2 = optuna.create_study(
                study_name=study_name2,
                storage=storage_name2,
                load_if_exists=True,
                direction="maximize",
            )
            # study2.enqueue_trial(bestparams)
            # study2.optimize(backtestoptim2,n_trials=100,n_jobs=1,catch=(IndexError,))
            bestparams2 = study2.best_params
            besttrial2 = study2.best_trial
            bestvalue2 = study2.best_value
            study_name2 = "backtest"
            storage_name2 = "sqlite:///backtest.db".format(study_name2)
            study2 = optuna.create_study(
                study_name=study_name2,
                storage=storage_name2,
                load_if_exists=True,
                direction="maximize",
            )
            print(bestvalue2)
            study2.enqueue_trial(bestparams2)
            study2.optimize(backtestoptim2, n_trials=100, n_jobs=1, catch=(IndexError,))

        return 1

    else:
        try:
            model = globals().model
        except:

            with open("model/models/model_full.pkl", "rb") as f:
                model = pkl.load(f)
        preds = model.predict(dataset)

        # preds = pd.DataFrame(lastdaypreds)

        p = preds.index[-1][0]
        nparr = preds.loc[p]
        # nparr = nparr.reindex(sorted(nparr.columns), axis=1)
        lasdayidx = preds.index[-1][0]

        lastdaypreds = [nparr, lasdayidx]
    with open("model//lastdaypreds.pkl", "wb") as handle:
        pkl.dump(lastdaypreds, handle)
    print(lasdayidx)


def read_preds(snap, clnam, time, default=True):
    time = rounded_to_the_last_epoch_1m(time)
    with open("model//lastdaypreds.pkl", "rb") as handle:
        lastdaypreds = pkl.load(handle)

    curtime = lastdaypreds[1]
    inont = lastdaypreds[1] - timedelta(hours=4)
    print(
        "Trading for time: {}".format(lastdaypreds[1]),
        "UTC",
        " or {} EST".format(inont),
    )
    lastdaypreds = lastdaypreds[0]
    # lastdaypreds.drop(labels=['BNBBUSD'],axis=0,inplace=True)

    for item_l in snap:
        if item_l == "BUSD":
            cash_on_hand = float(snap[item_l]["amount"])
    if "BUSD" in snap:
        del snap["BUSD"]

    exchng = Exchange(start_time=time, end_time=time)

    # position = Position(cash=cash_on_hand,position_dict=snap)
    # position.fill_stock_value(start_time=curtime,freq='1min')
    # posval = position.calculate_value()

    # with open("C:/Users/Ian/Documents/Financial testing/balances/{}.csv".format(clnam),'a',newline='') as clbal:
    #     write = csv.writer(clbal)

    #     write.writerow([inont,posval])

    starttime = pd.Timestamp(time - timedelta(minutes=2))
    endtime = pd.Timestamp(time)
    snap["cash"] = cash_on_hand

    acct = create_account_instance(starttime, endtime, benchmark=None, account=snap)
    common_infra = CommonInfrastructure(trade_account=acct, trade_exchange=exchng)
    level_infra = LevelInfrastructure()
    level_infra.reset_infra(common_infra=common_infra)
    study_name2 = "backtest"
    storage_name2 = "sqlite:///backtest.db"
    study2 = optuna.load_study(study_name=study_name2, storage=storage_name2)
    bestparams2 = study2.best_params

    dropstrat = TopkDropoutStrategyNew(
        topk=bestparams2["topk"],
        n_drop=1,
        key=0,
        signal=lastdaypreds,
        trade_exchange=exchng,
        level_infra=level_infra,
        common_infra=common_infra,
        test=False,
    )
    buy_order_list, sell_order_list, _ = dropstrat.generate_trade_decision(lastdaypreds)

    return buy_order_list, sell_order_list


if __name__ == "__main__":
    provider_uri = (
        "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-qlib"
    )
    init(provider_uri={"1min": provider_uri}, region=REG_CRY)
    run(datetime.now(pytz.utc), True)
