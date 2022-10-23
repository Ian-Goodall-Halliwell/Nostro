import os
import pickle as pkl
import pandas as pd


def getprocs():
    for dset in os.listdir("F:/datasets"):
        with open(os.path.join("F:/datasets", dset)) as f:
            dataset = pkl.load(f)
        print("e")


def joinpreds():
    labels = []
    preds = []
    for pred in os.listdir("model/models/stockpreds"):
        with open(os.path.join("model/models/stockpreds", pred), "rb") as f:
            predset = pkl.load(f)
        # predlabel = pred.split("_")[3]
        labels.append(predset["label"])
        preds.append(predset["score"])
    labels = pd.concat(labels, axis=0)
    preds = pd.concat(preds, axis=0)
    print("e")


def createpreds(preddict):
    pass


joinpreds()
