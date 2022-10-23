import pickle as pkl
import optuna
import plotly.express as px
from sklearn import metrics
from optuna import visualization

stud = True
if stud == True:
    study_namei = "LGBM_{}_FULL".format(0)
    storage_namei = "sqlite:///LGBM_0_FULL.db".format(study_namei)
    studyi = optuna.create_study(
        study_name=study_namei,
        storage=storage_namei,
        load_if_exists=True,
        direction="maximize",
    )
    # bv = studyi.best_trial

    stock = "ADABUSD"
    study_name = "{}".format(stock)
    # storage_name = "sqlite:///stockdbs/{}.db".format(study_name)
    storage_name = "sqlite:///LGBM_0.db"
    studystock = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )
    vis = visualization.plot_param_importances(studystock)
    vis.show()
    binv = studystock.best_trial
dset = False
if dset == True:
    with open(
        "F:\datasets\dataset_DOTBUSD.pkl",
        "rb",
    ) as f:
        labl = pkl.load(f)._learn["label"]["LABEL0"]
        fig = px.histogram(labl, x="LABEL0", nbins=100)
        fig.show()

        # hist_data = [labl]
        # group_labels = ["distplot"]  # name of the dataset

        # fig = ff.create_distplot(hist_data, group_labels, bin_size=0.0002)
        # fig.show()
        print("e")
with open(
    "C:/Users/Ian/Documents/Financial testing/model/models/stockpreds/ADABUSD/0.5315_0.4876_backtestpreds_ADABUSD_True.pkl",
    "rb",
) as f:
    labls = pkl.load(f)
    labl = labls["score"]
    fig = px.histogram(labl, x=0, nbins=100, title="score")
    fig.show()

    labl = labls["label"]
    fig = px.histogram(labl, x=0, nbins=100, title="label")
    fig.show()

    conf = metrics.confusion_matrix(labls["label"], labls["score"])
    disp = metrics.ConfusionMatrixDisplay(conf)

    disp.plot()
    acc = metrics.roc_auc_score(labls["label"], labls["score"])
    # hist_data = [labl]
    # group_labels = ["distplot"]  # name of the dataset

    # fig = ff.create_distplot(hist_data, group_labels, bin_size=0.0002)
    # fig.show()
    print("e")
