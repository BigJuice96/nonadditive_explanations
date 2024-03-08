import os
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code/GIT_Repo')
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sksurv.column import encode_categorical
from sksurv.svm import FastKernelSurvivalSVM
from sklearn.preprocessing import normalize
from GIT_Repo.old_files.choquex_kernel import *
from GIT_Repo.old_files.mlex_kernel import *


def survivalMLeX(data_x, y, alpha=1.0, finetune=True, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, range_alpha= (2.0 ** np.arange(-12, 13, 2))):
    """
    Takes a dataframe with survival targets and returns Shapley values indicating the feature importance for each feature.
    data_x is a pandas dataframe and y are the survival targets.
    returns a pandas dataframe.
    If finetune is set to true, it returns the optimal alpha.
    """
    x = encode_categorical(data_x)
    x_normalized = normalize(x.values, norm='l2')
    kernel_matrix = Shapley_kernel(x_normalized,x_normalized) #clinical_kernel(data_x) 
    param_grid = {"alpha": range_alpha}
    cv = ShuffleSplit(n_splits=n_shuffle_splits, test_size=0.5, random_state=0)
    kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0, rank_ratio=rank_ratio, fit_intercept=fit_intercept)

    if finetune==True:
        kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=-1, refit=False, cv=cv)
        kgcv = kgcv.fit(kernel_matrix, y)
        alpha = kgcv.best_params_["alpha"]
    kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0, rank_ratio=rank_ratio, fit_intercept=fit_intercept)
    kssvm = kssvm.fit(kernel_matrix, y)    

    ## Computing shapley value
    shap_vals = Shapley_value(x_normalized, kssvm.coef_, np.arange(0,data_x.shape[0])) 
    result = pd.DataFrame()
    result["shapley_values"]=shap_vals
    result["shapley_values_absolute"]=np.absolute(shap_vals)
    result["feature_names"]= list(data_x.columns)
    result = result.sort_values(by="shapley_values_absolute", ascending=False)
    del result["shapley_values_absolute"]
    return result



def survivalChoqEx(data_x, y, alpha=1.0, finetune=True, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, range_alpha= (2.0 ** np.arange(-12, 13, 2))):
    """
    Takes a dataframe with survival targets and returns Shapley values indicating the feature importance for each feature.
    data_x is a pandas dataframe and y are the survival targets.
    returns a pandas dataframe.
    If finetune is set to true, it returns the optimal alpha.
    """
    x = encode_categorical(data_x)
    x_normalized = normalize(x.values, norm='l2')
    kernel_matrix = ChoquetKernel(x_normalized) # This line is different 
    kernel_matrix = kernel_matrix.get_kernel()
    param_grid = {"alpha": range_alpha}
    cv = ShuffleSplit(n_splits=n_shuffle_splits, test_size=0.5, random_state=0)
    kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0, rank_ratio=rank_ratio, fit_intercept=fit_intercept)

    if finetune==True:
        kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=-1, refit=False, cv=cv)
        kgcv = kgcv.fit(kernel_matrix, y)
        alpha = kgcv.best_params_["alpha"]
    kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0, rank_ratio=rank_ratio, fit_intercept=fit_intercept)
    kssvm = kssvm.fit(kernel_matrix, y)    

    ## Computing shapley value
    shap_vals_choqu = Shapley_value_Choquet(x_normalized, kssvm.coef_, "") #137 is the number of data points
    result = pd.DataFrame()
    result["shapley_values"]=shap_vals_choqu
    result["shapley_values_absolute"]=np.absolute(shap_vals_choqu)
    result["feature_names"]= list(data_x.columns)
    result = result.sort_values(by="shapley_values_absolute", ascending=False)
    del result["shapley_values_absolute"]
    return result

