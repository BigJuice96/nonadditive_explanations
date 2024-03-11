import os 
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code/GIT_Repo')
from explainer import ExplainSurvival
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
import os
import time
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import set_config
from sksurv.linear_model import CoxPHSurvivalAnalysis
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code/GIT_Repo')
from survshap.survshap import *
from survshap.survlime import *
from explainer import ExplainSurvival
import surv_utils
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest










    # load the relevant data
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/')
with open('/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/cat_feature_dict.pkl', 'rb') as fp:
    cat_feature_dict = pickle.load(fp)
data_w1_dummied = pd.read_csv("/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/data_w1_dummied.csv", index_col=0)
data_w1_dummied_unstandardised = pd.read_csv("/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/data_w1_dummied_unstandardised.csv", index_col=0)
variable_definitions = pd.read_csv("/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/variable_definitions.csv", index_col=0)
data_w1 = pd.read_csv("/Users/abdallah/Desktop/Kings College Project/Code/Data prepation ELSA Dementia/DATA/data_w1.csv", index_col=0)
#TODO put that in the preprocessing file


    # feature aggregation of predictors indicating social participation 
data_w1_dummied["social"] = data_w1_dummied[['scorg1', 'scorga2', 'scorg4', 'scorg5', 'scorg6', 'scorg7', 'scorg8']].sum(axis=1)

# Features to include according to literature:
    # age, sex and race  , grip-strength, number of chronic diseases and depressive symptoms
    # educational degree, increased age, being female and absent religious status 
    # neuropsychiatric symptoms, cognitive engagement
selected_features = ['scorg3', 'social',
                         "cesd_sc",'n_chronic_diseases',"n_neuropsychiatric_symptoms",
                          'ethnicity_White', 'gender_Men', 'mmgsd_w2', 
                        'edqual_Degree or equiv', 'edqual_Foreign/other',
                        'edqual_NVQ/GCE O/A Level equiv', 'edqual_No qualification', 
                         'memory1', 'procspeed1', # left out 'exe1',  as it correlates strongly (0.46) with memory1 and is a similar construct
                          "age", "dem_hse_w8", "time"]
    # the categorical features of the selected features
cat_features = [ 'scorg3','social',
                          'ethnicity_White', 'gender_Men', 
                        'edqual_Degree or equiv', 'edqual_Foreign/other',
                        'edqual_NVQ/GCE O/A Level equiv', 'edqual_No qualification', 
                          ]
    # random seed
random_state=1
    # use 100 instances for testing, the rest for training
test_indices = data_w1_dummied.sample(100, random_state=random_state).index
training_indices = [x for x in data_w1_dummied.index if x not in test_indices]
training = data_w1_dummied.loc[training_indices, selected_features]
test = data_w1_dummied.loc[test_indices, selected_features]

    # Split into predictors and targets
data_x, data_y = surv_utils.convert_time_y_to_surv_y(df = training, y = "time", event_indicator = "dem_hse_w8", only_targets = False, round_y= False) 
test_x, test_y = surv_utils.convert_time_y_to_surv_y(df = test, y = "time", event_indicator = "dem_hse_w8", only_targets = False, round_y= False)

    # Fit a survival model that we will try to explain: a RSF
rsf = RandomSurvivalForest(
    n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=1
)
rsf.fit(data_x, data_y)


    # Select an instance to explain
i = data_x.iloc[0,:]

    # Create an explainer object
explainer = ExplainSurvival(
                 model = rsf, # the model whose output we want to explain 
                 x = data_x, # DUMMIED data that was used to train the blackbox model
                 y = data_y, # the outcome data in a format suitable for sklearn survival models  
                 categorical_features = cat_features, # List of strings of the feature labels corresponding to categorical features
                 random_state = None, # random state for intilisation of random functions
                 )

    # Explain an instance using any of the four methods "SurvMLeX", "SurvChoquEx", "SurvLIME" or "SurvSHAP"
explanation = explainer.explain_instance( 
                        instance = i , # Pass instance as a np.array or pd.series                         
                        method = "SurvMLeX", # Explanation method: either "SurvMLeX", "SurvChoquEx", "SurvLIME" or "SurvSHAP"
                        binary=False, # Using a binary representation for the perturbed samples. Default is False
                        aggregation_method = "percentile", # How to aggregate survival function predictions to a time-to-event estimate: "trapezoid", "median" or "percentile"
                        threshold_aggregation = 0.5, # Which threshold to use when aggregating survival curves to a time-to-event estimate using the percentile method (if it is not finetuned)
                        find_best_cutoff = True, # Finds optimal value for threshold_blackbox_clf argument
                        n_perturbed=1000, # Size of the perturbed dataset
                        weighing_multiplier = 2.0, # How much larger the perturbed dataset will be after weighing through sampling with replacement  
                        finetune_explainer=True, # Whether to finetune expainer (only for SurvMLeX, SurvChoquEx and SurvSHAP)
                        only_shapley_values=False, # Not relevant if method="SurvLIME". For other methods it wont use identified features to fit a Cox regression model but give the shapley values right away
                        k_features=3, # The number of features to retain if a penalised Cox regression is fit (only if only_shapley_values=False)
                        plot_curve=True, # Whether explainer should also plot survival curve (only if only_shapley_values=False)
                        alpha_min_ratio=0.01, # alpha_min_ratio argument for the penalised Cox regression (only if only_shapley_values=False)
                        alpha=1.0, # Alpha parameter of SurvMLeX or SurvChoquEx
                        survshap_calculation_method="sampling" # for survshap method: can be either "kernel", "sampling", "treeshap" or "shap_kernel"
                        )



import surv_utils
surv_utils.train_cox_and_explain(3, 
                          data_x.iloc[:2000,:], 
                          data_y[:2000], 
                          rsf.predict_survival_function(data_x, return_array=True), 
                          rsf.predict_survival_function(data_x)[0].x, 
                          False,
                          alpha_min_ratio=0.01,
                          plot_curve=True)

    # retrieve the shapley values and other useful data from the explanation object
explanation["shapley_values"] 
explanation["explanation_text"]

    # You can also get a global measure of feature importance using SurvMLeX or SurvChoquEx
    # Simply apply the methods on the entire dataset
from nonadditive_explainers import nonadditive_explainer
import numpy as np
global_importance = nonadditive_explainer(data_x, data_y, method = "SurvMLeX", alpha=0.1, finetune=True, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=1, range_alpha= (2.0 ** np.arange(-8, 7, 2)))


    # TODO maybe also show how to apply it to different packages besides sklearn


