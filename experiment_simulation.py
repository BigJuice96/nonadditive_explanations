import os
import pandas as pd
import numpy as np
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code')
import explain_survival
from explain_survival import ExplainSurvival
from survshap.survshap import *
import matplotlib.pyplot as plt
from survshap.survshap.predict_explanations import *
from survshap.survlime import *
import surv_utils
import math
import importlib
import time
from survivalSVM import survivalSVM
from survivalSVM import survivalChoqEx
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code')
importlib.reload(surv_utils)
importlib.reload(explain_survival)


    # TODO STILL TO BE ADAPTED TO THE NEW EXPLAINER CLASS

    # This one is from dataADAPTED
data = pd.read_csv("/Users/abdallah/Desktop/Kings College Project/Code/exp1_data_complexADAPTED.csv")

categorical_features = ['x1', 'x2'] # x1 and x2 are binary

# Initialise explainer object
explainer=ExplainSurvival( # Takes a little less than a minute (without balancing)
            data, # the data to be used for training the blackbox and the booster
            categorical_features,
            "time",
            "event",
            blackbox= "rsf", # can be "gbm" or "rsf"
            balance_data=False, # Whether to balance the data before training
            threshold_blackbox_clf=0.7,
            feature_selection = None, # Arguments None (all features are used) or a list of strings containing feature labels
            random_state=1 # random state for intilisation of random functions
            )

# Histogram of the distribution of the time variable 
# Print that again when done and save it
plt.hist(np.array(data["time"]))
plt.show()

# Calculate the ground truth
ground_truth_global = explainer.global_feature_importance(15) # The two variables from the interaction are most important
ground_truth_global


# Can calculate the CIs for the global ground truth in terms of feature importance here. n=20 bc of n_repeats see docs:
# https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
# https://statisticsglobe.com/draw-plot-with-confidence-intervals-in-r

x = ground_truth_global["feature"]
y = ground_truth_global["feature_importance"]
lower = ground_truth_global["feature_importance"] - 1.96*(ground_truth_global["importances_std"]/np.sqrt(16))
upper = ground_truth_global["feature_importance"] + 1.96*(ground_truth_global["importances_std"]/np.sqrt(16))


PVI_confidence_intervals = {"x":x,
                            "y":y,
                            "lower":lower,
                            "upper":upper
                            }
PVI_confidence_intervals = pd.DataFrame(PVI_confidence_intervals)
PVI_confidence_intervals.to_csv("/Users/abdallah/Desktop/Kings College Project/Code/results/PVI_confidence_intervals.csv")
    # Use the R file to plot the CIs



    # Specify number of features to consider + number of explanations
N_test = 100
k_features = len(data.drop(["time","event"], axis=1).columns) # include all features in the original dataset
binary=False
n_perturbed = 2000
n_samples = 500
only_shapley_values=True
alpha=1.0
explanations = dict()
explanations["survLIME"] = list()
explanations["survSHAP"] = list()
explanations["SVM"] = list()
explanations["ChoquEx"] = list()

for instance in list(data.index)[:N_test]:
    instance_id = instance
    instance = np.array(data.drop(["event","time"],axis=1).loc[instance_id,:])

#     # SurvLIME EXPLANATIONS
#     st = time.time()
#     explanationLIME = explainer.explain_instance( instance,
#                         k_features=k_features, 
#                         binary=binary, 
#                         plot_inverse_surv_curve = True, 
#                         bb_surv_curve_in_explanation = True,
#                         alpha_min_ratio=0.01,
#                         n_perturbed=n_perturbed,
#                         plot_curve=False
#                         )
#     topkfeatures_LIME = list(explanationLIME["hazard_ratios"].sort_values(ascending=False).index)
#     explanations["survLIME"].append(topkfeatures_LIME)
#     et = time.time()
#     elapsed_time = et - st # print('Execution time:', elapsed_time, 'seconds')
#     print("SurvLIME: Explained instance ", instance_id, ": ", topkfeatures_LIME,". Duration in seconds: ",elapsed_time, "\n")

#     # SurvSHAP EXPLANATIONS
#     st = time.time()
#     bb_exp = SurvivalModelExplainer(explainer.blackboxmodel, explainer.train_x, explainer.train_y)
#     explanationSHAP = PredictSurvSHAP()
#     explanationSHAP.fit(bb_exp, instance)
#     importance_orderingsSHAP = (explanationSHAP.result.sort_values(by="aggregated_change", key=lambda x: -abs(x)).index.to_list())
#     explanationSHAP.result.variable_name[importance_orderingsSHAP[:k_features]] # k most important variables in order (highest = most important)
#     target_curve = explanationSHAP.predicted_function
#     t_columns = [x for x in list(explanationSHAP.result.columns) if "t = " in x]
#     predicted_curve = explanationSHAP.baseline_function+np.sum(explanationSHAP.result.loc[importance_orderingsSHAP[:k_features],:])[t_columns]
#     # How to get top k explanations from survshap output
#     explanationSHAP.result["aggregated_change"] = np.absolute(explanationSHAP.result["aggregated_change"])
#     topkfeatures_SHAP = list(explanationSHAP.result.sort_values(by= "aggregated_change", ascending=False).variable_name[:k_features])
#     explanations["survSHAP"].append(topkfeatures_SHAP)
#     et = time.time()
#     elapsed_time = et - st # print('Execution time:', elapsed_time, 'seconds')
#     print("SurvSHAP: Explained instance ", instance_id, ": ", topkfeatures_SHAP,". Duration in seconds: ",elapsed_time, "\n")


#     # SVM EXPLANATIONS
#     st = time.time()
#         # Generate the perturbed data
#     scaled_data, inverse, weights = surv_utils.create_weighted_perturbed_data(explainer.train_x, instance, [0,1], binary, weigh_data=False, n=n_samples)
#         # Weigh the perturbed data by sampling with replacements and adding them to the perturbed dataset
#     add_perturbations = surv_utils.weigh_df(scaled_data, (n_perturbed-n_samples), weights)
#     scaled_data, inverse = pd.concat([scaled_data, add_perturbations]), pd.concat([inverse, inverse.loc[add_perturbations.index,:]])            
#         # Get predictions for perturbed data
#     surv_curves_pert_bb = explainer.blackboxmodel.predict_survival_function(inverse, return_array=True)
#         # Aggregate the predicted survival curves to achieve time-to-event estimates for perturbed data
#     predictions_pert = surv_utils.aggregate_surv_function(surv_curves_pert_bb, "median", threshold = 0.5, timestamps = explainer.time_stamps)
#         # Boost the predictions of the perturbed data (only for uncensored cases)
#         # TODO Tricky: what if after boosting some of those instance predicted to have dementia have a time to event estimate past 17 years?
# #    predictions_pert.loc[predictions_pert["event"]==True,"time"] = np.array(predictions_pert.loc[predictions_pert["event"]==True,"time"])- explainer.regr.predict(inverse.loc[predictions_pert["event"]==True,:])
#     x, y_lasso = surv_utils.convert_time_y_to_surv_y(df = predictions_pert, y = "time", event_indicator = "event", only_targets = False, round_y= False)  
#         # Computing shapley value
#     SVM_shapley_values = survivalSVM(scaled_data, y_lasso, finetune=True)
#         # Feature selection with SVM 
#     topkfeatures_SVM = SVM_shapley_values       
#     topkfeatures_SVM["shapley_values"] = np.absolute(topkfeatures_SVM["shapley_values"]) 
#     topkfeatures_SVM = list(topkfeatures_SVM.sort_values(by="shapley_values", ascending=False).feature_names[:k_features])
#     explanations["SVM"].append(topkfeatures_SVM)
#     et = time.time()
#     elapsed_time = et - st # print('Execution time:', elapsed_time, 'seconds')
#     print("SVM: Explained instance ", instance_id, ": ", topkfeatures_SVM,". Duration in seconds: ",elapsed_time, "\n")


    # ChoquEx EXPLANATIONS
    st = time.time()
        # Generate the perturbed data
    scaled_data, inverse, weights = surv_utils.create_weighted_perturbed_data(explainer.train_x, instance, [0,1], binary, weigh_data=False, n=n_samples)
        # Weigh the perturbed data by sampling with replacements and adding them to the perturbed dataset
    add_perturbations = surv_utils.weigh_df(scaled_data, (n_perturbed-n_samples), weights)
    scaled_data, inverse = pd.concat([scaled_data, add_perturbations]), pd.concat([inverse, inverse.loc[add_perturbations.index,:]])            
        # Get predictions for perturbed data
    surv_curves_pert_bb = explainer.blackboxmodel.predict_survival_function(inverse, return_array=True)
        # Aggregate the predicted survival curves to achieve time-to-event estimates for perturbed data
    predictions_pert = surv_utils.aggregate_surv_function(surv_curves_pert_bb, "median", threshold = 0.5, timestamps = explainer.time_stamps)
        # Boost the predictions of the perturbed data (only for uncensored cases)
        # TODO Tricky: what if after boosting some of those instance predicted to have dementia have a time to event estimate past 17 years?
#    predictions_pert.loc[predictions_pert["event"]==True,"time"] = np.array(predictions_pert.loc[predictions_pert["event"]==True,"time"])- explainer.regr.predict(inverse.loc[predictions_pert["event"]==True,:])
    x, y_lasso = surv_utils.convert_time_y_to_surv_y(df = predictions_pert, y = "time", event_indicator = "event", only_targets = False, round_y= False)  
        # Computing shapley value
    Choquet_shapley_values = survivalChoqEx(scaled_data, y_lasso, finetune=True)
        # Feature selection with SVM 
    topkfeatures_ChoquEx = Choquet_shapley_values       
    topkfeatures_ChoquEx["shapley_values"] = np.absolute(topkfeatures_ChoquEx["shapley_values"]) 
    topkfeatures_ChoquEx = list(topkfeatures_ChoquEx.sort_values(by="shapley_values", ascending=False).feature_names[:k_features])
    explanations["ChoquEx"].append(topkfeatures_ChoquEx)
    et = time.time()
    elapsed_time = et - st # print('Execution time:', elapsed_time, 'seconds')
    print("ChoquEx: Explained instance ", instance_id, ": ", topkfeatures_ChoquEx,". Duration in seconds: ",elapsed_time, "\n")



# survLIMEorderings = pd.DataFrame(explanations["survLIME"], columns=np.arange(1,6))
# survSHAPorderings = pd.DataFrame(explanations["survSHAP"], columns=np.arange(1,6))
# SVMorderings = pd.DataFrame(explanations["SVM"], columns=np.arange(1,6))
ChoquExorderings = pd.DataFrame(explanations["ChoquEx"], columns=np.arange(1,6))


# survLIMEorderings.to_csv("results/exp2_survLIME_n{}_binary_{}.csv".format(n_perturbed,binary))
# survSHAPorderings.to_csv("results/exp2_survSHAP.csv")
# SVMorderings.to_csv("results/exp2_SVSL_nsamples{}_ntotal{}_binary_{}.csv".format(n_samples,n_perturbed,binary))
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code')
ChoquExorderings.to_csv("results/exp2_ChoquEx_nsamples{}_ntotal{}_binary_{}.csv".format(n_samples,n_perturbed,binary))



# surv_utils.convert_data_for_plot(survLIMEorderings).to_csv("results/exp2_survLIME_n{}_binary_{}_PLOT.csv".format(n_perturbed,binary))
# surv_utils.convert_data_for_plot(survSHAPorderings).to_csv("results/exp2_survSHAP_PLOT.csv")
# surv_utils.convert_data_for_plot(SVMorderings).to_csv("results/exp2_SVSL_nsamples{}_ntotal{}_binary_{}_PLOT.csv".format(n_samples,n_perturbed,binary))
surv_utils.convert_data_for_plot(ChoquExorderings).to_csv("results/exp2_ChoquEx_nsamples{}_ntotal{}_binary_{}_PLOT.csv".format(n_samples,n_perturbed,binary))



survLIME_results = pd.read_csv("results/exp2_survLIME_n{}_binary_{}_PLOT.csv".format(n_perturbed,binary))
survSHAP_results = pd.read_csv("results/exp2_survSHAP_PLOT.csv")
SVSL_results = pd.read_csv("results/exp2_SVSL_nsamples{}_ntotal{}_binary_{}_PLOT.csv".format(n_samples,n_perturbed,binary))
ChoquEx_results = pd.read_csv("results/exp2_ChoquEx_nsamples{}_ntotal{}_binary_{}_PLOT.csv".format(n_samples,n_perturbed,binary))


# Prepare data for chi square test in R

true_ranking = list(ground_truth_global["feature"])
n_correct = {}
tools = [survLIME_results, survSHAP_results, SVSL_results, ChoquEx_results]
tools_str = ["survLIME", "survSHAP", "SVSL", "ChoquEx"]

for j in range(4):
    n_correct[tools_str[j]] = 0

    for i in range(5):
        mask1 = tools[j]["variable"] == true_ranking[i] 
        temp = tools[j][mask1]
        mask2 = temp["importance_ranking"] == i+1
        temp = temp[mask2]
        n_correct[tools_str[j]] += np.int(temp["value"])
        # print(i,j)

simulation_results_for_chi_square = pd.DataFrame()

tool = (["ChoquEx"]*500) + (["SVSL"]*500) + (["survSHAP"]*500) + (["survLIME"]*500)
classification = (["correct"]*n_correct["ChoquEx"]) + (["incorrect"]*(500-n_correct["ChoquEx"]))  + (["correct"]*n_correct["SVSL"]) + (["incorrect"]*(500-n_correct["SVSL"])) +  (["correct"]*n_correct["survSHAP"]) + (["incorrect"]*(500-n_correct["survSHAP"])) +  (["correct"]*n_correct["survLIME"]) + (["incorrect"]*(500-n_correct["survLIME"]))




simulation_results_for_chi_square["tool"] = tool
simulation_results_for_chi_square["classification"] = classification
simulation_results_for_chi_square.to_csv("results/exp2_simulation_results_for_chi_square.csv")


# Offtopic: preperation of survey data for R evaluation

