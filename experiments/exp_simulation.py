import pandas as pd
import numpy as np
from explainer import ExplainSurvival
from datetime import datetime
import matplotlib.pyplot as plt
import surv_utils
import time
from sksurv.ensemble import RandomSurvivalForest
import pickle

# from survshap.survshap import *
# from survshap.survshap.predict_explanations import *
# from survshap.survlime import *


# Load data and train model
    # generated using script in data_generation.R
datetime_identifier = datetime.now().strftime("_%d_%m_%Y_%H:%M:%S")
    # Random seed always at 1
random_state = 1
    # Import survival data
data = pd.read_csv("experiments/synthetic_data.csv") # Already dummied
    # Save names of categorical features (needed as an argument for the explainer) AFTER one-hot-encoding 
cat_features = ['x1', 'x2'] # x1 and x2 are binary
    # Train (for instance) a random survival forest
rsf = RandomSurvivalForest(
    n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=1
)
    # Split into training samples and survival targets
data_x, data_y = surv_utils.convert_time_y_to_surv_y(df = data, y = "time", event_indicator = "event", only_targets = False, round_y= False) 
rsf.fit(data_x, data_y)

    # Create an explainer object
explainer_simulation = ExplainSurvival(
                        model = rsf, # the model whose output we want to explain 
                        x = data_x, # DUMMIED data that was used to train the blackbox model
                        y = data_y, # the outcome data in a format suitable for sklearn survival models  
                        categorical_features = cat_features, # List of strings of the feature labels corresponding to categorical features
                        random_state = random_state, # random state for intilisation of random functions
                        )


# Histogram of the distribution of the time variable 
# Print that again when done and save it
plt.hist(np.array(data["time"]))
plt.show()

# Calculate the ground truth 
ground_truth_global = explainer_simulation.global_feature_importance(15) # The two variables from the interaction are most important
ground_truth_global
    #Save the explanations object somewhere so I dont ever have to run this again
with open('data/ground_truth_global{}.pkl'.format(datetime_identifier), 'wb') as fp:
    pickle.dump(ground_truth_global, fp)
    print('timer dictionary saved successfully to file')
    # Can start from here again
with open('data/ground_truth_global{}.pkl'.format(datetime_identifier), 'rb') as fp:
    ground_truth_global = pickle.load(fp)

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
PVI_confidence_intervals.to_csv("Results/PVI_confidence_intervals{}.csv".format(datetime_identifier))
    # Use the R file to plot the CIs



    # Specify number of features to consider + number of explanations
N_test_simulation = 30
k_features_simulation = len(data.drop(["time","event"], axis=1).columns) # include all features in the original dataset
n_perturbed_simulation = 1400 # Size of the perturbed dataset
explanations_simulation = dict()
timer_simulation = dict()
tools = ["SurvMLeX", 
        #  "SurvLIME", 
        #  "SurvSHAP", 
         "SurvChoquEx"]
for tool in tools:
    explanations_simulation[tool] = list()
    timer_simulation[tool] = []



counter = 0
for instance_id in data_x.index[:N_test_simulation]:
    instance = np.array(data_x.loc[instance_id, :])
    counter +=1
    for tool in tools: # change to tools 
        # calculate the explanations for the survMLeX method 
        st = time.time()
        print("Explaining instance {} using {}.".format(counter, tool))
        explanation = explainer_simulation.explain_instance( 
                            instance = instance , # Pass instance as a np.array or pd.series                         
                            method = tool, # Explanation method: either "SurvMLeX", "SurvChoquEx", "SurvLIME" or "SurvSHAP"
                            threshold_aggregation = 0.7, # Which threshold to use when aggregating survival curves to a time-to-event estimate using the percentile method (if it is not finetuned)
                            find_best_cutoff = True, # Finds optimal value for threshold_blackbox_clf argument
                            n_perturbed=n_perturbed_simulation, # Size of the perturbed dataset
                            finetune_explainer=True, # Whether to finetune expainer (only for SurvMLeX, SurvChoquEx and SurvSHAP)
                            only_shapley_values=True, # Not relevant if method="SurvLIME". For other methods it wont use identified features to fit a Cox regression model but give the shapley values right away
                            k_features=k_features_simulation, # The number of features to retain if a penalised Cox regression is fit (only if only_shapley_values=False)
                            plot_curve=False, # Whether explainer should also plot survival curve (only if only_shapley_values=False)
                            alpha_min_ratio=0.01, # alpha_min_ratio argument for the penalised Cox regression (only if only_shapley_values=False)
                            alpha=1.0, # Alpha parameter of SurvMLeX or SurvChoquEx
                            survshap_calculation_method="sampling" # for survshap method: can be either "kernel", "sampling", "treeshap" or "shap_kernel"
                            ) 
        if tool == "SurvLIME":
            explanations_simulation[tool].append(list(explanation["hazard_ratios"].index))
        else:
            explanations_simulation[tool].append(list(explanation["shapley_values"]["feature_names"]))
        et = time.time()
        elapsed_time = et - st # print('Execution time:', elapsed_time, 'seconds')
        timer_simulation[tool].append(elapsed_time)
        if tool == "SurvLIME":
            print("SurvLIME explanation: ", list(explanation["hazard_ratios"].index))        
        else:
            print("{} explanation: ".format(tool, list(explanation["shapley_values"]["feature_names"])))
        print("{} needed {} seconds.\n".format(tool, elapsed_time))
    print("counter: ", counter, " instances explained.\n\n\n")    



    #Save the explanations object somewhere so I dont ever have to run this again
with open('data/explanations_simulation{}.pkl'.format(datetime_identifier), 'wb') as fp:
    pickle.dump(explanations_simulation, fp)
    print('explanations dictionary saved successfully to file')
with open('data/timer_simulation{}.pkl'.format(datetime_identifier), 'wb') as fp:
    pickle.dump(timer_simulation, fp)
    print('timer dictionary saved successfully to file')

    # Can start from here again
with open('data/explanations_simulation{}.pkl'.format(datetime_identifier), 'rb') as fp:
    explanations_simulation = pickle.load(fp)
with open('data/timer_simulation{}.pkl'.format(datetime_identifier), 'rb') as fp:
    timer_simulation = pickle.load(fp)


SurvMLeXorderings = pd.DataFrame(explanations_simulation["SurvMLeX"], columns=np.arange(1,6))
SurvLIMEorderings = pd.DataFrame(explanations_simulation["SurvLIME"], columns=np.arange(1,6))
for i in range(N_test_simulation):
    explanations_simulation["SurvSHAP"][i] = explanations_simulation["SurvSHAP"][i][:k_features_simulation]
SurvSHAPorderings = pd.DataFrame(explanations_simulation["SurvSHAP"], columns=np.arange(1,6))
SurvChoquExorderings = pd.DataFrame(explanations_simulation["SurvChoquEx"], columns=np.arange(1,6))

SurvMLeXorderings.to_csv("Results/exp2_SurvMLeXorderings{}.csv".format(datetime_identifier))
SurvLIMEorderings.to_csv("Results/exp2_SurvLIMEorderings{}.csv".format(datetime_identifier))
SurvSHAPorderings.to_csv("Results/exp2_SurvSHAPorderings{}.csv".format(datetime_identifier))
SurvChoquExorderings.to_csv("rResults/exp2_SurvChoquExorderings{}.csv".format(datetime_identifier))

surv_utils.convert_data_for_plot(SurvMLeXorderings).to_csv("Results/exp2_SurvMLeXorderings_PLOT{}.csv".format(datetime_identifier))
surv_utils.convert_data_for_plot(SurvLIMEorderings).to_csv("Results/exp2_SurvLIMEorderings_PLOT{}.csv".format(datetime_identifier))
surv_utils.convert_data_for_plot(SurvSHAPorderings).to_csv("Results/exp2_SurvSHAPorderings_PLOT{}.csv".format(datetime_identifier))
surv_utils.convert_data_for_plot(SurvChoquExorderings).to_csv("Results/exp2_SurvChoquExorderings_PLOT{}.csv".format(datetime_identifier))

    # Can start from here again 
SurvMLeXorderings = pd.read_csv("Results/exp2_SurvMLeXorderings{}.csv".format(datetime_identifier))
SurvLIMEorderings = pd.read_csv("Results/exp2_SurvLIMEorderings{}.csv".format(datetime_identifier))
SurvSHAPorderings = pd.read_csv("Results/exp2_SurvSHAPorderings{}.csv".format(datetime_identifier))
SurvChoquExorderings = pd.read_csv("Results/exp2_SurvChoquExorderings{}.csv".format(datetime_identifier))



# Prepare data for chi square test in R  #TODO test this part again 
true_ranking = list(ground_truth_global["feature"])
n_correct = {}
tools_str = [SurvMLeXorderings, SurvLIMEorderings, SurvSHAPorderings, SurvChoquExorderings]

for j in range(4):
    n_correct[tools[j]] = 0

    for i in range(5):
        mask1 = tools_str[j]["variable"] == true_ranking[i] 
        temp = tools_str[j][mask1]
        mask2 = temp["importance_ranking"] == i+1
        temp = temp[mask2]
        n_correct[tools[j]] += np.int(temp["value"])
        # print(i,j)

tools_str[j]["variable"]

simulation_results_for_chi_square = pd.DataFrame()
tool = (["SurvMLeX"]*500) + (["SurvLIME"]*500) + (["SurvSHAP"]*500) + (["SurvChoquEx"]*500)
classification = (["correct"]*n_correct["SurvMLeX"]) + (["incorrect"]*(500-n_correct["SurvMLeX"]))  + (["correct"]*n_correct["SurvLIME"]) + (["incorrect"]*(500-n_correct["SurvLIME"])) +  (["correct"]*n_correct["SurvSHAP"]) + (["incorrect"]*(500-n_correct["SurvSHAP"])) +  (["correct"]*n_correct["SurvChoquEx"]) + (["incorrect"]*(500-n_correct["SurvChoquEx"]))

simulation_results_for_chi_square["tool"] = tool
simulation_results_for_chi_square["classification"] = classification
simulation_results_for_chi_square.to_csv("/Users/abdallah/Desktop/Kings College Project/Code/Results/exp2_chi_square{}.csv".format(datetime_identifier)) 
    # Now refer to R file for the simulation evaluation (chi-square test)

