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
ground_truth_global = explainer_simulation.global_feature_importance(n_repeats=15) # The two variables from the interaction are most important
ground_truth_global
    #Save the explanations object somewhere so I dont ever have to run this again
with open('data/ground_truth_global{}.pkl'.format(""), 'wb') as fp:
    pickle.dump(ground_truth_global, fp)
    print('timer dictionary saved successfully to file')
    # Can start from here again
with open('data/ground_truth_global{}.pkl'.format(""), 'rb') as fp:
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
PVI_confidence_intervals.to_csv("Results/PVI_confidence_intervals{}.csv".format(""))
    # Use the R file to plot the CIs



    # Specify number of features to consider + number of explanations
N_test_simulation = 10
Batch_size_simulation = 5
k_features_simulation = len(data.drop(["time","event"], axis=1).columns) # include all features in the original dataset
n_perturbed_simulation = 1400 # Size of the perturbed dataset
explanations_simulation = dict()
timer_simulation = dict()
tools = ["SurvMLeX", 
         "SurvLIME", 
         "SurvSHAP", 
         "SurvChoquEx"]
for tool in tools:
    explanations_simulation[tool] = list()
    timer_simulation[tool] = []




###################

instance = data_x.iloc[0,:]
n_perturbed = 1000
kernel_width = None
finetune_explainer = True
alpha = 1.0 

explainer_simulation.perturbed_data = surv_utils.generate_neighbourhood(
    training_features = explainer_simulation.x,
    data_row = instance,
    random_state = explainer_simulation.random_state,
    num_samples=n_perturbed,
    kernel_width = kernel_width
    )[0]

    # Get the time-to-event predictions for the neighbours using from the bb model 
threshold_aggregation = surv_utils.find_optimal_threshold(explainer_simulation.x, explainer_simulation.y, explainer_simulation.model, explainer_simulation.model.unique_times_)
explainer_simulation.perturbed_data_targets = surv_utils.create_survival_targets(explainer_simulation.model, explainer_simulation.perturbed_data, explainer_simulation.model.unique_times_, explainer_simulation.y, threshold = threshold_aggregation)
    # Feature selection with SVM 
explainer_simulation.perturbed_survival_curves = explainer_simulation.model.predict_survival_function(explainer_simulation.perturbed_data, return_array=True)
explainer_simulation.perturbed_data = pd.DataFrame(explainer_simulation.perturbed_data, columns=explainer_simulation.x.columns)

method = "SurvMLeX"

from nonadditive_explainers import nonadditive_explainer




data_x = explainer_simulation.perturbed_data
y = explainer_simulation.perturbed_data_targets
cat_features = explainer_simulation.categorical_features
alpha=1.0
finetune=True
rank_ratio=1.0
fit_intercept=False
n_shuffle_splits=5
random_state=None
range_alpha= (2.0 ** np.arange(-8, 7, 2))


set_config(display="text")  # displays text representation of estimators
sns.set_style("whitegrid")  


data_x = pd.DataFrame(data_x)
print("encode_categorical")
x = encode_categorical(pd.DataFrame(data_x), columns=cat_features)
x_normalized = normalize(x.values, norm='l2')
if method == "SurvMLeX":
    print("Shapley_kernel")
    kernel_matrix = Shapley_kernel(x_normalized,x_normalized) #clinical_kernel(data_x) 
elif method == "SurvChoquEx":
    print("ChoquetKernel")
    kernel_matrix = ChoquetKernel(x_normalized) # This line is different 
    kernel_matrix = kernel_matrix.get_kernel()
else: 
    assert("No method specified: can be either SurvMLeX or SurvChoquEx")

param_grid = {"alpha": range_alpha}
print("ShuffleSplit")
cv = ShuffleSplit(n_splits=n_shuffle_splits, test_size=0.5, random_state=random_state)
kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0, rank_ratio=rank_ratio, fit_intercept=fit_intercept)
if finetune==True:
    print("GridSearchCV")
    kgcv = GridSearchCV(kssvm, param_grid, scoring=score_survival_model, n_jobs=-1, refit=False, cv=cv)
    print("kgcv.fit(kernel_matrix, y)")
    kgcv = kgcv.fit(kernel_matrix, y)
    alpha = kgcv.best_params_["alpha"]
print("FastKernelSurvivalSVM")
kssvm = FastKernelSurvivalSVM(alpha=alpha, optimizer="rbtree", kernel="precomputed", random_state=0, rank_ratio=rank_ratio, fit_intercept=fit_intercept)
kssvm = kssvm.fit(kernel_matrix, y)    



X_full = x_normalized
alpha_hat_eta = kssvm.coef_
sv_ind = np.arange(0,data_x.shape[0])
q_additivity = None
method_shap='dp'
feature_type = 'numerical'


X = X_full[sv_ind,:]
n, d = X.shape
if q_additivity is None: 
    q_additivity = d

val = np.zeros((d,))


######### Dynamic programming approach for computing Shapley value
# TODO ERROR MUST BE HERE SOMEWHERE, IT IS ACTUALLY A LOOP

    ############################
from joblib import Parallel, delayed
import os


def temp_func(X_arg, i, q_additivity_arg, feature_type_arg, alpha_hat_eta_arg):
    omega_dp, dp = Omega(X_arg,i, q_additivity_arg, feature_type_arg)
    val[i] = np.inner(omega_dp, alpha_hat_eta_arg)
    return 

val = Parallel(n_jobs=8)(
    delayed(temp_func)(X,i, q_additivity, feature_type, alpha_hat_eta) for i in range(d)
)



def parallel_compute_shap_values(X, q_additivity, feature_type, alpha_hat_eta):
    d = X.shape[1]
    
    # Define a helper that computes the value for a single feature index.
    def compute_val(i):
        # Compute Omega for feature i (and dp, if needed)
        omega_dp, dp = Omega(X, i, q_additivity=q_additivity, feature_type=feature_type)
        return np.inner(omega_dp, alpha_hat_eta)
    
    # Use Parallel to execute compute_val(i) for each i in parallel.
    results = Parallel(n_jobs=os.cpu_count()-4)(
        delayed(compute_val)(i) for i in range(d)
    )
    
    # Convert the list of results into a NumPy array.
    val = np.array(results)
    return val

val = parallel_compute_shap_values(X, q_additivity, feature_type, alpha_hat_eta)

 ####################
for i in range(d):
    print(i, " from ", d)
    omega_dp, dp = Omega(X,i, q_additivity, feature_type=feature_type)
    val[i] = np.inner(omega_dp, alpha_hat_eta)
    



# LOOP START
#for i in range(d):
i = 0

n, d = X.shape
if q_additivity == None: 
    q_additivity = d


idx = np.arange(d)
idx[i] = 0
idx[0] = i
X = X[:,idx]


if feature_type == 'binary':
    print("binary feature type")
    omega = np.zeros((n,))
    ind_nonzeros = np.where(X[:,0] > 0)[0].tolist()
    print(1)
    for i in ind_nonzeros:
        xi_ones = np.where(X[i,1:] > 0)[0].tolist()
        xi_ones_count = len(xi_ones)
        temp = 0
        print(2)
        for j in range(1,q_additivity):
            temp += (1 / (j+1)) * (math.comb(xi_ones_count,j)) 
            print(3)
        
        omega[i] = temp 
    omega[ind_nonzeros] = (1 + omega[ind_nonzeros])
    omega_dp, dp = omega, None
    
dp = np.zeros((q_additivity, d, n))
    
# To store the answer for
# current value of k
sum_current = np.zeros((n,))


# For k = 1, the answer will simply
# be the sum of all the elements
for i in range(d):
    dp[0,i,:] = X[:,i]
    sum_current += X[:,i]


# Filling the dp table in bottom up manner
for i in range(1 , q_additivity):
    # The sum of the row to be used for calculating the values in the next row
    temp_sum = np.zeros((n,))

    for j in range( 0,  d):
        print(1)

        # We will subtract previously computed value
        # so as to get the sum of elements from j + 1
        # to n in the (i - 1)th row
        sum_current -= dp[i - 1,j,:]
        print(2)

        dp[i,j,:] = (i / (i+1)) * X[:,j] * sum_current #((i-1) / i) * arr[j - 1] * sum_current
        temp_sum += dp[i,j,:]
        print(3)
    sum_current = temp_sum

omega = np.sum(dp[:,0,:],axis=0)
omega_dp, dp = omega, dp

val[i] = np.inner(omega_dp, alpha_hat_eta)

# LOOP END


## Computing shapley value
if method == "SurvMLeX":
    shapley_values = nonadditive_explainer(explainer_simulation.perturbed_data, explainer_simulation.perturbed_data_targets, explainer_simulation.categorical_features, method = "SurvMLeX", alpha=alpha, finetune=finetune_explainer, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=explainer_simulation.random_state, range_alpha= (2.0 ** np.arange(-8, 7, 2)))
elif method == "SurvChoquEx":
    shapley_values = nonadditive_explainer(explainer_simulation.perturbed_data, explainer_simulation.perturbed_data_targets, explainer_simulation.categorical_features, method = "SurvChoquEx", alpha=alpha, finetune=finetune_explainer, rank_ratio=1.0, fit_intercept=False, n_shuffle_splits=5, random_state=explainer_simulation.random_state, range_alpha= (2.0 ** np.arange(-8, 7, 2)))


###################


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

timer_simulation_old # TODO compare 
timer_simulation_old["SurvMLeX"]


np.mean(timer_simulation_old["SurvMLeX"])
np.mean(timer_simulation["SurvMLeX"])
np.mean(timer_simulation_old["SurvChoquEx"])
np.mean(timer_simulation["SurvChoquEx"])
np.mean(timer_simulation_old["SurvSHAP"])
np.mean(timer_simulation["SurvSHAP"])
np.mean(timer_simulation_old["SurvLIME"])
np.mean(timer_simulation["SurvLIME"])

np.median(timer_simulation_old["SurvMLeX"])
np.median(timer_simulation["SurvMLeX"])
np.median(timer_simulation_old["SurvChoquEx"])
np.median(timer_simulation["SurvChoquEx"])
np.median(timer_simulation_old["SurvSHAP"])
np.median(timer_simulation["SurvSHAP"])
np.median(timer_simulation_old["SurvLIME"])
np.median(timer_simulation["SurvLIME"])






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

