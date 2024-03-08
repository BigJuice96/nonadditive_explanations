import os 
os.chdir('/Users/abdallah/Desktop/Kings College Project/Code/GIT_Repo')
from explainer import ExplainSurvival
import surv_utils
from survshap.survshap import *
from survshap.survlime import *
from sksurv.ensemble import RandomSurvivalForest
import time
import pandas as pd
import numpy as np
import pickle



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
categ_features = [ 'scorg3','social',
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
training_x, training_y = surv_utils.convert_time_y_to_surv_y(df = training, y = "time", event_indicator = "dem_hse_w8", only_targets = False, round_y= False) 
test_x, test_y = surv_utils.convert_time_y_to_surv_y(df = test, y = "time", event_indicator = "dem_hse_w8", only_targets = False, round_y= False)

    # Fit a survival model that we will try to explain: a RSF
rsf = RandomSurvivalForest(
    n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=1
)
rsf.fit(training_x, training_y)


    # initialise explanations and timer dictionaries to save results
explanations=dict()
explanations["SurvMLeX"] = []
explanations["SurvLIME"] = []
explanations["SurvSHAP"] = []
explanations["SurvChoquEx"] = []
timer=dict()
timer["SurvMLeX"] = []
timer["SurvLIME"] = []
timer["SurvSHAP"] = [] 
timer["SurvChoquEx"] = []

    # initialise counter at 0
counter=0

    # Fit the explainer 
explainer_fidelity = ExplainSurvival(
                        model = rsf, # the model whose output we want to explain 
                        x = training_x, # DUMMIED data that was used to train the blackbox model
                        y = training_y, # the outcome data in a format suitable for sklearn survival models  
                        categorical_features = categ_features, # List of strings of the feature labels corresponding to categorical features
                        random_state = random_state, # random state for intilisation of random functions
                        )

    # Get global measures of feature importance
explainer_fidelity.global_feature_importance() #TODO check if this works



for instance_id in test_x.index:
    instance = np.array(test_x.loc[instance_id, :])
    counter +=1

    for tool in ["SurvMLeX", "SurvChoquEx", "SurvLIME", "SurvSHAP"]:
        # calculate the explanations for the survMLeX method 
        st = time.time()
        print("Explaining instance {} using {}.".format(counter, tool))
        explanation = explainer_fidelity.explain_instance( 
                            instance = instance , # Pass instance as a np.array or pd.series                         
                            method = tool, # Explanation method: either "SurvMLeX", "SurvChoquEx", "SurvLIME" or "SurvSHAP"
                            binary=False, # Using a binary representation for the perturbed samples. Default is False
                            aggregation_method = "percentile", # How to aggregate survival function predictions to a time-to-event estimate: "trapezoid", "median" or "percentile"
                            threshold_aggregation = 0.7, # Which threshold to use when aggregating survival curves to a time-to-event estimate using the percentile method (if it is not finetuned)
                            find_best_cutoff = False, # Finds optimal value for threshold_blackbox_clf argument
                            n_perturbed=1400, # Size of the perturbed dataset
                            weighing_multiplier = 1.5, # How much larger the perturbed dataset will be after weighing through sampling with replacement  
                            finetune_explainer=True, # Whether to finetune expainer (only for SurvMLeX, SurvChoquEx and SurvSHAP)
                            only_shapley_values=True, # Not relevant if method="SurvLIME". For other methods it wont use identified features to fit a Cox regression model but give the shapley values right away
                            k_features=10, # The number of features to retain if a penalised Cox regression is fit (only if only_shapley_values=False)
                            plot_curve=False, # Whether explainer should also plot survival curve (only if only_shapley_values=False)
                            alpha_min_ratio=0.01, # alpha_min_ratio argument for the penalised Cox regression (only if only_shapley_values=False)
                            alpha=1.0, # Alpha parameter of SurvMLeX or SurvChoquEx
                            survshap_calculation_method="sampling" # for survshap method: can be either "kernel", "sampling", "treeshap" or "shap_kernel"
                            ) 
        if tool == "SurvLIME":
            explanations[tool].append(list(explanation["hazard_ratios"].index))
        else:
            explanations[tool].append(list(explanation["shapley_values"]["feature_names"]))
        et = time.time()
        elapsed_time = et - st # print('Execution time:', elapsed_time, 'seconds')
        timer[tool].append(elapsed_time)
        if tool == "SurvLIME":
            print("SurvLIME explanation: ", list(explanation["hazard_ratios"].index))        
        else:
            print("{} explanation: ".format(tool, list(explanation["shapley_values"]["feature_names"])))
        print("{} needed {} seconds.\n".format(tool, elapsed_time))
    print("counter: ", counter, " instances explained.\n\n\n")    


