from explainer import ExplainSurvival
import surv_utils
import pandas as pd
import numpy as np
import pickle
from sksurv.ensemble import RandomSurvivalForest
from survlimepy.utils.neighbours_generator import NeighboursGenerator




with open('data/cat_feature_dict.pkl', 'rb') as fp:
    cat_feature_dict = pickle.load(fp)
data_w1_dummied = pd.read_csv("data/data_w1_dummied.csv", index_col=0)
data_w1_dummied_unstandardised = pd.read_csv("data/data_w1_dummied_unstandardised.csv", index_col=0)
variable_definitions = pd.read_csv("data/variable_definitions.csv", index_col=0)
data_w1 = pd.read_csv("data/data_w1.csv", index_col=0)

categ_features = [ 'scorg3','social',
                          'ethnicity_White', 'gender_Men', 
                        'edqual_Degree or equiv', 'edqual_Foreign/other',
                        'edqual_NVQ/GCE O/A Level equiv', 'edqual_No qualification', 
                          ]

selected_features = ['scorg3', #'social',
                         "cesd_sc",'n_chronic_diseases',"n_neuropsychiatric_symptoms",
                          'ethnicity_White', 'gender_Men', 'mmgsd_w2', 
                        'edqual_Degree or equiv', 'edqual_Foreign/other',
                        'edqual_NVQ/GCE O/A Level equiv', 'edqual_No qualification', 
                         'memory1', 'procspeed1', # left out 'exe1',  as it correlates strongly (0.46) with memory1 and is a similar construct
                          "age", "dem_hse_w8", "time"]

    # random seed


from sklearn.utils import check_random_state
random_state=1
random_state = check_random_state(random_state)


    # use 100 instances for testing, the rest for training
test_indices = data_w1_dummied.sample(100, random_state=random_state).index
training_indices = [x for x in data_w1_dummied.index if x not in test_indices]
training = data_w1_dummied.loc[training_indices, selected_features]
test = data_w1_dummied.loc[test_indices, selected_features]


    # Split into predictors and targets
training_x, training_y = surv_utils.convert_time_y_to_surv_y(df = training, y = "time", event_indicator = "dem_hse_w8", only_targets = False, round_y= False) 
test_x, test_y = surv_utils.convert_time_y_to_surv_y(df = test, y = "time", event_indicator = "dem_hse_w8", only_targets = False, round_y= False)


    # Fit a survival model that whose predictions we will try to explain: a RSF
rsf = RandomSurvivalForest(
    n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=random_state
)
rsf.fit(training_x, training_y)


instance_to_explain = training_x.iloc[0] # Define which instance we want to explain


    # Generate perturbed dataset in the neighbourhood of the instance to be explained
perturbed_data = surv_utils.generate_neighbourhood(
    training_features = training_x,
    data_row = instance_to_explain
    )[0]

######### Get the time-to-event predictions for the neighbours using from the bb model 


optimal_threshold = surv_utils.find_optimal_threshold(training_x, training_y, rsf, rsf.unique_times_)
targets_opt = surv_utils.create_survival_targets(rsf, perturbed_data, rsf.unique_times_, training_y, threshold = optimal_threshold)
targets_50 = surv_utils.create_survival_targets(rsf, perturbed_data, rsf.unique_times_, training_y, threshold = 0.5)


sum(targets_opt["event"])
sum(targets_50["event"])

