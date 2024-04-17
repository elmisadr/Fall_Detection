#sensor on right thigh and waist
from falldetection.sensor import Sensor

csv_file_by_sensor = {Sensor.RIGHT_THIGH: 'data/features_right_thigh.csv', Sensor.WAIST: 'data/features_waist.csv'}
csv_file = csv_file_by_sensor[Sensor.RIGHT_THIGH]

import os
from falldetection.feature_extractor_workflow import extract_features_and_save, FeatureExtractorWorkflow

def create_extracted_features_file(csv_file, sensor):
    if not os.path.isfile(csv_file):
        extract_features_and_save(sensor=sensor,
                                  baseDir=baseDir,
                                  csv_file=csv_file,
                                  autocovar_num=11,
                                  dft_amplitudes_num=0)

create_extracted_features_file(csv_file_by_sensor[Sensor.RIGHT_THIGH], Sensor.RIGHT_THIGH)
create_extracted_features_file(csv_file_by_sensor[Sensor.WAIST], Sensor.WAIST)
import numpy as np
import pandas as pd
from time import time
from IPython.display import display

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

%matplotlib inline

data = pd.read_csv(csv_file, index_col=0)

# Success - Display the first record
display(data.head(n=30))    #n=as many as needed
# TODO: Total number of records
n_records = len(data)

n_fall = len(data[data['fall'] == True])
n_adl = len(data[data['fall'] == False])

def fall_percent(y):
    return (len(y[y == True]) / len(y)) * 100    

# Print the results
print("Total number of records: {}".format(n_records))
print("Number of falls: {}".format(n_fall))
print("Number of Activities of Daily Life: {}".format(n_adl))
print("Percentage of falls: {:.2f}%".format(fall_percent(data['fall'])))
print("Percentage of Activities of Daily Life: {:.2f}%".format(100 - fall_percent(data['fall'])))

y_feature = data['fall']
X_feature = data.drop(columns=['fall', 'sensorFile'])
X_feature.head()
#information of the data set
from sklearn.model_selection import train_test_split

X_feature_train, X_feature_test, y_feature_train, y_feature_test = train_test_split(X_feature, 
                                                                                    y_feature, 
                                                                                    test_size=0.2, 
                                                                                    random_state=815)

# Show the results of the split
print("Training set has {} samples.".format(X_feature_train.shape[0]))
print("Testing set has {} samples.".format(X_feature_test.shape[0]))
    
print("Percentage of train falls: {:.2f}%".format(fall_percent(y_feature_train)))
print("Percentage of test falls: {:.2f}%".format(fall_percent(y_feature_test)))
always_fall_prediction = np.array([True]*len(X_feature_test))
always_adl_prediction = np.array([False]*len(X_feature_test))
from sklearn.metrics import accuracy_score

random_prediction = np.random.choice([True, False], len(X_feature_test))

print("accuracy_score(always_fall_prediction): {0:.0f}%".format(accuracy_score(y_feature_test, always_fall_prediction) * 100))
print("accuracy_score(always_adl_prediction): {0:.0f}%".format(accuracy_score(y_feature_test, always_adl_prediction) * 100))
print("accuracy_score(random_prediction): {0:.0f}%".format(accuracy_score(y_feature_test, random_prediction) * 100))
