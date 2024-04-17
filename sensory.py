from sklearn import tree
import numpy as np
import graphviz

baseDir='data/FallDataSet' #Directory for data set

def export_decision_tree(clf, feature_names):
    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=feature_names,
                                    class_names=['not survived', 'survived'],
                                    filled=True)
    return graphviz.Source(dot_data)
  from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from falldetection.feature_extractor import FeatureExtractor 

df = FeatureExtractor.sensor_file_2_df('/media/shailendra/New Volume/Master_Data_Science_Oslo_met/Second_Sem/Machine_learning/Final_project/newProject/data/FallDataSet/101/Testler Export/901/Test_1/340535.txt')
df.head()
from falldetection.sensor import Sensor 
from falldetection.sensor_files_provider import SensorFilesProvider
from falldetection.sensor_files_to_exclude import get_sensor_files_to_exclude_for
from falldetection.fall_predicate import isFall

def get_sensor_files(baseDir, sensor, sensor_file_filter):
    def get_sensor_files():
        return createSensorFilesProvider().provide_sensor_files()

    def createSensorFilesProvider():
        return SensorFilesProvider(baseDir, sensor, get_sensor_files_to_exclude_for(sensor))

    return list(filter(sensor_file_filter, get_sensor_files()))

fall_sensor_files = get_sensor_files(baseDir, Sensor.RIGHT_THIGH, isFall)
non_fall_sensor_files = get_sensor_files(baseDir, Sensor.RIGHT_THIGH, lambda sensor_file: not isFall(sensor_file))
# plotting different activies like fall/ non fall and so on

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from falldetection.feature_extractor import FeatureExtractor

def get_total_acceleration(df):
    return np.sqrt(df['Acc_X'] ** 2 + df['Acc_Y'] ** 2 + df['Acc_Z'] ** 2)

def reindex(series):
    return pd.Series(series.values, name='total_acceleration', index=pd.Index(np.arange(-50, 51, 1)))

def get_reindexed_total_acceleration(sensor_file):
    return reindex(get_total_acceleration(FeatureExtractor.sensor_file_2_df(sensor_file)))

def get_total_accelerations(sensor_files):
    return list(map(get_reindexed_total_acceleration, sensor_files))

def set_labels():
    plt.xlabel('time (s)')  # ('time (1/25 s)')
    plt.ylabel('total acceleration ($m/s^2$)')    

def plot_total_accelerations(sensor_files):
    total_accelerations = get_total_accelerations(sensor_files)
    ax = sns.lineplot(data=total_accelerations, ci='sd', legend=False)
    set_labels()
    
def plot_individual_total_accelerations(sensor_files):    
    def set_individual_names(total_accelerations):
        for counter, total_acceleration in enumerate(total_accelerations):
            total_acceleration.name = 'total_acceleration' + str(counter)
        
    total_accelerations = get_total_accelerations(sensor_files)
    set_individual_names(total_accelerations)
    ax = sns.lineplot(data=total_accelerations, dashes=False, legend=False)
    set_labels()
  plot_individual_total_accelerations(fall_sensor_files[:5])
plt.show()
plt.savefig('/media/shailendra/New Volume/Master_Data_Science_Oslo_met/Second_Sem/Machine_learning/Final_project/newProject/some_fall_total_accelerations.png')
plot_individual_total_accelerations(non_fall_sensor_files[:5])
plt.savefig('images/some_non_fall_total_accelerations.png')
plot_total_accelerations(fall_sensor_files)
plt.savefig('images/fall_total_accelerations.png')
plot_total_accelerations(non_fall_sensor_files)
plt.savefig('images/non_fall_total_accelerations.png')
