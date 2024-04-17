from falldetection.time_series_extractor_workflow import extract_time_series
from falldetection.sensor import Sensor

X_raw, y_raw = extract_time_series(
    sensor=Sensor.RIGHT_THIGH,
    baseDir=baseDir,
    columns=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])
# columns=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
import pandas as pd

def train_predict_multiple(classifiers, X_train, y_train, X_test, y_test, scorer):
    def train_predict(classifier):
        classifier = classifier.fit(X_train, y_train)
        return scorer(y_test, classifier.predict(X_test))

    results = {'classifier': list(classifiers.keys()),
               'score': list(map(train_predict, classifiers.values()))}
    return pd.DataFrame(results).sort_values(by='score', ascending=False)

def plot_training_results(df):
    ax = df.plot.bar(x='classifier', y='score', legend=False, rot=0)
    ax.set_ylabel('score')
    ax.axhline(y=benchmark_score, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
  import visuals as vs

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

classifiers = {'SVC': SVC(random_state=815),
               'DecisionTree': DecisionTreeClassifier(random_state=815),
               'RandomForest': RandomForestClassifier(random_state=815),
               'KNeighbors': KNeighborsClassifier(n_neighbors=3)}

df = train_predict_multiple(classifiers,
                            X_feature_train,
                            y_feature_train,
                            X_feature_test,
                            y_feature_test,
                            scorer=accuracy_score)

plot_training_results(df)
plt.hist(df,color = 'r')
plt.savefig('images/scoresByClassifier.png')
display(df)
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from tensorflow import set_random_seed
np.random.seed(815)
set_random_seed(815)
data_dim = X_raw.shape[2] # = number of features = [{Acc_x, Acc_y, Acc_z, Gyr_*, Max_*, }] = 9
timesteps = X_raw.shape[1] # = 101 = (half_window_size:=50) * 2 + 1
num_classes = 1 # fall?
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
