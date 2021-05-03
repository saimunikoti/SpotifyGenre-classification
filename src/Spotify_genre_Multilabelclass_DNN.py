# use mlp for prediction on multi-label classification
from numpy import asarray
from sklearn.datasets import make_multilabel_classification
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

## build model
# get the dataset
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.layers import MaxPooling1D
# get the model
def get_model():
	model = Sequential()
	model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(11,1)))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	# model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(20, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(17, kernel_initializer='he_uniform', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

model = get_model()

# load dataset

n_inputs, n_outputs = X_train.shape[1], y_train.shape[1]

# get model

model.summary()
# fit the model on all data
history = model.fit(X_train, y_train, verbose=2, epochs=200)
# make a prediction for new data

y_pred = model.predict(X_test)
# print('Predicted: %s' % yhat[0])

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
y_pred[y_pred>=0.5]= 1
y_pred[y_pred<0.5]= 0

def get_multilabel_accuracy(y_test, y_pred):
	acc=[]
	acc = [ accuracy_score(y_test[ind], y_pred[ind]) for ind in range(y_test.shape[0]) ]
	return np.array(acc)

Accuracylist = get_multilabel_accuracy(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

print(np.mean(Accuracylist))