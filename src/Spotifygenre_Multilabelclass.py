import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.problem_transform import ClassifierChain

filepath = r"C:\Users\saimunikoti\stellargraph-datasets\data_by_artist_o.csv"
datadf = pd.read_csv(filepath)

datadf.info()
datadf.shape

df = datadf[datadf.genres != "[]" ]
df = df.dropna(how="any")
df = df.reset_index(drop=True)

basegenre = ["rock","comic","pop","rap","hip","trap","folk","soul","blues","invasion","jazz","band","mellow","urban",
             "funk","adult","others"]

## aggregate genre
for countrow in range(df.shape[0]):
    genrelist = df.iloc[countrow, 0]
    genrelist = genrelist[1:-1]
    wordlist = genrelist.split(", ")
    Newlist=[]
    for countword in range(len(wordlist)):
        elementlist = wordlist[countword][1:-1].split(" ")
        flag = 0
        for countelelem in range(len(elementlist)):
            if elementlist[countelelem] in basegenre:
                Newlist.append(elementlist[countelelem])
                flag =1
                break

        if flag ==0:
            Newlist.append("others")

    Newlist = np.unique(Newlist)
    df.loc[countrow,'newgenres'] = str(Newlist)[1:-1]

filepath = r"C:\Users\saimunikoti\stellargraph-datasets\data_by_artist_processed.csv"

df.to_csv(filepath)
## generate training data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
vectorizer = CountVectorizer(vocabulary=basegenre)
corpus = (df['newgenres'])
X = vectorizer.fit_transform(corpus)
y = (X.toarray())
Xdata = min_max_scaler.fit_transform(df.iloc[:,2:-4])
Xdata = np.array(Xdata)
X_train, X_test, y_train, y_test = train_test_split( Xdata ,y, test_size=0.20, random_state=42)

##
classifier = ClassifierChain(RandomForestClassifier(n_estimators=200))

# train
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

## evaluate model

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
print(accuracy)

def get_multilabel_accuracy(y_test, y_pred):
	acc=[]
	acc = [ accuracy_score(y_test[ind], y_pred[ind]) for ind in range(y_test.shape[0]) ]
	return np.array(acc)

Accuracylist = get_multilabel_accuracy(y_test, y_pred)

