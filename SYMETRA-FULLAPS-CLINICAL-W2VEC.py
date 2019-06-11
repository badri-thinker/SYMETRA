import nltk                       # the natural langauage toolkit, open-source NLP
# install if not already downloaded nltk.download(stopwords)
#nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd               # pandas dataframe
import re                         # regular expression
import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import gensim
import string
import ast
from gensim.models import KeyedVectors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import  pairwise
from sklearn.preprocessing import normalize
import datetime
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from textblob import TextBlob
from spellchecker import SpellChecker
from collections import Counter
from interpret.glassbox import ExplainableBoostingClassifier
spell=SpellChecker(distance=1)
stop_words = set(stopwords.words('english'))

CLINICAL_DIR="c:/USHUR/PythonProg/DATAREP/clinical"
f=os.path.join(CLINICAL_DIR,'word2vec.model')
m = gensim.models.KeyedVectors.load(f)
clinical_embedding = (m[m.wv.vocab])
#print(type(clinical_embedding),clinical_embedding.shape)
DATA_DIR="c:/USHUR/PythonProg/DATAREP/symetra"
if1=os.path.join(DATA_DIR,'FULL_Aps.csv')
df_L=pd.read_csv(if1)
df_L['Policy Number']=df_L['Policy Number'].apply(lambda x:re.sub("[^0-9]",'',x) )
#remove non numeric prefiexes from policy ID
df_L.columns=df_L.columns.str.replace(' ','')

#remove dupicate APPID
df_L=df_L[df_L.PolicyNumber!='00008906']
df_L=df_L[df_L.PolicyNumber!='00011820']
#spl app id were removed (duplicate and  no clinical embedding words

df_L['Aps']=df_L['Aps'].apply(lambda x:re.sub("APS\n|\n",' ',x) )
#df_L['Aps']=df_L['Aps'].apply(lambda x:gensim.utils.simple_preprocess(x))
print(df_L.info())
APS=df_L.PolicyNumber.tolist()
print(len(APS),APS)

Y_train=df_L['Quoted'].values
print(Y_train)
X_train=np.empty((0,300))
#clinical word2vec is 100D, mean, min, max is 300D

for AP in APS:
    df_CP = df_L.loc[df_L['PolicyNumber'] == AP,'Aps'].tolist()
    #print(type(df_CP[0]),df_CP[0])
    words = df_CP[0].translate(str.maketrans('', '', string.punctuation)).split()
    words = [x for x in words if (x not in stop_words)]
    #print("number of words", type(words), len(words))



    DOC = np.empty((0, 100))

    embeddingwords=0
    for word in words:
        #print(word)
        try:
            clinical_embedding[m.wv.vocab[word].index]
            DOC = np.append(DOC,clinical_embedding[m.wv.vocab[word].index].reshape(1,-1) , axis=0)
            embeddingwords=embeddingwords+1
        except KeyError:
            continue
        #print("number of words with embedding",embeddingwords)
        if embeddingwords==0:
            DOC=np.zeros((0,100))
            print("APP with no embedding found-- RAISE EXCEPTION",AP)
            exit()

    me = np.mean(DOC, axis=0)
    mn = np.min(DOC, axis=0)
    mx = np.max(DOC, axis=0)
    features = np.concatenate([mn, me, mx])
        # each document is encoded into  3x vector
        # print(mx.shape)

        # print(embedding.shape)
    # convert embedding of all docs into a single vector
    # print(embedding.shape)
    X_train = np.append(X_train, features.reshape(1, -1), axis=0)

print(X_train.shape)


X = pd.DataFrame(X_train)
y = pd.Series(Y_train)
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, x_test.shape, Y_train.shape, y_test.shape)

print(x_test)




#  === LOGISTIC REGRESSION CLASSIFIER
lr=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                    intercept_scaling=1,
            class_weight=None, solver='liblinear',
max_iter=1000, multi_class='ovr', verbose=0, warm_start=False)
flr=lr.fit(X_train, Y_train)
y_pred=flr.predict(x_test)
print(" Predicted vs Expected", y_pred,y_test.values.tolist())
print("LogReg accuracy",np.mean(y_pred==y_test))

# === Linear SVM classifier
svr= LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     loss='squared_hinge', max_iter=10000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)

fsvr=svr.fit(X_train, Y_train)
y_pred=fsvr.predict(x_test)
print(" Predicted vs  Expected",y_pred,y_test.values.tolist())
print("Linear SVM accuracy",np.mean(y_pred==y_test))


y = fsvr.decision_function(x_test)
w_norm = np.linalg.norm(fsvr.coef_)
dist = y / w_norm
print(dist)
print(type(y_test))
y=y_test.index.values.tolist()
plt.scatter(y,dist)
#find APPID chosen to be in test set
ann=[APS[i] for i in y]
print(ann)
yt=y_test.values.tolist()
for i, appid  in enumerate(ann):
    x1=y_pred[i]
    x2=yt[i]
    l=str(x1)+str('-')+str(appid)+str('-')+str(x2)
    plt.annotate(l, xy=(y[i], dist[i]))

plt.show()

exit()


