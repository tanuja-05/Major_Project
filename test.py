from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
import os
from nltk.stem import PorterStemmer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

textdata = []
labels = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

'''
dataset = pd.read_csv("Dataset/labeled_tweets.csv")
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'full_text')
    label = dataset.get_value(i, 'label')
    label = label.strip()
    msg = msg.strip().lower()
    clean = cleanPost(msg)
    textdata.append(msg)
    if label == 'Offensive':
        labels.append(1)
    else:
        labels.append(0)
    print(str(i)+" "+str(clean))    
        

dataset = pd.read_csv("Dataset/public_data_labeled.csv")
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'full_text')
    label = dataset.get_value(i, 'label')
    label = label.strip()
    msg = msg.strip().lower()
    clean = cleanPost(msg)
    textdata.append(msg)
    if label == 'Offensive':
        labels.append(1)
    else:
        labels.append(0)
    print(i)    

textdata = np.asarray(textdata)
labels = np.asarray(labels)

vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace',max_features=7500)
wordembed = vectorizer.fit_transform(textdata).toarray()
np.save("model/X", wordembed)
np.save("model/Y", labels)
with open('model/vector.txt', 'wb') as file:
    pickle.dump(vectorizer, file)
file.close()

print(wordembed)

print(wordembed.shape)
print(labels.shape)
'''

with open('model/vector.txt', 'rb') as file:
    vectorizer = pickle.load(file)
file.close()
X = np.load("model/X.npy")
Y = np.load("model/Y.npy")

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
print(X)
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)

if os.path.exists('model/ada.txt'):
    with open('model/ada.txt', 'rb') as file:
        ab = pickle.load(file)
    file.close()
else:
    ab = AdaBoostClassifier()
    ab.fit(X_train, y_train)
    with open('model/ada.txt', 'wb') as file:
        pickle.dump(ab, file)
    file.close()
predict = ab.predict(X_test)
acc = accuracy_score(predict, y_test)
print(acc)

if os.path.exists('model/sgd.txt'):
    with open('model/sgd.txt', 'rb') as file:
        sgd = pickle.load(file)
    file.close()
else:
    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    with open('model/sgd.txt', 'wb') as file:
        pickle.dump(sgd, file)
    file.close()
predict = sgd.predict(X_test)
acc = accuracy_score(predict, y_test)
print(acc)

if os.path.exists('model/nb.txt'):
    with open('model/nb.txt', 'rb') as file:
        nb = pickle.load(file)
    file.close()
else:
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    with open('model/nb.txt', 'wb') as file:
        pickle.dump(nb, file)
    file.close()

predict = sgd.predict(X_test)
acc = accuracy_score(predict, y_test)
print(acc)

msg = []
msg.append(cleanPost('you bitch go to hell'))
msg.append(cleanPost('today weather is good and peacefull'))

test = vectorizer.transform(msg).toarray()
print(test.shape)

predict = ab.predict(test)
print(predict)



    
