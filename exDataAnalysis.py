#Imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Import review file. 2 separate ones for differnt environments
#df = pd.read_json("~/Documents/Data/amazon_videogame_reviews.json", lines = True)
df = pd.read_json("Video_Games_5.json", lines = True)

#Drop NaNs in review text and change it from object to string
df["reviewText"].dropna(inplace= True)
df["reviewText"] = df["reviewText"].astype(str)

#Turn 4 and higher into 2 category; 2 and less into 0 category; 3s are 1
def target_setter(value):
    if value >= 4:
        return(2)
    elif value <= 2:
        return(0)
    else:
        return(1)

#Call previous function as lambda format
df["target"] = df["overall"].apply(lambda x: target_setter(x))

#X is feature and Y is target
X = df["reviewText"]
Y = df["target"]

#Split data 67-33 split chosen arbitrarily
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.33)

#Create Count Vectorizer
count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(X_train)

#Create TF-IDF Transformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Create Multinomial Naive Bayes Model
model = MultinomialNB().fit(X_train_tfidf, Y_train)

#Format test data
X_test_counts = count_vector.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = model.predict(X_test_tfidf)

#Create Classification report: Overall Accuracy 0.79
target_names = ["negative", "neutral", "positive"]
actual = list(Y_test)
predicted = list(predicted)
print(classification_report(actual, predicted, target_names=target_names))
