import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_json("~/Documents/Data/amazon_videogame_reviews.json", lines = True)

##(
def target_setter(value):
    if value >= 4:
        return(2)
    elif value <= 2:
        return(0)
    else:
        return(1)
##)

df["target"] = df["overall"].apply(lambda x: target_setter(x))

X = df["reviewText"]
Y = df["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.33)

count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model = MultinomialNB().fit(X_train_tfidf, Y_train)

X_test_counts = count_vector.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = model.predict(X_test_tfidf)
