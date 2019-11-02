from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

reviews_train = []
for line in open('full_train.txt', 'r', encoding='utf8'):
    reviews_train.append(line.strip())

reviews_test = []
for line in open('full_test.txt', 'r', encoding='utf8'):
    reviews_test.append(line.strip())



REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)
english_stop_words = stopwords.words('english')


target = [1 if i < 12500 else 0 for i in range(25000)]

stop_words = ['in','of', 'at', 'a', 'the', 'but', 'we','I','they', 'to']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), stop_words=stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)
X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = .75
    )
# for c in [0.001,0.007, 0.008 0.005, 0.01, 0.05, 0.1]:
#
#     svm = LinearSVC(C=c)
#     svm.fit(X_train, y_train)
#     print ("Accuracy for C=%s: %s"% (c, accuracy_score(y_val, svm.predict(X_val))))


final = LinearSVC(C=0.007)
final.fit(X, target)
print ("Final Accuracy: %s"
       % accuracy_score(target, final.predict(X_test)))
