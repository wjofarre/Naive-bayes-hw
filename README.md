# Naive-bayes-hw

``` 
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT7/master/data/sms.tsv'
sms = pd.read_table(url, sep='\t', header=None, names=['label', 'msg'])

sms.head()

sms.label.value_counts()

sms.msg.describe()


sms['label'] = sms.label.map({'ham':0, 'spam':1})

sms.head()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sms.msg, sms.label, random_state=1)
X_train.shape
X_test.shape

print X_train.shape
print X_test.shape


from sklearn.feature_extraction.text import CountVectorizer

train_simple = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']

vect = CountVectorizer()

vect.fit(train_simple)

vect.get_feature_names()


# document term matrix
train_simple_dtm = vect.transform(train_simple)

train_simple_dtm

# put into array
train_simple_dtm.toarray()

pd.DataFrame(train_simple_dtm.toarray(), columns=vect.get_feature_names())

test_simple = [" please don't call me"]

test_simple_dtm = vect.transform(test_simple)

test_simple_dtm.toarray()


pd.DataFrame(test_simple_dtm.toarray(), columns=vect.get_feature_names())

vect= CountVectorizer()

train_dtm = vect.fit_transform(X_train)

test_dtm = vect.transform(X_test)

train_dtm

test_dtm

```
