import os
import pickle
import json
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

n_estimators = 200
random_state = 42
test_size = 0.3

iris = datasets.load_iris(as_frame=True)
iris_df = iris.frame

iris_df = iris_df.rename(columns={
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)':  'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)':  'petal_width',
})

X = iris_df.drop(columns=[iris_df.columns[-1]])
y = iris_df[iris_df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average="weighted")
precision_weighted = precision_score(y_test, y_pred, average="weighted")
recall_weighted = recall_score(y_test, y_pred, average="weighted")

os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'model.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)