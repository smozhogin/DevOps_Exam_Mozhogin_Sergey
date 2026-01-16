import os
import yaml
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

n_estimators_list = params['train']['n_estimators']
random_state = params['train']['random_state']
test_size = params['train']['test_size']

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

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))

mlflow.set_experiment('Эксперимент с классификацией ирисов')

for d in ['models', 'plots']:
    os.makedirs(d, exist_ok=True)

best_f1 = -1.0
best_model = None
best_n = None

for n in n_estimators_list:
    model = RandomForestClassifier(n_estimators=n, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plot_path = os.path.join('plots', f'plot_{n}.png')

    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred)
    plt.title('Результаты классификации ирисов')
    plt.xlabel('Параметр 1')
    plt.ylabel('Параметр 2')
    plt.savefig(plot_path)

    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')

    if f1_weighted > best_f1:
        best_f1 = f1_weighted
        best_model = model
        best_n = n

    model_path = os.path.join('models', f'model_{n}.pkl')

    joblib.dump(model, model_path)

    with mlflow.start_run(run_name=f'RandomForestClassifier с n_estimators == {n}'):
        mlflow.log_param('model', 'RandomForestClassifier')
        mlflow.log_param('n_estimators', n)
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('f1_weighted', f1_weighted)
        mlflow.log_metric('precision_weighted', precision_weighted)
        mlflow.log_metric('recall_weighted', recall_weighted)
        mlflow.log_artifact(plot_path, artifact_path='artifacts')
        mlflow.sklearn.log_model(model, artifact_path='sklearn_model')

best_model_path = os.path.join('models', 'model.pkl')
joblib.dump(best_model, best_model_path)

with mlflow.start_run(run_name='Лучшая модель'):
    mlflow.set_tag('best_model', 'true')
    mlflow.log_param('model', 'RandomForestClassifier')
    mlflow.log_param('n_estimators', best_n)
    mlflow.log_metric('best_f1_weighted', best_f1)

    signature = infer_signature(X_train, best_model.predict(X_train))

    mlflow.sklearn.log_model(
        best_model,
        artifact_path='model',
        signature=signature,
        input_example=X_train.head(1),
        registered_model_name='iris_random_forest'
    )