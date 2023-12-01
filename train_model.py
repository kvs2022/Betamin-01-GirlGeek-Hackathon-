from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

def train_model(x, y):
    model = RandomForestClassifier()
    model.fit(x, y)

    joblib.dump(model, 'your_model.joblib')

if __name__ == "__main__":
    data = pd.read_csv('your_data.csv')
    x = data[['feature1', 'feature2']]
    y = data['target']

    train_model(x, y)