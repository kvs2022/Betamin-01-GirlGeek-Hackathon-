import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

def generate_fake_data():
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    df = pd.DataFrame(data={'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})
    df.to_csv('your_data.csv', index=False)

    fake_model = RandomForestClassifier()
    fake_model.fit(X, y)
    joblib.dump(fake_model, 'your_model.joblib')

if __name__ == "__main__":
    generate_fake_data()
