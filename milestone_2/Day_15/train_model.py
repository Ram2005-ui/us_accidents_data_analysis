# train_model.py
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import joblib

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
joblib.dump(model, 'iris_model.pkl')

print("✅ Model trained and saved as 'iris_model.pkl'")
