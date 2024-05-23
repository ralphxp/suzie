import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv('labeled_data.csv')

# Preprocess data
# (You need to implement data preprocessing steps here)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['hate_speech'], test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(random_state=42))
])

# Define hyperparameters for grid search
param_grid = {
    'tfidf__max_features': [1000, 2000, 3000],
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Extract features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict on test set
y_pred = model.predict(X_test_tfidf)

y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(model, 'model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
