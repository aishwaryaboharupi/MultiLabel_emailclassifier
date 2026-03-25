from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        # Encapsulation: TF-IDF and Classifier are bundled in one Pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def print_results(self, y_true, y_pred, label_name):
        acc = accuracy_score(y_true, y_pred)
        print(f"Results for {label_name}: Accuracy = {acc * 100:.2f}%")