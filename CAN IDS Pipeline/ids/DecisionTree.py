from ids.base import IDS
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib
from features import feature_extractor
from config import *

class DecisionTree(IDS):
    def __init__(self):
        self.dt = DecisionTreeClassifier(max_depth = 4)

    def train(self, **kwargs):
        super().train()
        self.dt.fit(self.X, self.Y)

    def test(self, **kwargs):
        super().test()
        Y_pred = self.predict(self.X)
        return accuracy_score(self.Y, Y_pred)

    def save(self, path):
        joblib.dump(self.dt, path)
    
    def predict(self, X_test):
        dt_preds = self.dt.predict(X_test)
        return dt_preds

    def load(self, path):
        self.dt = joblib.load(path)
