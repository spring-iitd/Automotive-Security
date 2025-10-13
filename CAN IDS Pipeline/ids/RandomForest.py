from ids.base import IDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from config import *
import sys
from features import feature_extractor

class RandomForest(IDS):
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=4)

    def train(self, **kwargs):
        super().train()
        self.rf.fit(self.X, self.Y)

    def test(self):
        super().test()
        Y_pred = self.predict(self.X)
        return accuracy_score(self.Y, Y_pred)

    def save(self, path):
        joblib.dump(self.rf, path)

    def predict(self, X_test):
        rf_preds = self.rf.predict(X_test)
        return rf_preds

    def load(self, path):
        self.rf = joblib.load(path)

