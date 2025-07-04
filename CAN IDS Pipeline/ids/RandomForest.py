from ids.base import IDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from config import *
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
        # super().predict(X_test)
        rf_preds = self.rf.predict(X_test)
        return rf_preds

    def load(self, path):
        self.rf = joblib.load(path)

    def extract_features(self):
        return feature_extractor.extract_features()

        # dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)

        # dataframe = stat_features.extract_features(dataset_path)
        # X, Y = dataframe.drop(columns = ['flag', 'timestamp']).values, dataframe['flag'].values

        
        # scaler = StandardScaler()
        # scaler.fit(X)

        # # Transform train and test sets
        # X = scaler.transform(X)

        # return X, Y