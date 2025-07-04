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
        # super().predict(X_test)
        dt_preds = self.dt.predict(X_test)
        return dt_preds

    def load(self, path):
        self.dt = joblib.load(path)

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