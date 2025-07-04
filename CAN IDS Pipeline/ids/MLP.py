from config import *
from ids.base import IDS
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] =  '0' #= all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from features import feature_extractor

class MLP(IDS):
    def __init__(self):
        self.mlp = Sequential()
        self.mlp.add(Input(shape = (4,)))
        self.mlp.add(Dense(128, activation = 'relu'))
        self.mlp.add(Dense(128, activation = 'relu'))
        self.mlp.add(Dense(4, activation='softmax'))

    def train(self, **kwargs):
        # super().train(X_train, Y_train)
        super().train()
        X_train = np.array(self.X).astype("float32")
        Y_train = np.array(self.Y).astype("int32")

        self.mlp.compile(optimizer='adam',
                        loss=SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])

        self.es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

        self.mlp_hist = self.mlp.fit(X_train, Y_train, epochs=EPOCHS, validation_split=0.2, callbacks = [self.es], batch_size = 8192)

    def test(self):
        # super().test(X_test, Y_test)
        super().test()
        X_test = np.array(self.X).astype("float32")
        Y_test = np.array(self.Y).astype("int32")
        Y_pred = self.predict(X_test)
        return accuracy_score(Y_test, Y_pred)

    def save(self, path):
        joblib.dump(self.mlp, path)

    def predict(self, X_test):
        # super().predict()
        # X_test = np.array(self.X).astype("float32")
        return self.mlp.predict(X_test, batch_size=8192).argmax(axis=1)
        # return self.mlp.predict(X_test).argmax(axis=1)

    def load(self, path):
        self.mlp = joblib.load(path)

    def extract_features(self):
        return feature_extractor.extract_features()
    

        # dataset_path = os.path.join(DIR_PATH, "..", "datasets", DATASET_NAME)

        # dataframe = stat_features.extract_features(dataset_path)
        # X, Y = dataframe.drop(columns = ['flag', 'timestamp']).values, dataframe['flag'].values

        # scaler = StandardScaler()
        # scaler.fit(X)

        # # Transform train and test sets
        # X = scaler.transform(X)
            
        # if Y is not None:
        #     Y = np.copy(Y)
        #     Y = (Y == 'T').astype(int)

        # return X, Y