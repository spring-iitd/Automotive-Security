from config import *
from ids.base import IDS
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
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
        super().train()
        X_train = np.array(self.X).astype("float32")
        Y_train = np.array(self.Y).astype("int32")

        self.mlp.compile(optimizer='adam',
                        loss=SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])

        self.es = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

        self.mlp_hist = self.mlp.fit(X_train, Y_train, epochs=EPOCHS, validation_split=0.2, callbacks = [self.es], batch_size = 8192)

    def test(self):
        super().test()
        X_test = np.array(self.X).astype("float32")
        Y_test = np.array(self.Y).astype("int32")
        Y_pred = self.predict(X_test)
        return accuracy_score(Y_test, Y_pred)

    def save(self, path):
        joblib.dump(self.mlp, path)

    def predict(self, X_test):
        return self.mlp.predict(X_test, batch_size=8192).argmax(axis=1)

    def load(self, path):
        self.mlp = joblib.load(path)
