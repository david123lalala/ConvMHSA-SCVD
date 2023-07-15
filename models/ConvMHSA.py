import numpy as np
# from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Embedding,Conv1D,GlobalMaxPooling1D,Dropout,Dense,GlobalAveragePooling1D,Input,LSTM,concatenate
from keras.optimizers import Adamax
from sklearn.metrics import confusion_matrix
from parser import parameter_parser
import matplotlib.pyplot as plt
import sys
from  models.MultiHeadAttention import MultiHeadSelfAttention_Model

from imblearn.over_sampling import SMOTE

args = parameter_parser()

class ConvMHSA_Model:
    def __init__(self, data, name="", batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, dropout=args.dropout):
        vectors = np.stack(data.iloc[:, 0].values)
        labels = data.iloc[:, 1].values

        oversample= SMOTE(random_state=42)
        new_vectors=np.reshape(vectors,(len(vectors),-1))

        X_resampled,Y_resampled = oversample.fit_sample(new_vectors, labels)
        new_X_resampled=np.reshape(X_resampled,(len(X_resampled),100,300))
        x_train, x_test, y_train, y_test = train_test_split(new_X_resampled, Y_resampled,
                                                            test_size=0.2, stratify=Y_resampled)
                                                            
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.name = name
        self.dropout=dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)

        embed_dim = x_train.shape[2]
        num_heads = 4 
        ff_dim = 32 

        embed_input = Input(shape=(x_train.shape[1],x_train.shape[2]))
        x = MultiHeadSelfAttention_Model(embed_dim, num_heads)(embed_input)
        x1 = Conv1D(filters=2,kernel_size=3,padding='VALID', strides = 1, activation='relu',name='conv2')(x)
        x2 = Conv1D(filters=2,kernel_size=6,padding='VALID', strides = 1, activation='relu',name='conv3')(x)
        x3 = Conv1D(filters=2,kernel_size=9,padding='VALID', strides = 1, activation='relu',name='conv4')(x)
        cnn1 = GlobalMaxPooling1D()(x1)
        cnn2 = GlobalMaxPooling1D()(x2)
        cnn3 = GlobalMaxPooling1D()(x3)
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        x = Dense(200, activation="relu")(cnn)
        x = Dropout(0.1)(x)
        output = Dense(2, activation="softmax")(x)
        model = Model(inputs=embed_input, outputs=output) 

        adamax = Adamax(lr)
        model.compile(optimizer=adamax, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model


    def train(self):
        history =self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,class_weight=self.class_weight,validation_data=(self.x_test, self.y_test))

    def test(self):
        values = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        predictions = (self.model.predict(self.x_test, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        print("validation Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
        print('validation  False positive rate(FPR): ', fp / (fp + tn))
        print('validation  False negative rate(FNR): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('validation Recall(TPR): ', recall)
        precision = tp / (tp + fp)
        print('validation Precision: ', precision)
        print('validation F1 score: ', (2 * precision * recall) / (precision + recall))




