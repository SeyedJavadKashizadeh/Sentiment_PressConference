import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, History, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix


def split_data(emotions, filename, train_set):
    df=pd.read_csv(filename, sep=',')
    # Uncomment to drop a feature
##    df = df.drop([col for col in df.columns if "mfccs" in col], axis=1)
##    df = df.drop([col for col in df.columns if "chroma" in col], axis=1)
##    df = df.drop([col for col in df.columns if "mel" in col], axis=1)
    df = df.drop([col for col in df.columns if "contrast" in col], axis=1)
    df = df.drop([col for col in df.columns if "tonnetz" in col], axis=1)
    df = df.drop([col for col in df.columns if "path" in col], axis=1)
    df = df.drop([col for col in df.columns if "dataset" in col], axis=1)
    df = df.drop([col for col in df.columns if "emotion_code" in col], axis=1)
    '''
        Create balanced training sample
    '''
    y_df = df['emotion_name'] 
    counts = [len(df[df.emotion_name == emotion]) for emotion in df.emotion_name.unique() if emotion in emotions]


    int2emotions = {i: e for i, e in enumerate(emotions)}
    emotions2int = {v: k for k, v in int2emotions.items()}

    min_count = math.floor(min(counts)*train_set)
    x_train, x_test = pd.DataFrame(), pd.DataFrame()
    y_train, y_test = pd.DataFrame(columns=['emotion_name']), pd.DataFrame(columns=['emotion_name'])

    for emotion in emotions:
        temp = df.loc[df.emotion_name == emotion]
        train_temp = temp.sample(n=min_count, random_state=100)
        # left df is the "big" one, right df is the sub-set for training, keep if data only appear in the former (i.e., testing data)
        test_temp = pd.merge(temp, train_temp, how='outer', indicator=True)\
                      .query('_merge == "left_only"').drop(columns=['_merge'])
        x_train = pd.concat([x_train, train_temp.drop(columns=['emotion_name'])])
        y_train = pd.concat([y_train, pd.DataFrame(train_temp['emotion_name'])])
        x_test = pd.concat([x_test, test_temp.drop(columns=['emotion_name'])])
        y_test = pd.concat([y_test, pd.DataFrame(test_temp['emotion_name'])])


    print('Training features:{}; Training output:{}; Testing features:{}; Testing output:{}'.format(x_train.shape,y_train.shape,x_test.shape,y_test.shape))
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
    return int2emotions, emotions2int, x_train, y_train, x_test, y_test

def test_score(model, y_test, x_test):
    """Compute accuracy assuming y_test is one-hot encoded."""
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    return accuracy_score(y_true, y_pred)

def conf_matrix(model, y_test, x_test, emotions2int, emotions):
    # y_test is one-hot -> convert to integer labels
    y_true = [np.argmax(v) for v in y_test]
    # Predict class probabilities then take argmax
    y_prob = model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=-1)
    cm = confusion_matrix(y_true, y_pred, labels=[emotions2int[e] for e in emotions])
    return pd.DataFrame(cm,
                        index=[f"t_{e}" for e in emotions],
                        columns=[f"p_{e}" for e in emotions])

'''
The code can be modified to build networks with different structures/layers
'''


def train_model(model_path, infile):
    # specify the list of emotions 
    emotions = ['happy', 'pleasant_surprise', 'neutral', 'sad', 'angry']

    int2emotions, emotions2int, x_train, y_train, x_test, y_test = split_data(emotions, infile, train_set=0.8)
    ###one hot coder
    y_train = to_categorical([emotions2int[str(e[0])] for e in y_train])
    y_test = to_categorical([emotions2int[str(e[0])] for e in y_test])
    
    '''
        Create neural network
    '''
    target_class = len(emotions)
    input_length = x_train.shape[1]

    #tuning parameters here
    dense_units = 200
    dropout = 0.3
    loss = 'categorical_crossentropy'
    optimizer = 'adam'

    model = Sequential()
    model.add(Dense(dense_units, input_dim=input_length))
    model.add(Dropout(dropout))
    model.add(Dense(dense_units))
    model.add(Dropout(dropout))
    model.add(Dense(dense_units))
    model.add(Dropout(dropout))
    model.add(Dense(target_class, activation='softmax'))
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[CategoricalAccuracy(),
                           Precision(),
                           Recall()])

    '''
        Training
    '''

    checkpointer = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)
    model_training = model.fit(x_train, y_train,
                               batch_size=64,
                               epochs=1000,
                               validation_data=(x_test, y_test),
                               callbacks=[checkpointer, lr_reduce])

    '''
        Checking accuracy score and confusion matrix
    '''
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    y_test = [np.argmax(i, out=None, axis=None) for i in y_test]

    print(accuracy_score(y_true=y_test, y_pred=y_pred))

    matrix = confusion_matrix(y_test, y_pred,
                              labels=[emotions2int[e] for e in emotions])
    matrix = pd.DataFrame(matrix, index=[f"t_{e}" for e in emotions],columns=[f"p_{e}" for e in emotions])
    print(matrix)
    
    return model, int2emotions, emotions2int