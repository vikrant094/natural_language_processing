
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.layers import Dropout,BatchNormalization
import nltk
import numpy as np
import pickle
from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def build_model(input_dim, hidden_neurons, output_dim):
    """
    buliding model architecture
    """
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=input_dim))
    model.add(Activation('relu'))

    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    #model.add(Dense(2*hidden_neurons))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(BatchNormalization())
    model.add(Dense(output_dim, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])
    model.summary()
    return model


# In[ ]:


def transform_data(X_train, y_train,X_test, y_test,X_eval,y_eval):
        
    '''transforming data into required form'''
    global dict_vectorizer
    global label_encoder
    
    dict_vectorizer = DictVectorizer(sparse=False)
    dict_vectorizer.fit(X_train + X_test)
    
    
    filehandler = open(b"dict_vectorizer.obj","wb")
    pickle.dump(dict_vectorizer,filehandler)
    filehandler.close()
    
    
    X_train = dict_vectorizer.transform(X_train)
    X_test = dict_vectorizer.transform(X_test)
    X_eval = dict_vectorizer.transform(X_eval)
    
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train + y_test + y_eval)
    
    filehandler = open(b"label_encoder.obj","wb")
    pickle.dump(label_encoder,filehandler)
    filehandler.close()
    
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    y_eval = label_encoder.transform(y_eval)
    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    y_eval = np_utils.to_categorical(y_eval)
    
    return [X_train, y_train,X_test, y_test,X_eval,y_eval]


# In[3]:


def deep_keras_model(X_train, y_train,X_test, y_test,X_eval,y_eval,epochs):
    
    '''training keras model'''
    
    data_list = transform_data(X_train, y_train,X_test, y_test,X_eval,y_eval)
    X_train =  data_list[0]
    y_train = data_list[1]
    X_test = data_list[2]
    y_test = data_list[3]
    X_eval = data_list[4]
    y_eval = data_list[5]

    model_params = {
        'build_fn': build_model,
        'input_dim': X_train.shape[1],
        'hidden_neurons': 512,
        'output_dim': y_train.shape[1],
        'epochs': epochs,
        'batch_size': 256,
        'verbose': 2,
        'validation_data': (X_test, y_test),
        'shuffle': True
    }
    
    model = KerasClassifier(**model_params)
    history = model.fit(X_train, y_train,callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])
    score = model.score(X_eval, y_eval, verbose=0)    
    print('\n model accuracy on eval dataset: {}'.format(score))
    print('accuracy %s, precision %s, recall_score %s,f1_score %s'%(accuracy,precision,recall_score,f1_score))
    
    #WRITING FILE
    accuracy_file = open(os.path.join('../output','accuracy.txt'),'a+')
    accuracy_file.write('---keras model---\n\n')
    accuracy_file.write('accuracy = '+ str(accuracy)+"\n")
    accuracy_file.write('precision = '+ str(precision)+"\n")
    accuracy_file.write('recall_score = '+ str(recall_score)+"\n")
    accuracy_file.write('f1_score = '+ str(f1_score)+"\n\n\n")
    accuracy_file.close()
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show();
    
    return model







