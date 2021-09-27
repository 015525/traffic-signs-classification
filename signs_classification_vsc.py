import numpy as np 
import pandas as pd
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import gaussian
from skimage.feature import canny
from skimage import color

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

from sklearn .metrics import confusion_matrix
import scikitplot as skplot
from random import randint

file_path = 'C:\\Users\\es-abdoahmed022'

class signs_classification :
    x_features2 = []
    y_labels2 = []
    x_features4 = []
    
    def __init__(self) :
        self.le = LabelEncoder()
    
    def get_data(self) :
        file_path = 'C:\\Users\\es-abdoahmed022'
        y_label = pd.read_csv(os.path.join(file_path, 'labels.csv'))
        x_features = []
        self.y_labels = []

        #len(os.listdir(os.path.join(file_path, 'myData')))
        self.length = len(os.listdir(os.path.join(file_path, 'myData')))
        for i in range(self.length) :
            for j in tqdm(os.listdir(os.path.join(file_path, 'myData', str(i)))) :
                img = plt.imread(os.path.join(file_path, 'myData', str(i), j))
                x_features.append(img)
                self.y_labels.append(y_label['Name'][i])
                
        signs_classification.x_features2 = x_features
        signs_classification.y_labels2 = self.y_labels

        fig = plt.figure(figsize=(20, 45))

        sum = 0
        for i in range(self.length) :
            if i < self.length:
                fig.add_subplot(9,5, i+1)
                plt.title(self.y_labels[sum])
                plt.imshow(x_features[sum])
                sum += len(os.listdir(os.path.join(file_path, 'myData', str(i))))

        plt.savefig('C:/Users/es-abdoahmed022/gui_for_signs/static/sample_data.png', facecolor = "green")
        
        #return y_label.head() , plt.imshow(signs_classification.x_features2[randint(0, len(signs_classification.x_features2))])
        
    
    def data_preprocess(self) :
        self.get_data()
        '''
        signs_classification.x_features4 = []
        for i in (signs_classification.x_features2) :
            edimg = color.rgb2gray(i)
            edimg = gaussian(edimg, sigma = 0.1)
            edimg = canny(edimg, sigma = 0.05)
            signs_classification.x_features4.append(edimg)
        ''' 
        for i in range(len(signs_classification.x_features2)) :
            signs_classification.x_features2[i] = color.rgb2grey(signs_classification.x_features2[i])
            
        for i in range(len(signs_classification.x_features2)):
            signs_classification.x_features2[i] = signs_classification.x_features2[i].reshape(32, 32, 1)

        fig = plt.figure(figsize=(20, 45))

        sum = 0
        for i in range(self.length) :
            if i < self.length:
                fig.add_subplot(9,5, i+1)
                plt.title(self.y_labels[sum])
                plt.imshow(signs_classification.x_features2[sum])
                sum += len(os.listdir(os.path.join(file_path, 'myData', str(i))))

        plt.savefig('C:/Users/es-abdoahmed022/gui_for_signs/static/sample_data_preprocessed.png', facecolor = "green")
            
        #return plt.imshow(signs_classification.x_features4[randint(0, len(signs_classification.x_features4))]), plt.imshow(signs_classification.x_features2[randint(0, len(signs_classification.x_features2))])
    
    def train(self) :
        self.data_preprocess()
        signs_classification.x_features2, signs_classification.y_labels2 = shuffle(signs_classification.x_features2, signs_classification.y_labels2, random_state = 32)
        signs_classification.y_labels2 = self.le.fit_transform(signs_classification.y_labels2)
        signs_classification.y_labels2 = to_categorical(signs_classification.y_labels2, 43)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(signs_classification.x_features2, signs_classification.y_labels2, test_size = 0.2, random_state = 32)
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        self.model = Sequential()

        self.model.add(Conv2D(filters = 30, kernel_size = (4,4), input_shape = (32,32,1), activation = 'relu'))
        self.model.add(Conv2D(filters = 20, kernel_size = (4,4), input_shape = (32,32,1), activation = 'relu'))

        self.model.add(MaxPool2D(pool_size = (2,2)))
        self.model.add(MaxPool2D(pool_size = (2,2)))


        self.model.add(Flatten())
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(43, activation = 'softmax'))

        self.model.summary()
        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        
        early_stop = EarlyStopping(monitor = 'val_loss', patience = 2)
        self.model.fit(self.x_train, self.y_train, epochs = 8, validation_data = (self.x_test, self.y_test))
        self.loss = pd.DataFrame(self.model.history.history)

        return self.loss['accuracy'][7], self.loss['val_accuracy'][7]

    def accs_plots (self) :
        self.train()
        self.loss[['accuracy', 'val_accuracy']].plot()
        plt.savefig('C:/Users/es-abdoahmed022/gui_for_signs/static/plots_off_accs.png')

        self.loss[['loss', 'val_loss']].plot()
        plt.savefig('C:/Users/es-abdoahmed022/gui_for_signs/static/plots_off_losses.png')

        predictions = np.argmax(self.model.predict(self.x_test), axis = -1)
        y_labels3 = self.y_labels

        y_labels3 = shuffle(y_labels3, random_state = 32)

        x,y = train_test_split(y_labels3, test_size = 0.2, random_state = 32)   
        ys = LabelEncoder().fit_transform(y)

        fig , ax = plt.subplots(figsize = (20,20))
        skplot.metrics.plot_confusion_matrix(ys, predictions, ax = ax)
        plt.savefig('C:/Users/es-abdoahmed022/gui_for_signs/static/confusion_matrix.png', facecolor = "green")

    def predict(self, x) :
        x = np.array(x)
        x = color.rgb2gray(x)
        value =  np.argmax(self.model.predict(x.reshape(1,32,32,1)))
        pred = self.le.inverse_transform([value])
        plt.imshow(x)
        print(pred)
        
        return  pred
'''
    def accuracy(self) :
        losses = pd.DataFrame(self.model.history.history)
        print(losses.columns)
        losses[['accuracy', 'val_accuracy']].plot()
        losses[['loss', 'val_loss']].plot()
        predictions = np.argmax(self.model.predict(x_test), axis = -1)
        y_labels3 = self.y_labels

        y_labels3 = shuffle(y_labels3, random_state = 32)

        x,y = train_test_split(y_labels3, test_size = 0.2, random_state = 32)
        ys = LabelEncoder().fit_transform(y)
        fig , ax = plt.subplots(figsize = (20,20))
        skplot.metrics.plot_confusion_matrix(ys, predictions, ax = ax)
'''