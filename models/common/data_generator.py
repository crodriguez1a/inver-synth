import numpy as np
import keras
from scipy.io import wavfile
import h5py

class SoundDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_file=None, batch_size=32, n_samps=16384,
                  shuffle=True, last:float=0., first:float=0.):
        'Initialization'
        self.dim = (1,n_samps)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_file = data_file
        self.n_channels = 1

        database = h5py.File(data_file,"r")
        self.database = database

        self.n_samps = self.read_file(0).shape[0]
        print("N Samps: {}".format(self.n_samps))

        # set up list of IDs from data files
        n_points = len(database['files'])
        self.list_IDs = range(len(database['files']))

        print("Num points: {}".format(len(self.list_IDs)))
        if last > 0.0:
            slice: int = int(n_points * (1-last))
            self.list_IDs = self.list_IDs[slice:]
            print("Last, with {} points".format(len(self.list_IDs)))
        elif first > 0.0:
            slice: int = int(n_points * first)
            self.list_IDs = self.list_IDs[:slice]
            print("First, with {} points".format(len(self.list_IDs)))

        # set up label size from data files
        self.label_size = len(database['labels'][0])
        self.on_epoch_end()

    def get_audio_length(self):
        return self.n_samps

    def get_label_size(self):
        return self.label_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_file(self,index):
        filename = self.database['files'][index]
        fs, data = wavfile.read(filename)
        return data

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim))
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        X = []
        y = []
        for i in list_IDs_temp:
            #Read labels
            y.append(self.database['labels'][i])
            # Load soundfile data
            data = self.read_file(i)
            if data.shape[0] > self.n_samps:
                print("Warning - too many samples: {} > {}".format(data.shape[0],self.n_samps))
            X.append(data[:self.n_samps])
        Xd = np.expand_dims(np.vstack(X), axis=1)
        yd = np.vstack(y)
        
        return Xd, yd
