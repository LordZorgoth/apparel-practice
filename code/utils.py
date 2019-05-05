import numpy as np

def convert_arrays_to_accuracy(y1,y2):
    """
    convert_arrays_to_accuracy(y1,y2)
    
    This function returns the rate of agreements between two sets of labels,
    so that for example convert_arrays_to_accuracy(model.predict(X_val),y_val)
    should produce validation accuracy.
    """
    assert y1.shape==y2.shape
    M=y1.shape[0]
    acc=sum(np.array([np.argmax(y1[i,:]) for i in range(0,M)])==
            np.array([np.argmax(y2[i,:]) for i in range(0,M)]))/M
    return acc

#This class is copied off of a StackOverflow answer.
#I am claiming it so that I can plot information on
#loss and accuracy vs computation time instead of just
#epochs.
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
