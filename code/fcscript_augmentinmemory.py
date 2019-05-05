from load_data import *
from modelbuilder import model_build_conv
from sklearn.model_selection import train_test_split
from keras.callbacks import History
from keras.metrics import categorical_accuracy
from utils import convert_arrays_to_accuracy
import augmentation as aug

#m is the number of training examples to load from the dataset.
m=60000
epochs=150
#n_transforms is the number of transformations to create for each image
n_transforms=10

X,y=load_train_data(m)

#Ensure that we always use the same training, cross-validation sets,
#then number of transformations.
np.random.seed(1)
Xtr,Xtst,ytr,ytst=train_test_split(X,y,train_size=0.6,test_size=0.4)
Xtr,ytr=aug.augment_dataset(Xtr,ytr,n_transforms,fixed_seeds=True)

model=model_build_fc()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[categorical_accuracy])
my_hist=History()
time_hist=TimeHistory()
#Because we are feeding in an augmented dataset using model.fit,
#the training categorical accuracy returned by the model will be
#based on the augmented training set. However, we are far more interested
#in the true training accuracy, based on predictions from the
#unaugmented training set.
train_accs=[]
for E in range(epochs):
    model.fit(Xtr,ytr,verbose=2,callbacks=[my_hist,time_hist],
              validation_data=(Xtst,ytst),epochs=E+1,initial_epoch=E,
              batch_size=32,steps_per_epoch=(1125*m*(1+n_transforms))//60000,
              shuffle=True)
    train_accs.append(model.predict(Xtr[0:m*(1+n_transforms):1+n_transforms]))
    print("True training accuracy is"+str(train_accs[-1])[0:6]+".\n")
