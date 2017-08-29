from __future__ import division,print_function
from utils import *


import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
path="S:\\workspace\\street\\signs\\GTSRB\\"

## For preprocessing
g = glob('*')
for d in g: os.mkdir(path+'valid\\'+d+'\\')

g = glob('*\\*.ppm')
shuf = np.random.permutation(g)
for i in range(500): os.rename(shuf[i], path+'\\valid\\' + shuf[i])


from shutil import copyfile

#To convert images to jpeg (Comment this block for most cases)
g=glob('*\\*.ppm')
for i in range(0,len(g)):
    im=Image.open(g[i])
    im.save(g[i][:-4]+'.jpg')

### Till here


#### Pretrained-finetuning and feature extraction

batch_size=64 # except this

batches = get_batches(path+'\\train\\', batch_size=batch_size)
val_batches = get_batches(path+'\\valid\\', batch_size=batch_size*2, shuffle=False)

(val_classes, trn_classes, val_labels, trn_labels, 
    val_filenames, filenames, test_filenames) = get_classes(path)

from vgg16bn import *
model = vgg_ft_bn(43)

trn = get_data(path+'train')
val = get_data(path+'valid')

save_array(path+'results/trn.dat', trn)
save_array(path+'results/val.dat', val)

### No need to run anything till this point

trn = load_array(path+'results\\trn.dat')
val = load_array(path+'results\\val.dat')

gen = image.ImageDataGenerator()

model.compile(optimizer=Adam(1e-4),
       loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(trn, trn_labels, batch_size=batch_size, nb_epoch=5, validation_data=(val, val_labels))

model.save_weights(path+'results/ft1.h5')

model.load_weights(path+'results/ft1.h5')

conv_layers,fc_layers = split_at(model, Convolution2D)

conv_model = Sequential(conv_layers)

conv_feat = conv_model.predict(trn)
save_array(path+'results/conv_feat.dat', conv_feat)
conv_val_feat = conv_model.predict(val)

save_array(path+'results/conv_val_feat.dat', conv_val_feat)
save_array(path+'results/conv_feat.dat', conv_feat)

### Feature extraction done


## Now creating the actual classifier
conv_feat = load_array(path+'results/conv_feat.dat')
conv_val_feat = load_array(path+'results/conv_val_feat.dat')

def get_fully_connected(p):
    return [
        MaxPooling2D(input_shape=(512,14,14)),
        BatchNormalization(axis=1),
        Dropout(p/4),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        Dense(43, activation='softmax')
    ]
    
p=0.6

model_fc = Sequential(get_fully_connected(p))
model_fc.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_fc.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=3, 
             validation_data=(conv_val_feat, val_labels))
model_fc.optimizer.lr=3e-6

model_fc.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=3, 
             validation_data=(conv_val_feat, val_labels))

model_fc.save_weights(path+'results/conv_512_7.h5')

## Training is done

model_fc.evaluate(conv_val_feat, val_labels)

model_fc.load_weights(path+'models/conv_512_6.h5')

signs=['20','30','50','60','70','80','below 80','100','120','No passing','No overtaking by heavy vehicles','Right of way at next crossroad','priority road','Give way','stop','No vehicles','No vehicles over 3.5Tons','No entry','General Caution','Dangerous Curve to left','Dangerous Curve to right','Double curves to left','Bumpy road','slippery road','road narrows on the right','Roadworks','Traffic Signal','Pedestrians','Children Crossing','Bicycle Crossing','Road freezes easily and is then slippery','Wild Animals Crossing','No Parking','Turn Right','Turn Left','Ahead only','Straight or Right','Straight or Left','Keep Right','Keep Left','Roundabout','End of No Overtaking','End of No Overtaking by Heavy vehicles' ]

def show_pred(i):
    temp=model_fc.predict(conv_val_feat[i:i+1])
    plt.imshow(np.rollaxis(val[i],0,3))
    print(signs[np.argmax(temp)])
print(signs[np.argmax(temp)])

