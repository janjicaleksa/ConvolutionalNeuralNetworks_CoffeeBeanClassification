from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%%

width = 128
height = 128
input_shape = (width, height)


#Putanje do podataka
train_dir = 'C:/Users/Aca/OneDrive - student.etf.bg.ac.rs/Desktop/nm - deep/train'
test_dir = 'C:/Users/Aca/OneDrive - student.etf.bg.ac.rs/Desktop/nm - deep/test'

max_epochs = 100
batch_size = 16

#Ucitavanje csv fajla
data_csv = pd.read_csv('Coffee Bean.csv')


#Raspodela podataka po klasama
data_csv['labels'].hist()




#%% Ucitavanje i augmentacija podataka


data_generator = ImageDataGenerator(
    rescale = 1/255.0,
    vertical_flip = True,
    horizontal_flip = True,
    rotation_range = 90,
    height_shift_range = 0.3,
    width_shift_range = 0.5,
    brightness_range = [0.1, 0.9],
    validation_split = 0.2
)

#Lista klasa
classes = listdir(train_dir)

train_gen = data_generator.flow_from_directory(
    train_dir,
    target_size = input_shape,
    batch_size = batch_size,
    class_mode = 'categorical',
    classes = classes,
    subset = 'training',
    shuffle = False
)

validation_gen = data_generator.flow_from_directory(
    test_dir,
    target_size = input_shape,
    batch_size = batch_size,
    class_mode = 'categorical',
    classes = classes,
    subset = 'training',
    shuffle = False   
)

test_data = ImageDataGenerator(
    rescale = 1/255.0,
    vertical_flip = True,
    horizontal_flip = True,
    rotation_range = 90,
    height_shift_range = 0.3,
    width_shift_range = 0.5,
    brightness_range = [0.1, 0.9]
)

test_gen = test_data.flow_from_directory(
    test_dir,
    target_size = input_shape,
    batch_size = 1,
    class_mode = None,
    classes = classes,
    shuffle = False,
)
#%% Prikaz slika
class_ind = train_gen.class_indices
class_ind = dict((v,k) for k,v in class_ind.items())

fig = plt.figure(figsize = (20,10))
for i, idx in enumerate(np.random.choice(test_gen.samples,size = 20,replace = False)):
   ax = fig.add_subplot(4,5,i+1,xticks =[],yticks=[])
   ax.imshow(np.squeeze(test_gen[idx]))
   ax.set_title(class_ind[test_gen.classes[idx]])
#%% Struktura modela
model = Sequential()

model.add(
    Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(width,height,3))
)

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, (5,5), activation='relu', padding='same'))
model.add(Conv2D(512, (5,5), activation='relu'))
model.add(MaxPooling2D((4,4)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(classes), activation='softmax'))

model.summary()

#%% Treniranje modela
lr = 0.0001
opt = Adam(learning_rate = lr)

es = EarlyStopping(monitor='val_loss', patience=95)

model.compile(loss = 'categorical_crossentropy',optimizer = opt,metrics = ['accuracy'])

history = model.fit(train_gen,epochs = max_epochs,validation_data = validation_gen, callbacks = [es])

#%% Loss i accuracy

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoha')
plt.legend(['Trening', 'Validacija'])
plt.show()

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Tacnost')
plt.xlabel('epoha')
plt.legend(['Trening', 'Validacija'])
plt.show()


#%% Rezultati
pred_classes = np.argmax(model.predict(test_gen),axis = 1)
true_classes = test_gen.classes

pred_classes_train = np.argmax(model.predict(train_gen),axis = 1)
true_classes_train = train_gen.classes

pred_classes_val = np.argmax(model.predict(validation_gen),axis = 1)
true_classes_val = validation_gen.classes

acc = accuracy_score(true_classes, pred_classes)*100
acc_train = accuracy_score(true_classes_train, pred_classes_train)*100
acc_val = accuracy_score(true_classes_val, pred_classes_val)*100

print("Tacnost na test skupu: " + str(acc) + "%")
print("Tacnost na trening skupu: " + str(acc_train) + "%")
print("Tacnost na validacionom skupu: " + str(acc_val)+ "%")

#%% Primeri rezultata
fig = plt.figure(figsize = (20,10))
for i, idx in enumerate(np.random.choice(test_gen.samples,size = 20,replace = False)):
   ax = fig.add_subplot(4,5,i+1,xticks =[],yticks=[])
   ax.imshow(np.squeeze(test_gen[idx]))
   pred_idx = pred_classes[idx]
   true_idx = true_classes[idx]
   plt.tight_layout()
   ax.set_title("{}\n({})".format(class_ind[pred_idx], class_ind[true_idx]),color=("green" if pred_idx == true_idx else "red"))
   
#%% Konfuzione matrice

cm = confusion_matrix(pred_classes,true_classes)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = classes)

cm_display.plot()
plt.show()

cm_train = confusion_matrix(pred_classes_train,true_classes_train)
cm_display_train = ConfusionMatrixDisplay(confusion_matrix = cm_train, display_labels = classes)

cm_display_train.plot()
plt.show()