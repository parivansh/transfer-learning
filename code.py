from keras.applications import MobileNet
img_rows, img_cols = 224, 224 
MobileNet = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))
for layer in MobileNet.layers:
    layer.trainable = False

for (i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
    
def lw(bottom_model, num_classes):

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(126,activation='relu')(top_model)
    top_model = Dense(64,activation='relu')(top_model)
    top_model = Dense(32,activation='relu')(top_model)
    top_model = Dense(16,activation='relu')(top_model)
    top_model = Dense(8,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model
    
 from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


num_classes = 2

FC_Head = lw(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())

from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'image//train//'
validation_data_dir = 'image//test//'


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      zoom_range=0.2,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
batch_size = 17
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("facemodel.h5",
                             monitor="loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'loss', 
                          min_delta = 0, 
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)

callbacks = [earlystop, checkpoint] 
model.compile(loss = 'categorical_crossentropy',
              optimizer = adam(lr = 0.001),
              metrics = ['accuracy'])

nb_train_samples = 100
nb_validation_samples = 2
epochs = 20
batch_size = 17

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)
    
from keras.models import load_model
classifier = load_model('facemodel.h5')
import PIL
from PIL import Image
im = Image.open("image/predict/George_W_Bush_0074.jpg")
aim = im.resize((224,224))
from keras.preprocessing import image
aim = image.img_to_array(aim)
import numpy as np
image = np.expand_dims(aim, axis=0)
result = classifier.predict(image)
result
