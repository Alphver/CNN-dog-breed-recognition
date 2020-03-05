import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
"""
#%%
import glob
from PIL import Image
directory='Images/'
name_list = glob.glob(directory + '*/*')
print(name_list)
"""
#%%
#preprocessing
directory='Images/'
data_gen=image.ImageDataGenerator(rescale=1./255,validation_split=0.1)
train_gen=data_gen.flow_from_directory(directory,batch_size=20,target_size=(150,150),class_mode='sparse',subset='training')
val_gen=data_gen.flow_from_directory(directory,batch_size=20,target_size=(150,150),class_mode='sparse',subset='validation')
"""
for i in range(1):
    print(train_gen[16506])
    train_gen.next()
#print(val_gen.class_indices)

"""

#%%
#model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16,3,activation="relu",input_shape=(150,150,3)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(32,3,activation="relu"))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(64,3,activation="relu"))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.GlobalAveragePooling2D(data_format=None))
model.add(keras.layers.Dense(120, activation="softmax"))
model.summary()

#%%
model.compile(loss = "sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999), metrics = ["accuracy"])
#%%
history = model.fit_generator(train_gen,epochs=60,validation_data=val_gen)
#model.evaluate(x_test.reshape(x_test.shape[0], 28, 28, 1), y_test)