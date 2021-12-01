import librosa
import librosa.display
import pydub
from pydub import AudioSegment
import os
from os import path
import random
from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
import seaborn as sns

from fuctions import remove_file_in_tmp, split_and_save_audio, create_melspectrogram_in_bg

from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

# Global Variable
ROOT_PATH  = os.getcwd()

E_POSITIVE = 'positive'
E_NEGATIVE = 'negative'

SAMPLE_RATING = 22050
FMIN=0.0 #Tần số thấp nhất
FMAX=20000.0 #Tần số nghe cao nhất của con người
MAX_DB = 100 # độ lớn âm thanh lớn nhất 
WIN_LENGTH = N_FFT = 2048
HOP_LENGTH=512 # win_length / 4

def split_and_make_spec_data_path(User_Root_Path):
    #Input: Directory of user audio data
    #Output: list of path of 4 seconds spectrogram cut from user audio

    # #First Remove Old Spec Data, Slicing user_data in to 4s file
    User_Data_Slices = 'user_data_slice'
    remove_file_in_tmp(User_Data_Slices)
    output_path = 'user_data_spec'
    remove_file_in_tmp(output_path) 

    #Make Path Of Audio File
    user_directory_list = pathlib.Path(User_Root_Path)
    user_directory_path = list(user_directory_list.glob('*'))
    user_directory_path = [str(path) for path in user_directory_path]
    spec_path_list = []    
     
    #Cut in to 4 seconds File and create spectrogram
    for path in user_directory_path:
        #Get Emotion
        print(path)
        # tmp_path = path.split('.')[0]
        # emotion = tmp_path.split('_')[-1]

        #Split And Save
        singe_list = split_and_save_audio(path, beginSecond=0,duration=4000,export_path=User_Data_Slices, code='audio',user_trainning=True)    
        #print('singe_list length:', len(singe_list))        
        for path in singe_list:            
            #Create spectrogram
            spec_path = create_spectrogram_from_audio_path(path)
            # print(spec_path)
            spec_path_list.append(spec_path)

    print('spec_list length:', len(spec_path_list))
    return spec_path_list

def create_spectrogram_from_audio_path(audio_path):    
    #Input data: path of audio have label in name
    #Output: path of spectrograme
    #Create Spectrogram
    output_path = 'user_data_spec'
    # output_path = 'tmp'
    # remove_file_in_tmp(output_path)  
    
    data , sr = librosa.load(
        audio_path, 
        sr=SAMPLE_RATING, 
        offset=0.0  ,
        duration = 4,
        )

    if data.shape[0] < SAMPLE_RATING* 4: # Less than 4 seconds
        data = librosa.util.pad_center(data, 88200, mode='constant')
    # librosa.display.waveplot(data)

    # print('path for spec:', audio_path)
    file_name = str(audio_path).split('\\')[-1]
    file_name = file_name.split('.')[0]
    emotion = file_name.split('_')[-1]


    spec_path = create_melspectrogram_in_bg(data, emotion, output_path, file_name)        
    
    return spec_path

def create_spectrogram(User_data_df):
    
    #Input data frame of Path and Emotions
    #Output: list of spectrogram with emotion label in file name
    #Create Spectrogram
    output_path = 'user_data_spec'
    remove_file_in_tmp(output_path)
    user_spec_path_list = []

    for path, emotion in zip(User_data_df.Path, User_data_df.Emotions):
        print(path)
        #print(emotion)    
        #Load Wave File
        data , sr = librosa.load(
            path, 
            sr=SAMPLE_RATING, 
            offset=0.0  ,
            duration = 4,
            )
        if data.shape[0] < SAMPLE_RATING* 4: # Less than 4 seconds
            data = librosa.util.pad_center(data, 88200, mode='constant')
        # librosa.display.waveplot(data)

        file_name = str(path).split('\\')[-1]
        file_name= file_name.split('.')[0]

        spec_path = create_melspectrogram_in_bg(data, emotion, output_path, file_name)
        print(spec_path)
        user_spec_path_list.append(spec_path)
    
    return user_spec_path_list

def preprocess_image(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    #image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path, label):
    image_raw = tf.io.read_file(path)
    image = preprocess_image(image_raw)
    return image, label

def prepare_dataset(all_path, all_label):
    #INPUT: Path of image and Label of Path
    test_size = 0.2 # (20%)
    X_train_path, X_test_path, y_train_label, y_test_label = train_test_split(all_path, all_label, 
                                                        test_size=test_size, random_state=42) # Để nó cùng 1 random_sate để các lần đều giống nhau

    #using tensor sliecs to improve performance
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_path, y_train_label))
    val_ds   = tf.data.Dataset.from_tensor_slices((X_test_path, y_test_label))

    #Building Pipeline
    train_ds = train_ds.cache().map(load_and_preprocess_image).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().map(load_and_preprocess_image).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds

def densenet_model_maker(base_model_3):

  # Preprocess_input
  inputs = keras.Input(shape=(224, 224, 3))
  # x = data_augmentation(inputs)
  x = tf.keras.applications.densenet.preprocess_input (inputs)
  x = base_model_3(x, training=False)

  # Connect to your base model
  x = keras.layers.GlobalAveragePooling2D()(x)  

  # Add your own classifier  
  x = keras.layers.Dense(256, activation='relu')(x)
  x = keras.layers.Dropout(0.2)(x)
  outputs = keras.layers.Dense(2, activation='sigmoid')(x) # 2 loại cảm xúc

  model = keras.Model(inputs, outputs)

  return model

def make_user_spectrogram():
    ######################################
    ########### Running ##################
    ######################################
    # Paths for data.
    User_Root_Path = "user_data"

    #Create Data Set Path And Emotions
    user_directory_path = split_and_make_spec_data_path(User_Root_Path)
    user_file_emotions = []
    user_directory_path = [str(path) for path in user_directory_path]

    print(len(user_directory_path))

    #Trainning Model
    all_path_result = user_directory_path
    all_label_index = []
    CLASS_NAME = ['negative', 'positive']
    for path in all_path_result:
        tmp = path.split('.')[0]
        emotion = tmp.split('_')[-1]
        if emotion == CLASS_NAME[1]:
            all_label_index.append(1)
        elif emotion == CLASS_NAME[0] :
            all_label_index.append(0)

    assert (len(all_path_result)) == (len(all_label_index))

    return all_path_result

# #Building Tensor Trainset and Valset
# train_ds, val_ds = prepare_dataset(all_path_result,all_label_index )

#Modelling:
# USER_MODEL = 'user_model'

# checkpoint_callback = ModelCheckpoint(filepath=str(USER_MODEL)+"/user_model_checkpoint.h5",
#                                       save_weights_only=False, # the whole model (False) or only weights (True) 
#                                       save_best_only=True, # keep the best model with lowest validation loss
#                                       monitor='val_loss',
#                                       verbose=1)

# earlystopping_callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     min_delta=1e-2, # "improving" threshold 0.01
#     patience=15,
#     verbose=1)

# restored_model = tf.keras.models.load_model('model\\binary_model_checkpoint.h5')
# # Unfreeze the base_model.
# restored_model.trainable = True

# base_model = restored_model.layers[4]

# # Fine-tune from 
# fine_tune_at = 700

# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#   layer.trainable =  False

# print(restored_model.summary())

# restored_model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# epochs=10
# history = restored_model.fit(train_ds,               # <<<<<<<
#                     validation_data=val_ds, # <<<<<<<
#                     epochs=epochs, callbacks=[checkpoint_callback, earlystopping_callback])

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# print('acc:', acc)
# print('val_acc:', val_acc)

# ############## END OF RUNNING #################










