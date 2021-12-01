import uuid
import pandas as pd
import numpy as np
import sounddevice as sd
import time
import streamlit as st

### import for Streaming Record
import argparse
import tempfile
import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

import os
import glob
from pydub import AudioSegment

import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
import librosa
import librosa.display
import matplotlib.pyplot as plt

import random

## Global Variable
# # MODEL_PATH = 'model\\binary_model_checkpoint.h5' 
# MODEL_PATH = 'user_model\\user_binary_model_checkpoint.h5'

SAMPLE_RATING = 22050
FMIN=0.0 #Tần số thấp nhất
FMAX=20000.0 #Tần số nghe cao nhất của con người
MAX_DB = 100 # độ lớn âm thanh lớn nhất 
WIN_LENGTH = N_FFT = 2048
HOP_LENGTH= 512 # win_length / 4
IMG_SIZE = 224
###

def get_positive_text():
    #long_text_1 = 'Bạn càng thường xuyên mở cửa lòng mình, thì dòng chảy năng lượng tuôn trào trong bạn càng nhiều. Đôi khi dòng năng lượng thâm nhập vào bạn nhiều đến mức nó bắt đầu tràn ra khỏi bạn. Bạn cảm thấy nó như một con sóng tràn qua bạn. Bạn có thể thực sự cảm nhận nó tràn qua đôi tay bạn, qua tim bạn và xuyên qua các trung tâm năng lượng khác. Theo đó, tất cả các trung tâm năng lượng này đều mở ra, và một lượng lớn năng lượng bắt đầu tràn ra khỏi bạn. Từ đó, năng lượng này bắt đầu tác động đến những người khác. Mọi người có thể thu nhận năng lượng từ bạn, và bạn đang dùng dòng chảy năng lượng này để nuôi dưỡng họ. Nếu bạn sẵn sàng mở lòng nhiều hơn nữa thì nguồn năng lượng này không bao giờ bị ngưng tắt. Bạn sẽ trở thành nguồn sáng cho tất cả những người xung quanh bạn.'
    long_text_2 = "...một khi anh cổ vũ cho tư tưởng yêu thương trong đầu anh, một khi những tư tưởng ấy đã hoạt động, khi ấy thái độ của anh đối với người khác sẽ tự động thay đổi. Nếu anh tiếp cận người khác với tư tưởng yêu thương, điều đó lập tức sẽ làm giảm sự sợ hãi và cho phép anh mở lòng với người khác. Nó tạo ra bầu không khí tích cực, thân thiện."
    long_text_2 =""
    return long_text_2

def get_short_text():
    #text_1 = "Hãy trau dồi thói quen biết ơn mọi điều tốt lành đến với bạn, và hãy cảm tạ thật thường xuyên. Bởi vì mọi điều đều giúp bạn tiến lên phía trước. Hãy đem mọi thứ đặt trong lòng biết ơn.” Raphal Waldol Ememson"
    df = pd.read_csv("positive_text.csv")
    idx = random.randint(0,df.shape[0])
    text = df.iloc[idx,0]
    return text

# text = get_short_text()
# print(text)

def countdown(fig, t=10):    

    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        fig.write(timer, end="\r")
        time.sleep(1)
        t -= 1
      
    # fig.write("Times'up")

def record_with_duration(stream, duration):
    myrecording = sd.rec(int(duration * SAMPLE_RATING), samplerate=SAMPLE_RATING, channels=2)
    sd.wait()
    print(myrecording.shape)
    stream.write('Recording completed, now playing')

    return myrecording

def save_audio(data, output_path):
    #Input: numpy data of audio
    #path for out_put
    #Saving file
    sf.write(output_path, data, SAMPLE_RATING)   

    return output_path

def split_and_save_audio(audio_path, beginSecond, duration=4000, export_path=None, endSecond=None, code='a', user_trainning = False): #cắt ra 5s một bài
    #Input: wave file path
    #Output: List of new data, remove last one
  if export_path != None and (os.path.isdir(export_path) == False ):
    os.mkdir(export_path)

  SongOrigin = AudioSegment.from_wav(audio_path)
  durationSecond = SongOrigin.duration_seconds

  path_list = []

  if endSecond == None :
    endSecond = durationSecond;  
  elif endSecond > durationSecond:
    endSecond = durationSecond
  

  t1 = beginSecond * 1000 # Works in milliseconds
  end = endSecond * 1000
  export_path = export_path + '\\'
  while (t1 + duration) < end:
    t2 = t1 + int(duration) # 1 lần cắt ra 5s
    tmp_newAudio = SongOrigin[t1:t2]

    if user_trainning:
        file_name = audio_path.split('\\')[-1]
        file_name = file_name.split('.')[0] # vd: user_f52ffa13-3015-11ec-a9f3-30d042099905_negative . wav           
        tmp_name = str(code) + '_'  + '-' + str(t1) + "-" + str(t2) +'_' + file_name + '.wav'  
    else:
        tmp_name = str(code) + '-' + str(t1) + "-" + str(t2) +'.wav'  
   
    tmp_newAudio.export(export_path + tmp_name, format="wav") #Exports to a wav file in the current path.      
    path_list.append(export_path + tmp_name)
     
    t1 = t1 + int(duration)
  return path_list

def remove_file_in_tmp(path):
    files = glob.glob(path + '/*', recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

def create_melspectrogram(data, emotion="", output_path='tmp'):

    #Create Mel Spectrogram, include Fourier Transform Steps
    mels = librosa.feature.melspectrogram(
        y=data, sr=SAMPLE_RATING,
        #win_length=win_length,
        hop_length=HOP_LENGTH,    
        fmin= FMIN,
        fmax=FMAX,
        power = 1,
        n_fft = N_FFT
    )

    mel_sgram = librosa.amplitude_to_db(
        mels, 
        ref=np.min, 
        top_db=MAX_DB)

    fig = plt.figure(figsize=(8,6))

    librosa.display.specshow(
        mel_sgram, 
        sr=SAMPLE_RATING, 
        hop_length=HOP_LENGTH, 
        x_axis='time', 
        y_axis='hz',
        fmin = FMIN,
        fmax = FMAX,
        
    )
    #plt.colorbar()
    plt.axis('off') 

    output_name = 'spec' + '_' + emotion +'.jpg'
    print(output_name)

    plt.savefig(output_path + output_name ,format='jpg' , transparent=False, dpi=150)
    #plt.close(fig)
    return output_path + output_name

def create_melspectrogram_in_bg(data, emotion='', output_path='tmp', file_name=''):

    #Create Mel Spectrogram, include Fourier Transform Steps
    mels = librosa.feature.melspectrogram(
        y=data, sr=SAMPLE_RATING,
        #win_length=win_length,
        hop_length=HOP_LENGTH,    
        fmin= FMIN,
        fmax=FMAX,
        power = 1,
        n_fft = N_FFT
    )

    mel_sgram = librosa.amplitude_to_db(
        mels, 
        ref=np.min, 
        top_db=MAX_DB)

    fig = plt.figure(figsize=(8,6))

    librosa.display.specshow(
        mel_sgram, 
        sr=SAMPLE_RATING, 
        hop_length=HOP_LENGTH, 
        x_axis='time', 
        y_axis='hz',
        fmin = FMIN,
        fmax = FMAX,
        
    )
    #plt.colorbar()
    plt.axis('off')  

    output_name = "spec"
    if len(file_name) > 0:
        output_name = output_name + '_' + file_name    
    output_name = output_name + '_'  + emotion +'.jpg'
    #print(output_name)
    #Save to folder
    output_path = output_path + '\\'

    plt.savefig(output_path + output_name ,format='jpg' , transparent=False, dpi=150)
    plt.close(fig)

    return output_path + output_name

def predict_audio(MODEL_PATH, audio_path):
    #Input: path of audio
    #Output: list of emotion probability 
    
    #Load model    
    restored_model = tf.keras.models.load_model(MODEL_PATH)
    
    data , sr = librosa.load(audio_path, duration = 4)
    print(data.shape)

    output_path ='tmp'
    file_name = str(uuid.uuid1())

    image_path = create_melspectrogram_in_bg(data, output_path, file_name=file_name)
    print(image_path)

    img        = tf_image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array  = tf_image.img_to_array(img)
    img_array  = np.expand_dims(img_array, axis=0)
    prediction = restored_model.predict(img_array)
    #print(list(prediction))

    return list(prediction)[0], image_path

# sample_path = "predict_tmp\\track-0-4000.wav"
# prediction = predict_audio(sample_path)

# print("negative_proba:", prediction[0])
# print("positive_proba:", prediction[1])


# MODEL_PATH = 'user_model\\user_binary_model_checkpoint_v1.h5'
# # MODEL_PATH = 'model\\binary_model_checkpoint.h5' 
# audio_path = 'tmp.wav'
# prediction = predict_audio(MODEL_PATH, audio_path)

# print(prediction)