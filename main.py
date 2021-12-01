import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fuctions import get_positive_text, countdown, get_short_text, record_with_duration, remove_file_in_tmp, save_audio, split_and_save_audio, predict_audio
import sounddevice as sd
from IPython.display import Audio
import librosa, librosa.display
import uuid
import time
import pathlib
from train_user_model import make_user_spectrogram 

# Global Variable
# USER_DATA = pd.DataFrame() #file lưu trữ đường dẫn âm thanh và label của người dùng
SAMPLE_RATING = 22050
PREDICTED_MOOD = ''
E_POSITIVE = 'positive'
E_NEGATIVE = 'negative'
MODEL_PATH = ''

#Menu Of Page
menu = ['Home', 'Trainning Me', 'User Data' ,'History','Recomendation']

choice = st.sidebar.selectbox('Vui Lòng Chọn', menu)
rec_duraton = st.sidebar.slider('Duration:',5,30, 5)
print(rec_duraton)

model_selection = st.sidebar.radio('Model', ['General Model', 'Personalized'])
if model_selection == 'Personalized':
    MODEL_PATH = 'user_model\\user_binary_model_checkpoint_v2.h5'
else:
    MODEL_PATH = 'model\\binary_model_checkpoint.h5' 

clean_user_data = st.sidebar.button('Clean User Data')
if clean_user_data:
    remove_file_in_tmp('user_data_slice')
    remove_file_in_tmp('user_data')
    remove_file_in_tmp('user_data_spec')
    
if choice == 'Home':
    st.title("VSR - Voice Sentimental Recognization")
    st.write("The application helps you understand your feelings through your voice")
    st.write('\n')

    st.write("Try to see what state your voice is in:")
    rec_button = st.button('Press To Record')
    output_path = 'tmp.wav'

    if rec_button:      
        # text = get_short_text()        
        # st.write(text)
        # st.write('words:', len(text.split()))        
        # duration = st.number_input('Enter duration you want to record:', value=len(text.split())//2.8)
        # print(duration)
        st.write('Samle text 1: Lòng biết ơn không chỉ là đức tính vĩ đại nhất mà còn là khởi nguồn của mọi đức tính tốt đẹp khác')
        st.write('Sample text 2: Sen là loại cây cảnh đẹp, được nhiều người ưu thích. Cây thường được trồng làm cây cảnh ngoại thất, trồng trong ao hồ nhân tạo hay tự nhiên.')
        with st.spinner(text='Recoding will start in 2 seconds...'):
            time.sleep(2)
            st.success('Start Recording')

        myrecording = record_with_duration(st, rec_duraton) 
        #sd.play(myrecording, SAMPLE_RATING)
        print('my recording', myrecording.shape)

        #Saving file
        save_audio(myrecording,output_path)   
        # output_path = 'user_data\\user_a741fdb8-3013-11ec-b9fe-30d042099905_positive.wav' #Test with trainning data

        #display waveplot
        fig = plt.figure(figsize=(10,5))
        data, sr = librosa.load(output_path, SAMPLE_RATING)
        librosa.display.waveplot(data, sr=SAMPLE_RATING)    
        st.pyplot(fig)
        st.audio(output_path, format='wav')

        with st.spinner(text='Prediction progress...'):
            #Cut And Predict
            export_path = 'predict_tmp'
            #Remove All File In Tmp Export Path
            path_list = split_and_save_audio(output_path, 0, duration=4000, export_path=export_path)
            print(len(path_list))
            path_df = pd.DataFrame(path_list, columns = ['Path'])
            emotions = []
            idx_cols = st.beta_columns(len(path_list)) 
            idx = 0                
            for path in path_list:
                prediction, spec_path = predict_audio(MODEL_PATH, path)
                print(prediction)
                idx_cols[idx].image(spec_path)
                idx += 1
                emotions.append(prediction)
            
            tmp_emotion_df = pd.DataFrame(emotions, columns=['Negative_Proba', 'Positive_Proba'])
            Full_df = pd.concat([path_df, tmp_emotion_df], axis=1)
            st.dataframe(Full_df)
            mean_positive = Full_df['Positive_Proba'].mean()
            mean_negative = Full_df['Negative_Proba'].mean()
            st.write('Probility of Positive:', mean_positive )
            st.write('Probility of Negative:', mean_negative )
            if mean_positive > mean_negative:
                PREDICTED_MOOD = 'POSITIVE'
            else:
                PREDICTED_MOOD = 'NEGATIVE'
            st.header('Your curret mood: ', PREDICTED_MOOD)
            st.success('Done')


elif choice=='Trainning Me':
    st.title("SSR - Speech Sentimental Recognization")
    st.write("The application helps you understand your feelings through your voice")
    st.write('\n')
    st.header('Now Please Help Me Understand More About You?')
    st.subheader('How do you feel right now ?')
    current_mood = st.selectbox('Select', 
                ["Please Choose", "Great","Good","Bad","Realy Bad"])

    if current_mood == 'Vui lòng chọn':
        # st.write(current_mood)
        print(current_mood)

    elif (current_mood == "Tốt") or (current_mood == "Tuyệt vời"):
        st.write('Your current mode:', current_mood)
        st.write('Thật tuyệt vời hãy cùng ghi âm để giúp tôi hiểu bạn nhé')
        rec_button = st.button('Press To Record')        

        if rec_button:
            text = get_short_text()        
            st.write(text)
            st.write('words:', len(text.split()))        
            duration = st.number_input('Enter duration you want to record:', value=len(text.split())//2.8)
            print(duration)
            with st.spinner(text='Recoding will start in 3 seconds...'):
                time.sleep(3)
                st.success('Start Recording')

            my_recording = record_with_duration(st, duration)
            #Saving file to tmp
            tmp_path = 'tmp.wav'
            save_audio(my_recording,tmp_path)   
            st.write('Saving completed, now playing')

            #display waveplot
            fig = plt.figure(figsize=(10,5))
            data, sr = librosa.load(tmp_path, SAMPLE_RATING)
            TMP_DATA = data
            librosa.display.waveplot(data, sr=SAMPLE_RATING)    
            st.pyplot(fig)
            st.audio(tmp_path, format='wav')

            # #Predict button
            # st.header("Bạn Có Muốn Predicit File Âm Thanh Này ?")
            # rec_button = st.button('Predict')        

        #display file name, label
        st.header("Bạn Có Muốn Lưu Lại File Này ?")
        label = E_POSITIVE
        st.write("Label of this audio file: " + label)       
        
        #Save Button    

        save_button = st.button('Save File')
                
        if save_button:
            ## Setting output Path
            userdata_path = 'user_data'       
            file_name = "user_" + str(uuid.uuid1()) + '_' + label + '.wav'         
            full_path = userdata_path + '\\' + file_name        
            
            print('saving...')
            data, sr = librosa.load('tmp.wav', SAMPLE_RATING)
            save_audio(data, full_path)

            st.success('File Saved')
            st.write(full_path)
            print(full_path)

    elif (current_mood == "Không Tốt Lắm") or (current_mood == "Rất Tệ"):
        
        st.write('Your current mode:', current_mood)
        label = E_NEGATIVE
        st.write('Oh, 7 Tỷ Người Đều Trải Qua Những Ngày Như Vậy.')
        st.write('Hãy giúp cho tôi biết giọng nói của bạn ở trạng thái này nhé')
        rec_button = st.button('Press To Record')        

        if rec_button:
            text = get_short_text()        
            st.write(text)
            st.write('words:', len(text.split()))        
            duration = st.number_input('Enter duration you want to record:', value=len(text.split())//3)
            print(duration)
            with st.spinner(text='Recoding will start in 3 seconds...'):
                time.sleep(3)
                st.success('Start Recording')

            my_recording = record_with_duration(st, duration)
            #Saving file to tmp
            tmp_path = 'tmp.wav'
            save_audio(my_recording,tmp_path)   
            st.write('Saving completed, now playing')

            #display waveplot
            fig = plt.figure(figsize=(10,5))
            data, sr = librosa.load(tmp_path, SAMPLE_RATING)            
            librosa.display.waveplot(data, sr=SAMPLE_RATING)    
            st.pyplot(fig)
            st.audio(tmp_path, format='wav')   

        #display file name, label
        st.header("Bạn Có Muốn Lưu Lại File Này ?")        
        st.write("Label of this audio file: " + label)       
        
        #Save Button    

        save_button = st.button('Save File')
                
        if save_button:
            ## Setting output Path
            userdata_path = 'user_data'       
            file_name = "user_" + str(uuid.uuid1()) + '_' + label + '.wav'         
            full_path = userdata_path + '\\' + file_name        
            
            print('saving...')
            data, sr = librosa.load('tmp.wav', SAMPLE_RATING)
            save_audio(data, full_path)

            st.success('File Saved')
            st.write(full_path)
            print(full_path)

elif choice == 'User Data': 
    st.title("SSR - Speech Sentimental Recognization")
    st.write("Ứng dụng giúp bạn hiểu rõ cảm xúc của mình thông qua giọng nói")
    st.write('\n')
    st.header('Dưới đây là các hình ảnh spectrogram của bạn')
    col1, col2 = st.beta_columns(2)
    col1.header('Negative Class')
    col2.header('Positive Class')
    
    # user_directory_list = pathlib.Path('user_data_spec')
    # user_directory_path = list(user_directory_list.glob('*'))
    # user_directory_path = [str(path) for path in user_directory_path]

    user_directory_path = make_user_spectrogram()

    for path in user_directory_path:
        tmp = path.split('.')[0]
        emotion = tmp.split('_')[-1]
        if emotion == 'negative':
            col1.image(path)
        else:
            col2.image(path)

if choice == 'Recomendation':
    st.title("SSR - Speech Sentimental Recognization")
    st.write("Ứng dụng giúp bạn hiểu rõ cảm xúc của mình thông qua giọng nói")
    st.write('\n')
    PREDICTED_MOOD = 'negative' #for testing
    st.header('Có phải trạng thái của bạn hiện là: ' + PREDICTED_MOOD)   

    if PREDICTED_MOOD == 'negative':
        st.write('Có 280 triệu người trên thế giới đang đồng cảm với bạn:')
        st.write('Hãy thử các liệu pháp sau:')
        col1,col2,col3 = st.beta_columns(3)
        col1.image('media\\reading-book.jpg')
        