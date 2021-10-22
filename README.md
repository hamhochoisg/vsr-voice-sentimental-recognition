# VOICE SENTIMENTAL RECOGITION (VSR)
## MOTIVATION - WHY VSR
According to WHO, about 5% of people worldwide suffer from mental health diseases. It's about to 280 million people. Inspired by the research of David R. Hawkins through the book Power and Force and water experiment of Masaru Emoto, VSR was born with the hope of being able to detect the user's mental status through the vibrations of the voice.

![image](https://user-images.githubusercontent.com/88182498/138410971-a0560bf0-2e2d-41f5-87b1-399ce4a21942.png)

## METHODOLOGY
![image](https://user-images.githubusercontent.com/88182498/138412436-ecf35440-65c3-4778-914d-5c2ae1a6549d.png)

First step, the audio file is converted into a sound wave by **sampling technique** to form a data array that is saved in the form of a matrix (np.array).

Second step, the sound wave is then converted into a frequency form, using the **Fourier Transform technique** so that it can be decomposed into different sound layers. The result of this step is also stored in matrix form.

Final step, the result of the Fourier Transform is converted into a Spectrogram representing the audio frequency, time and magnitude through an image.

The audio files are cut into 4 seconds segments and export into the same color range of the spectrogram with the parameters to output the file as: 

SAMPLE_RATING = 22050
FMIN= 0.0 : Minimun Frequency 
FMAX=20000.0 : Maximun Frequency 
MAX_DB = 100 : Maximun Decibel
WIN_LENGTH = N_FFT = 2048 : Windows size

Sample Spectrogram for Happy Class and Sad Class
![image](https://user-images.githubusercontent.com/88182498/138415184-1c9bb017-a764-4780-af35-ca38b746cc98.png)

### Why Spectrogram ?

After researching similar solutions on Kaggle, I want to test using Machine Learning models that have been very successful in the imagenet competition specializing in image classification to solve this problem with the hypothesis that the use of pretrained models will support in the analysis of images in spectrograms.

However, I also test with extract features methodology with lower rating. In near future, I will continue to test more this method. 

## MACHINE LEARNING MODEL
After build my own model to test and have not good score (lower thang 60% accuracy). I decied to use tranfer learning to inherited 

test 

## FUTURE PLAN

## FILE EXPLORER
