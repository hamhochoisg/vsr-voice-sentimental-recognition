# VOICE SENTIMENTAL RECOGITION (VSR)
## MOTIVATION - WHY VSR
According to WHO, about 5% of people worldwide suffer from mental health diseases. It's about to 280 million people. Inspired by the research of David R. Hawkins through the book Power and Force and water experiment of Masaru Emoto, VSR was born with the hope of being able to detect the user's mental status through the vibrations of the voice.

![image](https://user-images.githubusercontent.com/88182498/138410971-a0560bf0-2e2d-41f5-87b1-399ce4a21942.png)

## APP MILE STONE
VSR have three main fuctions: 
* Predict User Voice Emotion 
* Personalize with User Data
* Recommendation to improve mental health 
![image](https://user-images.githubusercontent.com/88182498/138417771-e1e90310-1c4a-45e2-80f5-3e70856f4fc6.png)


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

## MACHINE LEARNING SOLUTION

To solve this problem, the machine learning approach is learning from a large amount of emotionally classified data to learn and find the right features. Then model will base on this knowledge to predict user's spectrograms. The model will needs more user data to be able to understand the user correctly. Personalization is definitely needed as user-side recordings use very different equipment from the lab.

### DATASET FOR MODEL

Most of the data below was created in a lab, professional actors were asked to say a few neutral sentences with differences emotions.

1. Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D) 
+ 7,442 audio data
+ 91 actors (20-74 years old), various ethnicities
+ 06 emotions

2. Toronto emotional speech set (TESS)
+ 2800 audio data
+ 02 actors (26 and 64 years old)
+ 08 feelings

3. Surrey Audio-Visual Expressed Emotion (SAVEE)
+ 1012 audio data
+ 4 actors (27-31 years old)
+ 07 emotions

4. Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
+ 1440 audio data
+ 24 actors (12 female, 12 male)
+ 07 emotions

![image](https://user-images.githubusercontent.com/88182498/138420600-aebc8ad0-c470-4193-87ca-2e75dbb8dc88.png)

Source and license: https://www.kaggle.com/shivamburnwal/speech-emotion-recognition/data
### MODELING AND TRAINNING
After build my own model to test and do not have good scores (lower than 60% accuracy). I decied to use tranfer learning to inherit model architechture and weights. I uses Densenet Model which includes 4 Dense block and a classification Layer. I also benchmark with other models: Resnet50, Xception and result for Densenet201 is better with validation lost: 0.96666

![image](https://user-images.githubusercontent.com/88182498/138421212-a7a8ddd0-bbcf-4288-9015-d92753efebd1.png)

#### Why Dense Net
![image](https://user-images.githubusercontent.com/88182498/138420786-79dc1c16-c2dd-4ecd-b3c3-a93f251e8b3e.png)
I use this model beause it get all the previous layers as input so it's very diverse in features so it maintaining features is not complicated and it is very suitable for a data problem with little difference like this case. Besides, this model has been pretrained with imagenet competition with top 5 accuracy = 0.936 

For detail about DenseNet model you can read this article: [Towardsdatascience.com & Cornell University](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)

### TRANSFER LEARNING
The purpose of this application is to analyze only 2 class of emotions: negative and positive. Negative includes emotions such as anger, fear, disgust and positive emotions include neutral, happy, calm, pleasant surprise.

Therefore, the Densenet201 model is removed the classification class and replaced by the GlobalAveragePooling2D class and the Binary Classifier (Positive & Negative) class. The model weights are also retained to utilize the knowledge trained on imagenet.

Model is opened and train 7 more classes with trainable parameter: 774,914

This is the summary of model structure before and after tunning: 
![image](https://user-images.githubusercontent.com/88182498/138422401-6d5a9884-22e1-4a6a-b5cb-0787f022a044.png)

This is the result of model trainning:
![image](https://user-images.githubusercontent.com/88182498/138424061-b86e8771-dac4-42e1-95d7-b7f584fc9ec7.png)

## APP SCREENSHOT
I use streamlit to present my app and this is some screenshot:

### Predict Mode
First user will need to record their voice, they can say anything or read the sample text:
![image](https://user-images.githubusercontent.com/88182498/138424712-bc5fa98e-dcdb-4a0a-ad60-6c37099bf5a0.png)

Then system save the audio and plot the sound wave:
![image](https://user-images.githubusercontent.com/88182498/138424872-632ce3cd-4600-48ca-b8ba-fb7130bf2932.png)

Then a list of spectrogram (4s per image) is created to predict user voice emotion:
![image](https://user-images.githubusercontent.com/88182498/138425151-fdf54951-1a1b-47d8-853e-db1e4b412001.png)

### Trainning Mode
For the trainning mode, user will able to record, save file and preview their spectrograme files. We cannot train userdata on their devices so now I just collect user data and train form another machine. In future plan, I will try to connect clound server so user and send data and make a trainning request.

![image](https://user-images.githubusercontent.com/88182498/138425648-6bd2012c-9420-47ee-bb9d-65fff9ce5518.png)

User data
![image](https://user-images.githubusercontent.com/88182498/138425835-597a742e-ebdc-49c1-9563-e61222d0b04c.png)

### Why Personalize Importance
The microphone of user devices are very difference with lab environment so it very hard for model to predict. With user data combine with model data will be the best practices for this soluton.

Besides, I do believe that only users understand themselves best

## FUTURE PLAN
* Combined face recognition (MediaPie)
* Incorporating language processing (NLP)
* Store mental health history and report
* Solve the problem of connecting to the cloud server to personalize the model
* Build health improvement solutions based on condition

## FILE EXPLORER
updating...

# REFERENCE
* [Speech Emotion Recognition by SHIVAM BURNWAL] (https://www.kaggle.com/shivamburnwal/speech-emotion-recognition/data)
* [DenseNet Model Review by Sik-Ho Tsang](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)

