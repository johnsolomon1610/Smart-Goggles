# Smart-Goggles

With the help of modern technology ( Machine Learning ) we build a system that converts visual inputs into audio signals which lead us to design a product to help the blind (visually impaired) people to identify person and to navigate.

The proposed system is designed an prototype for blind people to help them do their regular activities and engage them with normal people. This project represents a Open-CV 
based person’s face recognition system using several algorithm’s of image processing, aimed for visually impaired people. This device detects and recognize human faces from 
the camera in the form of several frames, fetches information from input frames and check the information with the database which is already trained image of the human faces 
by the system and spells name through a microphone applying text to speech method. If these information does not match with the database,the system ask the people to store the information in the database through speech and store it in the folder with people name. For the next time they automatically detect and recognise the people and gives a output through voice. In the experimental case, the system gives 92% accuracy recognizing single person.Later the system train a lot of images of the people,they give more accuracy
output in the real time suitation and the blind people live like a normal people in the society.

### **MODULES :**
- Face detection and Recognition
- Text to speech recognition
- General modules

### **Face Detection :** 
The face detection is generally considered as finding the faces (location and size) in an image and probably extract them to be used by the face detection algorithm.(Caffe model).

### **Face Recognition :** 
The face recognition algorithm is used in finding features that are uniquely described in the image. The facial image is already they are extracted , cropped , resized, and usually converted in the grayscale.(Pytorch , Imutils , Sklearn-encoder , SVC, open-cv ).

### **Text to Speech Recognition :**
Text-To-Speech is a process in which input text is first analyzed, then processed and understood, and then the text is converted to digital audio and then
Spoken ( Pyttsx3 , Gtts , Speech recognition ).

### **General Modules :**
These modules support for the image processing like converting the input into several frames and these frame processing with the already trained images in the
dataset and recognition of the image with the database. (numpy , pickle ,os).

