# Facial Expression Recognition
It is a python based application which is able to recognize human facial expressions. This application successfully detects facial expressions of the people present in the image loaded by the user. The user is also able to detect faces and recognize its expression in Real-time using a camera.    

This project uses CNN (Convolutional Neural Network) model. Two CNN models are trained for this application:
* Shallow Model  
Low Accuracy but Hig prediction rate (good for real-time prediction)
* Deep Model  
High Accuracy but Low prediction rate (not suitable for real-time prediction)

### Dependendencies:
* Python3
* Python Libraries:
  * Opencv  
  * Keras
  * Tensorflow
  * numpy
  * pyQT5

_Note: Camera is required for this application ( if user wish to use real-time functionality )_

## General Working:
### Training Dataset:
FER2013 dataset was used to train the CNN model. This dataset is freely available on [Kaggle](www.kaggle.com) for academic purposes.  
  
A summary about the dataset:
* Contains 48x48 pixel human face image with expression label
* 28,709 images in training set
* 3,589 images in test set
* Expression label contains 7 emotions
  * Angry
  * Disgust
  * Fear
  * Happy
  * Sad
  * Surprise
  * Neutral

I trained the model harnessing GPU's power which speeded up the training process.  
_I'll be uploading a Notebook guide on training the CNN model for this project shortly._
### Saved Model and Weights:
Once the model is constructed, it is saved to an external file using the following code in python:
```
model_json = model.to_json()
with open("face_model.json", "w") as json_file:
    json_file.write(model_json)
```
After the complete training of the model, trained model weights are saved in the external file with extension .h5 using the following python code:
```
checkpointer = ModelCheckpoint(filepath='face_model.h5', verbose=1, save_best_only=True)
```
### Prediction on an image:
Using OpenCV's CascadeClassifier, faces are detected in the image. Every face then is converted to 48x48 pixel, as the input dimension must match with the CNN model.  
These 48x48 pixels are passed to CNN model and using Keras predict function, we get 7 values in the output node, each representing score for each expression.
```
predictions = model.predict(image)
```
To obtain the index of highest output node, we use numpy's argmax function
```
numpy.argmax(predictions)
```
## Using the Application
### Running the application:
Clone/Download this project, cd into project directory, and type the following command
```
python final.py
```
### A walkthrough:
At the top of the application interface, User should select any one model, "deep" or "shallow". (An error will be encountered if model is not selected and other functions are accessed)  
There are 6 buttons in the app:
* **Open Camera:** 
Opens the Camera (if available) for the purpose of real-time facial expression detection.
* **Close Camera:** Closes the Camera.
* **Load Image:** Lets user to load an image from the disk. The image may contain 0, 1, or multiple faces, app would detect the faces and predict an expression for every face detected.
* **Show Image:** Shows the loaded image using Matplotlib's pyplot method.
* **Image Prediction:** Applies CNN prediction to the image loaded.
* **CloseApplication:** Closes the application