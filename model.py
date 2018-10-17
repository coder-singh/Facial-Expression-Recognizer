from keras.models import model_from_json
import numpy as np

class FERModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Sad", "Surprise",
                     "Neutral"]

    '''
    Class initialization takes 2 parameters
        Model Architecture (JSON)
        Model Weight (h5)

    This loades model from JSON file and sets h5 weights to it.
    These weights are learned through Training.
    '''
    def __init__(self, model_json_file, model_weights_file):

        # load model architecture from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load model weights and set it to the newly loaded model
        self.loaded_model.load_weights(model_weights_file)

        print("Model loaded from disk")
        #self.loaded_model.summary()
        

    '''
    predict_emotion function takes 1 argument
        Image file

    This function uses Keras model's predict method.
    output class <- model.predict(input matrix)

    Output layer contains 7 outputs of emotions,
    np.argmax(preds) returns the index with highest prediction value
    This index is passed to EMOTIONS_LIST to get a string output
    '''
    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)

        return self.EMOTIONS_LIST[np.argmax(self.preds)]


if __name__ == '__main__':
    pass