from tensorflow.keras.models import model_from_json
import numpy as np



class FacialExpressionModel(object):

    EXPRESSIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Suprise"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file,"r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded = model_from_json(loaded_model_json)

        self.loaded_model.load_weight(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_expression(self, img):
        self.predictions = self.loaded_model.predict(img)
        return FacialExpressionModel.EXPRESSIONS[np.argmax(self.predictions)]