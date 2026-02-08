import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


# =========================
# Prediction Pipeline
# =========================
class PredictPipeline:
    """
    Loads the trained model and preprocessor, applies preprocessing, and predicts.
    """
    def __init__(self):
        # Paths to saved artifacts
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        """
        Predicts target values for the input features.

        Parameters:
        -----------
        features : pd.DataFrame
            Input features with same columns as training data.

        Returns:
        --------
        preds : np.array
            Predicted target values
        """
        try:
            print("Loading model and preprocessor...")

            # Load trained model
            model = load_object(file_path=self.model_path)

            # Load preprocessor
            preprocessor = load_object(file_path=self.preprocessor_path)

            print("Transforming input features...")
            # Apply preprocessing
            data_scaled = preprocessor.transform(features)

            print("Predicting...")
            # Make predictions
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            # Wrap exceptions in CustomException
            raise CustomException(e, sys)


# =========================
# CustomData Class
# =========================
class CustomData:
    """
    Converts user inputs into a DataFrame for prediction.
    """
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Converts the input data into a single-row DataFrame
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert dictionary to DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            return df

        except Exception as e:
            raise CustomException(e, sys)