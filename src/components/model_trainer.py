import os
import sys
from dataclasses import dataclass

# Regression models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom modules for exception handling, logging, and utilities
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


# =========================
# CONFIGURATION CLASS
# =========================
@dataclass
class ModelTrainerConfig:
    """
    Stores configuration related to model training
    """
    # Path where the best trained model will be saved
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# =========================
# MODEL TRAINER CLASS
# =========================
class ModelTrainer:
    """
    Handles training, evaluation, and saving of multiple regression models
    """

    def __init__(self):
        # Initialize configuration for saving the model
        self.model_trainer_config = ModelTrainerConfig()

    # =========================
    # TRAIN & SELECT BEST MODEL
    # =========================
    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple regression models, evaluates them on test data,
        selects the best model, and saves it as a pickle file.

        Parameters
        ----------
        train_array : np.array
            Training data array (features + target)
        test_array : np.array
            Testing data array (features + target)

        Returns
        -------
        float
            R2 score of the best model on test data
        """
        try:
            logging.info("Splitting training and test input & target data")

            # Split train and test arrays into X (features) and y (target)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # all columns except last
                train_array[:, -1],   # last column = target
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define all candidate regression models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameter grids for model tuning
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},  # no hyperparameters
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            logging.info("Evaluating all candidate models with hyperparameter tuning")

            # Train, evaluate, and return a dictionary of model names -> scores
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Get the best R2 score
            best_model_score = max(sorted(model_report.values()))

            # Get the name of the best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Raise exception if no model is good enough
            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 > 0.6")

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            # Save the best model as a pickle file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best model saved at: {self.model_trainer_config.trained_model_file_path}")

            # Predict on test data
            predicted = best_model.predict(X_test)

            # Calculate R2 score for the best model
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            # Wrap any exception into CustomException with sys info
            raise CustomException(e, sys)
