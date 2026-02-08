import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


# =========================
# SAVE OBJECT (MODEL / PREPROCESSOR)
# =========================
def save_object(file_path, obj):
    """
    Save any Python object (model, preprocessor, etc.) to disk as a pickle file.

    Parameters:
    -----------
    file_path : str
        Path where the object should be saved (including file name).
    obj : any
        Python object to save (e.g., model, pipeline, preprocessor).
    """
    try:
        # Extract directory path from file path
        dir_path = os.path.dirname(file_path)

        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # Wrap exceptions in CustomException with system info
        raise CustomException(e, sys)


# =========================
# LOAD OBJECT (MODEL / PREPROCESSOR)
# =========================
def load_object(file_path):
    """
    Load a Python object from a pickle file.

    Parameters:
    -----------
    file_path : str
        Path to the pickle file to load.

    Returns:
    --------
    obj : any
        The loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# =========================
# EVALUATE MODELS
# =========================
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains and evaluates multiple regression models.
    - Uses GridSearchCV for sklearn models.
    - Skips GridSearchCV for CatBoost to avoid sklearn compatibility errors.
    - Returns R2 scores of each model on the test data.

    Parameters:
    -----------
    X_train, y_train : np.array
        Training data and target.
    X_test, y_test : np.array
        Test data and target.
    models : dict
        Dictionary of model name -> model instance.
    param : dict
        Dictionary of model name -> hyperparameter grid.

    Returns:
    --------
    report : dict
        Dictionary of model name -> R2 score on test data.
    """
    try:
        report = {}

        # Loop through each model
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            params = param.get(model_name, {})

            # -----------------------------
            # Handle CatBoost separately
            # -----------------------------
            if model_name == "CatBoosting Regressor":
                # Fit directly without GridSearchCV
                model.fit(X_train, y_train)
            # -----------------------------
            # Handle sklearn models with params
            # -----------------------------
            elif params:
                # Perform GridSearchCV for hyperparameter tuning
                gs = GridSearchCV(model, params, cv=3, n_jobs=-1, scoring='r2')
                gs.fit(X_train, y_train)

                # Set model to best parameters found
                model.set_params(**gs.best_params_)

                # Train the model on full training data
                model.fit(X_train, y_train)
            # -----------------------------
            # Sklearn models without params
            # -----------------------------
            else:
                model.fit(X_train, y_train)

            # -----------------------------
            # Make predictions
            # -----------------------------
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # -----------------------------
            # Calculate R2 scores
            # -----------------------------
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score in report
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

