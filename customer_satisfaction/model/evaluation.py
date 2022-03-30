from ast import Pass
import numpy as np
import logging

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class Evaluation: 
    '''
    Evaluation class which evaluates the model performance using the sklearn metrics 
    '''
    def __init__(self) -> None:
        pass 
    
    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: 
        '''
        Mean Squared Error (MSE) is the mean of the squared errors.
        Args: 
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float 
        '''
        try:
            logging.info(
                "Entered the mean_squared_error method of the Evaluation class",
            )
            mse = mean_squared_error(y_true, y_pred)
            logging.info("The mean squared error value is: " + str(mse),)

            return mse
        except Exception as e:
            logging.info(
                "Exception occured in mean_squared_error method of the Evaluation class. Exception message:  "
                + str(e),
            )
            logging.info(
                "Exited the mean_squared_error method of the Evaluation class",
            )
            raise Exception()

    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray): 
        ''' 
        R2 Score (R2) is a statistical measure of how close the observed values
        are to the predicted values. It is also known as the coefficient of
        determination.

        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        '''
        try:
            logging.info(
                "Entered the r2_score method of the Evaluation class",
            )
            r2 = r2_score(y_true, y_pred)
            logging.info("The r2 score value is: " + str(r2),)
            logging.info("Exited the r2_score method of the Evaluation class",)
            return r2
        except Exception as e:
            logging.info(
                "Exception occured in r2_score method of the Evaluation class. Exception message:  "
                + str(e),
            )
            logging.info("Exited the r2_score method of the Evaluation class",)
            raise Exception()

    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray):
         
        '''
        Root Mean Squared Error (RMSE) is the square root of the mean of the
        squared errors.

        Args:
            y_true: np.ndarray  
            y_pred: np.ndarray 
        Return: 
            rmse: float  
        '''
        try:
            logging.info(
                "Entered the root_mean_squared_error method of the Evaluation class",
            )
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("The root mean squared error value is: " + str(rmse),)
            return rmse
        except Exception as e:
            logging.info(
                "Exception occured in root_mean_squared_error method of the Evaluation class. Exception message:  "
                + str(e),
            )
            logging.info(
                "Exited the root_mean_squared_error method of the Evaluation class",
            )
            raise Exception()

