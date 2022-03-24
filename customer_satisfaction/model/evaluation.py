from ast import Pass
import numpy as np
import logging

# import sklearn regression evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class Evaluation:
    def __init__(self) -> None:
        pass 
    
    def mean_squared_error(self, y_true, y_pred):
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

    def r2_score(self, y_true, y_pred):
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

    def root_mean_squared_error(self, y_true, y_pred):
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


# class ModelEvaluater:
#     def __init__(self, x_test, y_test) -> None:
#         self.x_test = x_test
#         self.y_test = y_test
#         self.evaluator = Evaluation()

#     def evaluate_trained_models(
#         self, lg_model, rf_model, lgbm_model, xgb_model
#     ):
#         try:
#             logging.info(
#                 "Entered the evaluate_trained_models method of the ModelEvaluater class",
#             )
#             lg_pred = lg_model.predict(self.x_test)
#             rf_pred = rf_model.predict(self.x_test)
#             lgbm_pred = lgbm_model.predict(self.x_test)
#             xgb_pred = xgb_model.predict(self.x_test)
#             logging.info(
#                 "The mean absolute percentage error value for the CatBoost model is: "
#                 + str(
#                     self.evaluator.mean_absolute_percentage_error(
#                         self.y_test, lg_pred
#                     )
#                 ),
#             )
#             logging.info(
#                 "The mean absolute percentage error value for the Random Forest model is: "
#                 + str(
#                     self.evaluator.mean_absolute_percentage_error(
#                         self.y_test, rf_pred
#                     )
#                 ),
#             )
#             logging.info(
#                 "The mean absolute percentage error value for the Light GBM model is: "
#                 + str(
#                     self.evaluator.mean_absolute_percentage_error(
#                         self.y_test, lgbm_pred
#                     )
#                 ),
#             )
#             logging.info(
#                 "The mean absolute percentage error value for the XGBoost model is: "
#                 + str(
#                     self.evaluator.mean_absolute_percentage_error(
#                         self.y_test, xgb_pred
#                     )
#                 ),
#             )
#             logging.info(
#                 "MSE for catboost {}, random forest {}, light gbm {}, xgboost {}".format(
#                     self.evaluator.mean_squared_error(self.y_test, lg_pred),
#                     self.evaluator.mean_squared_error(self.y_test, rf_pred),
#                     self.evaluator.mean_squared_error(self.y_test, lgbm_pred),
#                     self.evaluator.mean_squared_error(self.y_test, xgb_pred),
#                 ),
#             )

#             logging.info(
#                 "RMSE for catboost {}, random forest {}, light gbm {}, xgboost {}".format(
#                     self.evaluator.root_mean_squared_error(
#                         self.y_test, lg_pred
#                     ),
#                     self.evaluator.root_mean_squared_error(
#                         self.y_test, rf_pred
#                     ),
#                     self.evaluator.root_mean_squared_error(
#                         self.y_test, lgbm_pred
#                     ),
#                     self.evaluator.root_mean_squared_error(
#                         self.y_test, xgb_pred
#                     ),
#                 ),
#             )

#             logging.info(
#                 "R2 for catboost {}, random forest {}, light gbm {}, xgboost {}".format(
#                     self.evaluator.r2_score(self.y_test, lg_pred),
#                     self.evaluator.r2_score(self.y_test, rf_pred),
#                     self.evaluator.r2_score(self.y_test, lgbm_pred),
#                     self.evaluator.r2_score(self.y_test, xgb_pred),
#                 ),
#             )

#             logging.info(
#                 "Exited the evaluate_trained_models method of the ModelEvaluater class",
#             )
#         except Exception as e:
#             logging.info(
#                 "Exception occured in evaluate_trained_models method of the ModelEvaluater class. Exception message:  "
#                 + str(e),
#             )
#             logging.info(
#                 "Exited the evaluate_trained_models method of the ModelEvaluater class",
#             )
#             raise Exception()
