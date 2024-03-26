import os
import tarfile
import tempfile
from typing import Type, Union

import joblib
from sagemaker import Predictor
from sklearn.base import ClassifierMixin
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.materializers.built_in_materializer import BuiltInMaterializer


class SagemakerMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (ClassifierMixin,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(
        self, data_type: Type[ClassifierMixin]
    ) -> Union[SGDClassifier, XGBClassifier]:
        """Read from artifact store."""
        fileio.copy(
            os.path.join(self.uri, "model.tar.gz"),
            os.path.join(tempfile.gettempdir(), "model.tar.gz"),
            overwrite=True,
        )
        est = None
        with tarfile.open(
            os.path.join(tempfile.gettempdir(), "model.tar.gz"), "r:gz"
        ) as tar:
            for member in tar.getmembers():
                tar.extract(member.name, tempfile.gettempdir())
                if member.name == "sklearn-model":
                    est = joblib.load(
                        os.path.join(tempfile.gettempdir(), "sklearn-model"),
                    )
                if member.name == "xgboost-model":
                    est = XGBClassifier()
                    est.load_model(os.path.join(tempfile.gettempdir(), "xgboost-model"))
                fileio.remove(os.path.join(tempfile.gettempdir(), member.name))
                if est:
                    break
        if est is None:
            raise RuntimeError("Failed to load estimator via SagemakerMaterializer...")
        return est

    def save(self, my_obj: ClassifierMixin) -> None:
        """Write to artifact store."""
        with tarfile.open(
            os.path.join(tempfile.gettempdir(), "model.tar.gz"), "w:gz"
        ) as tar:
            is_xgboost = isinstance(my_obj, XGBClassifier)
            file_name = ("xgboost" if is_xgboost else "sklearn") + "-model"
            tmp_ = os.path.join(tempfile.gettempdir(), file_name)
            if is_xgboost:
                # if model supports saving - use it over joblib
                my_obj.save_model(tmp_)
            else:
                joblib.dump(my_obj, tmp_)
            tar.add(tmp_, arcname=file_name)
            fileio.remove(tmp_)
        fileio.copy(
            os.path.join(tempfile.gettempdir(), "model.tar.gz"),
            os.path.join(self.uri, "model.tar.gz"),
            overwrite=True,
        )
        fileio.remove(os.path.join(tempfile.gettempdir(), "model.tar.gz"))


class SagemakerPredictorMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Predictor,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.SERVICE

    def load(self, data_type: Type[Predictor]) -> Predictor:
        """Read from artifact store."""
        return Predictor(endpoint_name=BuiltInMaterializer(self.uri).load(str))

    def save(self, my_obj: Predictor) -> None:
        """Write to artifact store."""
        BuiltInMaterializer(self.uri).save(my_obj.endpoint_name)
