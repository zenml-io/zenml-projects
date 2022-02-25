from zenml.steps import step
from model_config import TranslationModelConfig, WSDModelConfig

from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from GlossBERT.tokenization import BertTokenizer
from GlossBERT.modeling import BertForSequenceClassification


import pandas as pd
from nltk.corpus import wordnet as wn

import os
from typing import Type
import pickle

from zenml.artifacts import DataArtifact
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


class BARTMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (MBartForConditionalGeneration, )
    ASSOCIATED_ARTIFACT_TYPES = (DataArtifact, )

    def handle_input(self, data_type: Type[MBartForConditionalGeneration]) -> MBartForConditionalGeneration:
        """Read from artifact store"""
        super().handle_input(data_type)
        with fileio.open(os.path.join(self.artifact.uri, 'data.p'), 'rb') as f:
            obj = pickle.load(f)
        return obj

    def handle_return(self, my_obj: MBartForConditionalGeneration) -> None:
        """Write to artifact store"""
        super().handle_return(my_obj)
        with fileio.open(os.path.join(self.artifact.uri, 'data.p'), 'wb') as f:
            pickle.dump(my_obj, f)


class BARTTokenizerMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (MBart50Tokenizer, )
    ASSOCIATED_ARTIFACT_TYPES = (DataArtifact, )

    def handle_input(self, data_type: Type[MBart50Tokenizer]) -> MBart50Tokenizer:
        """Read from artifact store"""
        super().handle_input(data_type)
        with fileio.open(os.path.join(self.artifact.uri, 'tokenizer.p'), 'rb') as f:
            obj = pickle.load(f)
        return obj

    def handle_return(self, my_obj: MBart50Tokenizer) -> None:
        """Write to artifact store"""
        super().handle_return(my_obj)
        with fileio.open(os.path.join(self.artifact.uri, 'tokenizer.p'), 'wb') as f:
            pickle.dump(my_obj, f)


class BertTMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (BertForSequenceClassification, )
    ASSOCIATED_ARTIFACT_TYPES = (DataArtifact, )

    def handle_input(self, data_type: Type[BertForSequenceClassification]) -> BertForSequenceClassification:
        """Read from artifact store"""
        super().handle_input(data_type)
        with fileio.open(os.path.join(self.artifact.uri, 'data.p'), 'rb') as f:
            obj = pickle.load(f)
        return obj

    def handle_return(self, my_obj: BertForSequenceClassification) -> None:
        """Write to artifact store"""
        super().handle_return(my_obj)
        with fileio.open(os.path.join(self.artifact.uri, 'data.p'), 'wb') as f:
            pickle.dump(my_obj, f)


class BertTokenizerMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (BertTokenizer, )
    ASSOCIATED_ARTIFACT_TYPES = (DataArtifact, )

    def handle_input(self, data_type: Type[BertTokenizer]) -> BertTokenizer:
        """Read from artifact store"""
        super().handle_input(data_type)
        with fileio.open(os.path.join(self.artifact.uri, 'tokenizer.p'), 'rb') as f:
            obj = pickle.load(f)
        return obj

    def handle_return(self, my_obj: BertTokenizer) -> None:
        """Write to artifact store"""
        super().handle_return(my_obj)
        with fileio.open(os.path.join(self.artifact.uri, 'tokenizer.p'), 'wb') as f:
            pickle.dump(my_obj, f)


@step
def translation_model(config: TranslationModelConfig) -> MBartForConditionalGeneration:
    model = MBartForConditionalGeneration.from_pretrained(config.model_name)
    return model

@step
def translation_tokenizer(config: TranslationModelConfig) -> MBart50Tokenizer:
    tokenizer = MBart50Tokenizer.from_pretrained(config.tokenizer_name)
    return tokenizer

@step
def wsd_model(config: WSDModelConfig) -> BertForSequenceClassification:

    label_list = ["0", "1"]
    num_labels = len(label_list)
    model = BertForSequenceClassification.from_pretrained(config.model_file,
                                                          num_labels=num_labels)
    return model


@step
def wsd_tokenizer(config: WSDModelConfig) -> BertTokenizer:
    tokenizer = BertTokenizer.from_pretrained(config.model_file, do_lower_case=True)
    return tokenizer


def gloss_model(config: WSDModelConfig) -> (dict, dict):
    sense_gloss_dict = pd.read_csv(config.sense_index_file, sep='\t').values.tolist()
    sense_gloss_dict = dict(zip([s[0] for s in sense_gloss_dict], [(s[1], s[2], s[3], s[4]) for s in sense_gloss_dict]))

    syns = list(wn.all_synsets())
    offsets_list = [(s.offset(), s) for s in syns]
    offsets_dict = dict(offsets_list)
    return sense_gloss_dict, offsets_dict
