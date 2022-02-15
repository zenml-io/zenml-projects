from zenml.steps import BaseStepConfig


class TranslationModelConfig(BaseStepConfig):
    """Translation Model Config"""

    model_name: str = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer_name: str = "facebook/mbart-large-50-many-to-many-mmt"


class WSDModelConfig(BaseStepConfig):
    """WSD Model Config"""

    model_name: str = "gloss-BERT"
    sense_index_file: str = "/Users/edacicek/Desktop/wsd_translation_pipelines/GlossBERT/wordnet/index.sense.gloss"
    model_file: str = "/Users/edacicek/Downloads/multi_cased_L-12_H-768_A-12"


class DatasetConfig(BaseStepConfig):
    """Sentence Dataset Config"""

    file_path: str = "/Users/edacicek/Downloads/comprehend.txt"
