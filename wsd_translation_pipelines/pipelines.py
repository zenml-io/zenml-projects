from zenml.pipelines import pipeline
from models import translation_model, BARTMaterializer, translation_tokenizer, BARTTokenizerMaterializer
from models import wsd_model, wsd_tokenizer, BertTMaterializer, BertTokenizer
from languages import Languages, selected_languages_codes
from model_config import DatasetConfig, TranslationModelConfig
from predictor import translate


@pipeline(enable_cache=False)
def translation_pipeline(
    importer,
    model_translate,
    tokenizer_translate,
    translator,
):
    sentences = importer()
    tokenizer = tokenizer_translate()
    model = model_translate()
    translated_sentences = translator(model=model, tokenizer=tokenizer, sentences=sentences)



# @pipeline(enable_cache=False)
# def wsd_pipeline(
#     sentence_dataset_path,
#     word,
#     importer,
#     model_wsd,
#     tokenizer_wsd,
#     wsd,
#     gloss_dict,
#     offset_dict
# ):
#     sentences = importer(sentence_dataset_path)
#     tokenizer = tokenizer_wsd()
#     model = model_wsd()
#     senses = wsd(wsd_model=model, wsd_tokenizer=tokenizer, sense_gloss_dict=gloss_dict, offsets_dict=offset_dict,
#                  sentences=sentences, word=word)


if __name__ == "__main__":
    from importer import sentence_importer, ListOfSentenceMaterializer

    pipeline_translate = translation_pipeline(
        # importer=sentence_importer(DatasetConfig()),
        importer=sentence_importer(DatasetConfig()).with_return_materializers(ListOfSentenceMaterializer),
        model_translate=translation_model(TranslationModelConfig()).with_return_materializers(BARTMaterializer),
        tokenizer_translate=translation_tokenizer(TranslationModelConfig()).with_return_materializers(BARTTokenizerMaterializer),
        translator=translate()
    )

    pipeline_translate.run()
