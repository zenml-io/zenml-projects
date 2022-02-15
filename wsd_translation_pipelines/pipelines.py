from zenml.pipelines import pipeline
from models import translation_model, BARTMaterializer, translation_tokenizer, BARTTokenizerMaterializer
from models import wsd_model, wsd_tokenizer, BertTMaterializer, BertTokenizer
from languages import Languages
from predictor import translate, disambiguate_sense


@pipeline(enable_cache=False)
def translation_pipeline(
    sentence_dataset_path,
    importer,
    model_translate,
    tokenizer_translate,
    language,
    translator
):
    sentences = importer(sentence_dataset_path)
    tokenizer = tokenizer_translate()
    model = model_translate()
    translated_sentences = translator(model=model, tokenizer=tokenizer, sentences=sentences, language_code=language)


@pipeline(enable_cache=False)
def wsd_pipeline(
    sentence_dataset_path,
    word,
    importer,
    model_wsd,
    tokenizer_wsd,
    wsd,
    gloss_dict,
    offset_dict
):
    sentences = importer(sentence_dataset_path)
    tokenizer = tokenizer_wsd()
    model = model_wsd()
    senses = wsd(wsd_model=model, wsd_tokenizer=tokenizer, sense_gloss_dict=gloss_dict, offsets_dict=offset_dict,
                 sentences=sentences, word=word)


if __name__ == "__main__":
    from importer import sentence_importer, ListOfSentenceMaterializer

    pipeline_translate = translation_pipeline(
        sentence_dataset_path="/Users/edacicek/Downloads/comprehend_VERB.txt",
        importer=sentence_importer().with_return_materializers(ListOfSentenceMaterializer),
        model_translate=translation_model().with_return_materializers(BARTMaterializer),
        tokenizer_translate=translation_tokenizer().with_return_materializers(BARTTokenizerMaterializer),
        language=Languages.GERMAN.value,
        translator=translate()
    )

    pipeline_translate.run()
