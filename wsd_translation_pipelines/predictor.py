from zenml.steps import step

from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from GlossBERT.tokenization import BertTokenizer
from GlossBERT.modeling import BertForSequenceClassification
from GlossBERT.run_infer_demo_sent_cls_ws import infer

from typing import List, Dict, Any
import pandas as pd
from lemmatize import get_word_token_and_index


@step
def translate(model: MBartForConditionalGeneration, tokenizer: MBart50Tokenizer, sentences: pd.DataFrame) -> pd.DataFrame:

    tokenizer.src_lang = "en_XX"
    translated_sentences = list()
    lang_code = 'de_DE'

    for sentence in sentences['sentences'].tolist():
    # for sentence in sentences:
        encoded_ar = tokenizer(sentence, return_tensors="pt")
        encoded_ar["output_hidden_states"] = True

        outputs = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code])
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_sentences.append(translated[0])

    sentences[lang_code] = translated_sentences

    return sentences


# @step
# def disambiguate_sense(wsd_model: BertForSequenceClassification, wsd_tokenizer: BertTokenizer, sense_gloss_dict: Dict[str, Any], offsets_dict: Dict[str, Any], sentences: pd.DataFrame, word: str) -> pd.DataFrame:
#
#     senses = list()
#     for sentence in sentences:
#         word_index = get_word_token_and_index(sentence, word)
#
#         ss, gloss = infer(sentence, word, word_index, wsd_model, wsd_tokenizer, sense_gloss_dict, offsets_dict)
#         offset = str(ss.offset()) + str(ss.pos())
#         senses.append(offset)
#
#     sentences['senses'] = senses
#
#     return sentences
