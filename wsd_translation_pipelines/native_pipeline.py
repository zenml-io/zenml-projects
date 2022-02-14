from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from simalign import SentenceAligner
from GlossBERT.run_infer_demo_sent_cls_ws import infer, load_gloss_bert_model
from languages import selected_languages_codes, selected_languages, Languages
from lemmatize import Lemmatizer
import torch

from typing import List, Tuple
from string import punctuation
from collections import defaultdict

punc_list = list(punctuation)


def import_sentences(filepath):
    sentences = list()
    with open(filepath, 'r') as f:
        for line in f:
            sentences.append(line.rstrip())
    return sentences

class LanguageLemmatizer:
    def __init__(self):
        self.language_models = dict()
        for code in selected_languages_codes + [Languages.ENGLISH.value]:
            lemmatizer_code = code.split('_')[0]
            self.language_models[code] = Lemmatizer(lemmatizer_code)

    def lemmatize(self, lang: str, token: str) -> str:
        return self.language_models[lang].lemmatize(token.lower())

    def lemmatize_sentence(self, lang: str, sentence: str) -> List[str]:
        lemmatized_tokens = list()
        for token in sentence.split():
            if token in punctuation:
                lemmatized_tokens.append(token)
                continue
            if token[-1] in punc_list:
                token = token[:-1]
            if token[0] in punc_list:
                token = token[1:]
            lemmatized_tokens.append(self.language_models[lang].lemmatize(token.lower()))
        return lemmatized_tokens

def get_word_token_and_index(sentence: str, target_word: str, lemmatizer: LanguageLemmatizer):
    tokens = lemmatizer.lemmatize_sentence('en_XX', sentence)
    for i, token in enumerate(tokens):
        if token == target_word:
            return tokens, i

    return None, -1

def translate_endpoint(filepath: str, word: str) -> List[Tuple[str, str, str]]:
    sentences = import_sentences(filepath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelBART = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
    tokenizerBART = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
    lemmatizers = LanguageLemmatizer()
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai", device='cuda')

    wsd_model, wsd_tokenizer, device, sense_gloss_dict, offsets_dict = load_gloss_bert_model()
    wsd_model.to(device)

    wsd_translations = dict()

    for code in selected_languages_codes:
        wsd_translations[code] = defaultdict(dict)

    for sentence in sentences[0:10]:
        tokens, word_index = get_word_token_and_index(sentence, word, lemmatizers)

        ss, gloss = infer(sentence, word, word_index, wsd_model, wsd_tokenizer, device, sense_gloss_dict, offsets_dict)
        offset = str(ss.offset())+str(ss.pos())

        encoded_ar = tokenizerBART(sentence, return_tensors="pt").to(device)
        encoded_ar["output_hidden_states"] = True

        for code in selected_languages_codes:
            outputs = modelBART.generate(**encoded_ar,
                                         forced_bos_token_id=tokenizerBART.lang_code_to_id[code])
            translated = tokenizerBART.batch_decode(outputs, skip_special_tokens=True)
            translated = translated[0]

            print(translated)

            trg_text = translated.split()
            src_text = sentence.split()

            alignments = aligner.get_word_aligns(src_text, trg_text)
            alignments_mwmf = alignments['mwmf']
            for align in alignments_mwmf:
                if word.lower() in src_text[align[0]].lower():
                    translated_lemma = lemmatizers.lemmatize_sentence(code, trg_text[align[1]])[0]
                    if translated_lemma in wsd_translations[code][offset]:
                        wsd_translations[code][offset][translated_lemma] += 1
                    else:
                        wsd_translations[code][offset][translated_lemma] = 1

    from pprint import pprint
    pprint(wsd_translations)


translate_endpoint("/Users/edacicek/Desktop/comprehend_VERB.txt", "comprehend")
