import simplemma
from languages import Languages, selected_languages_codes

from typing import List
from string import punctuation
punc_list = list(punctuation)


class Lemmatizer:
    def __init__(self, lang: str):
        self.langdata = simplemma.load_data(lang)

    def lemmatize(self, token):
        return simplemma.lemmatize(token, self.langdata)


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
            if len(token) < 1:
              continue
            lemmatized_tokens.append(self.language_models[lang].lemmatize(token.lower()))
        return lemmatized_tokens


def get_word_token_and_index(sentence: str, target_word: str):
    tokens = sentence.split()
    for i, token in enumerate(tokens):
        if target_word in token.lower():
            return i

    return -1
