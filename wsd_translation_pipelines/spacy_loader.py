import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import config


def custom_tokenizer(nlp: spacy.Language) -> Tokenizer:
    inf = list(nlp.Defaults.infixes)  # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")  # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)  # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])",
                           r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x]  # Remove - between letters rule
    infixes.append('(?:–|—|--|---|——|~)')
    infix_re = compile_infix_regex(infixes)
    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                     suffix_search=nlp.tokenizer.suffix_search,
                     infix_finditer=infix_re.finditer,
                     token_match=nlp.tokenizer.token_match,
                     rules=nlp.Defaults.tokenizer_exceptions)


def get_model(trained_pipeline_name: str) -> spacy.Language:
    """
    example call: get_model('en_core_web_trf')
    :param trained_pipeline_name: name of trained pipeline
    :return: spacy model based on trained pipeline.
    """
    nlp = spacy.load(trained_pipeline_name)
    if trained_pipeline_name == config.spacy_model_name:
        nlp.tokenizer = custom_tokenizer(nlp)
    return nlp


nlp = get_model('en_core_web_sm')
