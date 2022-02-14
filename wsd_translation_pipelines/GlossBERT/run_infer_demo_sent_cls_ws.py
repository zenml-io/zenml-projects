import pandas as pd
import argparse
from GlossBERT.tokenization import BertTokenizer
from GlossBERT.modeling import BertForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
import sys
import logging
from nltk.corpus import wordnet as wn

import config

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


def construct_context_gloss_pairs(input, target_start_id, target_end_id, lemma):
    """
    construct context gloss pairs like sent_cls_ws
    :param input: str, a sentence
    :param target_start_id: int
    :param target_end_id: int
    :param lemma: lemma of the target word
    :return: candidate lists
    """
    sent = input.split(" ")
    assert 0 <= target_start_id and target_start_id < target_end_id  and target_end_id <= len(sent)
    target = " ".join(sent[target_start_id:target_end_id])
    if len(sent) > target_end_id:
        sent = sent[:target_start_id] + ['"'] + sent[target_start_id:target_end_id] + ['"'] + sent[target_end_id:]
    else:
        sent = sent[:target_start_id] + ['"'] + sent[target_start_id:target_end_id] + ['"']

    sent = " ".join(sent)
    lemma = lemma


    sense_data = pd.read_csv(config.gloss_sense_index,sep="\t",header=None).values
    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0],sense_data[i][-1]))
        except:
            d[s[:pos + 2]]=[(sense_data[i][0], sense_data[i][-1])]

    # print(len(d))
    # print(len(d["happy%3"]))
    # print(d["happy%3"])

    candidate = []
    for category in ["%1", "%2", "%3", "%4", "%5"]:
        query = lemma + category
        try:
            sents = d[query]
            for sense_key, gloss in sents:
                candidate.append((sent, f"{target} : {gloss}", target, lemma, sense_key, gloss))
        except:
            pass
    assert len(candidate) != 0, f'there is no candidate sense of "{lemma}" in WordNet, please check'
    # print(f'there are {len(candidate)} candidate senses of "{lemma}"')


    return candidate


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_to_features(candidate, tokenizer, max_seq_length=512):

    candidate_results = []
    features = []
    for item in candidate:
        text_a = item[0] # sentence
        text_b = item[1] # gloss
        candidate_results.append((item[-2], item[-1])) # (sense_key, gloss)


        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))


    return features, candidate_results



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def load_gloss_bert_model():
    label_list = ["0", "1"]
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(config.gloss_bert_model, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(config.gloss_bert_model,
                                                          num_labels=num_labels)
    sense_gloss_dict = pd.read_csv(config.gloss_sense_index, sep='\t').values.tolist()
    sense_gloss_dict = dict(zip([s[0] for s in sense_gloss_dict], [(s[1], s[2], s[3], s[4]) for s in sense_gloss_dict]))

    syns = list(wn.all_synsets())
    offsets_list = [(s.offset(), s) for s in syns]
    offsets_dict = dict(offsets_list)

    return model, tokenizer, sense_gloss_dict, offsets_dict


def infer(sentence, tar_word_lempos, target_word_index, model, tokenizer, sense_gloss_dict, offsets_dict):

    lemma = tar_word_lempos
    target_start_id = target_word_index
    target_end_id = target_word_index + 1

    # print(f"input: {sentence}\nlemma: {lemma}")
    examples = construct_context_gloss_pairs(sentence, target_start_id, target_end_id, lemma)
    eval_features, candidate_results = convert_to_features(examples, tokenizer)
    input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)


    model.eval()
    input_ids = input_ids
    input_mask = input_mask
    segment_ids = segment_ids
    with torch.no_grad():
        logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
    logits_ = F.softmax(logits, dim=-1)
    logits_ = logits_.detach().cpu().numpy()
    output = np.argmax(logits_, axis=0)[1]
    # print(f"results:\nsense_key: {candidate_results[output][0]}\ngloss: {candidate_results[output][1]}")

    sense_key = candidate_results[output][0]
    gloss = candidate_results[output][1]

    # Retrieve synset based on the offset.
    try:
        # sense = wn.synset_from_sense_key(sense_key)
        sense_fields = sense_gloss_dict[sense_key]
        synset_sense = offsets_dict[sense_fields[0]]
        synset_def = synset_sense.definition()
        if gloss == synset_def:
            sense = synset_sense
    except:
        sense = None

    return sense, gloss


if  __name__ == "__main__":
    from collections import OrderedDict
    import pickle
    import config

    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default=config.gloss_bert_files, type=str) #"bert-base-uncased", type=str) # default=r"H:\Elia\code\elia_nlp\resources\uncased_L-12_H-768_A-12", type=str)
    parser.add_argument("--no_cuda", default=True, action='store_true', help="Whether not to use CUDA when available")

    args = parser.parse_args()

    # demo input
    input = "I did n\'t want to part with them even when I went through an extremely difficult financial crisis."
    target_start_id = 5
    target_end_id = 6
    lemma = "part"

    model, tokenizer, device, sense_gloss_dict, offsets_dict = load_gloss_bert_model()
    sense, gloss = infer(input, lemma, target_start_id, model, tokenizer, device, sense_gloss_dict, offsets_dict)
    print(sense, gloss)
