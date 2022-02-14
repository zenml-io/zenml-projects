# fname_britamer_wordlist = '/home/boo/repos/senEx/resources/BritishAmericanSpelling_MappingWordlist.csv'
fname_britamer_wordlist = r'H:\Elia\code\elia_nlp\resources\BritishAmericanSpelling_MappingWordlist.csv'

# Synonyms parameters
wordnet_distance = 1

# WSD Lesk parameters
lesk_algorithm = 'gloss_bert' # Select one from ['custom_lesk', 'simple_lesk', 'adapted_lesk', 'cosine_lesk', 'gloss_bert']
gloss_bert_model = r'/Users/edacicek/Downloads/Sent_CLS_WS' # https://drive.google.com/file/d/1iq_h3zLTflraEU_7tVLnPcVQTeyGDNKE/view?usp=sharing
no_cuda = True
gloss_sense_index = r'/Users/edacicek/Desktop/wsd_translation_pipelines/GlossBERT/wordnet/index.sense.gloss' # https://github.com/HSLCY/GlossBERT
gloss_bert_files = r'/Users/edacicek/Downloads/multi_cased_L-12_H-768_A-12'

# Translation parameters
source_language = 'en'
target_language = 'de'
target_languages = []
spacy_model_name = 'en_core_web_trf'
de_spacy_model_name = 'de_dep_news_trf'

# Marian MT model parameters
huggingface_s3_base_url = r'https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP'
trans_filenames = ['pytorch_model.bin', 'config.json', 'source.spm', 'target.spm', 'tokenizer_config.json', 'vocab.json']
trans_model_path = 'data'

# Sim Align parameters
simalign_model = 'bert'
simalign_token_type = 'bpe'
simalign_matching_method = 'inter'
