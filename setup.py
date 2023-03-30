"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile


def download_url(url, output_path, show_progress=True):
    # this function downloads a file from a given URL to a given output path, with the option of displaying a progress bar during the download.
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    # This function takes a URL as input and returns a string representing the full path to a local file with the same name as the last component of the URL.
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    # This function downloads files and resources needed for the BiDAF model to run, including GloVe word vectors and the Spacy language model for English
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    # Finally, the function downloads the Spacy language model for English using the run() function from the 
    # subprocess module to run the command python -m spacy download en (which downloads and installs the 'en' model for Spacy).
    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])

def word_tokenize(sent):
    # word_tokenize(sent) is a function that takes a sentence (as a string) as input and returns a list of the individual words (as strings) in that sentence.
    doc = nlp(sent)
    temp = [token.text for token in doc]
    # print(f'---> word_tokenize: {temp[:10]}')
    return temp


def convert_idx(text, tokens):
    # convert_idx(text, tokens) is a function that takes two arguments: text, a string representing a piece of text, and tokens, a list of strings representing the individual tokens in that text. 
    # The function returns a list of pairs of integers, where each pair represents the starting and ending character positions of a token in the original text.
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    '''The process_file function takes as input the filename, the data type (train, dev, or test), and two counters to keep track of the words and characters in the dataset. 
    It returns two dictionaries, examples and eval_examples, which contain information about the preprocessed data.

    The function first opens the input file and loads the JSON data into a variable called source. It then iterates over each article and its corresponding paragraphs in the data. 
    For each paragraph, it preprocesses the context by tokenizing the text into words and characters using the word_tokenize function and the list function, respectively.
    It also generates a list of spans for each word in the context using the convert_idx function.

    For each question-answer pair in the paragraph, it preprocesses the question in the same way as the context. 
    It also generates a list of answers and their corresponding spans in the context. 
    It then creates an example dictionary containing the tokenized context and question, the character-level representations of the tokens, and the answer spans. 
    It adds this example to the examples list and creates a corresponding entry in the eval_examples dictionary.

    Finally, the function prints the total number of questions preprocessed and returns the examples and eval_examples dictionaries.
    print(f"Pre-processing {data_type} examples...")'''
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r", encoding='utf-8') as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        # Here is where we store the span of the answer present in context using the word index as y1(start) and y2(end)
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens,
                               "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars,
                               "y1s": y1s,
                               "y2s": y2s,
                               "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {"context": context,
                                                 "question": ques,
                                                 "spans": spans,
                                                 "answers": answer_texts,
                                                 "uuid": qa["id"]}
        print(f"{len(examples)} questions in total")
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    '''This function preprocesses the embedding vectors for a given data type, filtering out elements with a frequency below the specified limit,
     and returns an embedding matrix and a dictionary mapping tokens to their corresponding indices in the matrix. If an embedding file is provided, 
     it reads in the file and only includes embeddings for tokens that are present in the filtered elements list. If no embedding file is provided, 
     it initializes random embeddings for each token in the filtered elements list. The function also adds special tokens for NULL and out-of-vocabulary (OOV) tokens, 
     and assigns them indices 0 and 1 in the embedding matrix, respectively.'''

    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding {data_type} embedding vector")
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print(f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    # an index-to-embedding dictionary (idx2emb_dict) is generated using the token2idx_dict dictionary to map the index of each token to its corresponding embedding vector
    #  the emb_mat list is created by iterating over the indices in idx2emb_dict and appending the corresponding embedding vector to the list. 
    # This creates a matrix where each row corresponds to the embedding vector for a particular token.
    return emb_mat, token2idx_dict


def convert_to_features(args, data, word2idx_dict, char2idx_dict, is_test):
    '''This function converts a given context and question into token and character level indices using 
    pre-defined dictionaries of word and character mappings. It also performs filtering to ensure that the 
    length of the context and questions do not exceed the specified limits. The resulting indices are returned in the form of numpy arrays'''
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    char_limit = args.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def is_answerable(example):
    '''This function takes an example and checks if it is answerable or not. 
    An example is considered answerable if it has both start and end indices 
    of the answer span in its 'y1s' and 'y2s' keys, respectively.'''
    return len(example['y2s']) > 0 and len(example['y1s']) > 0


def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    '''The build_features function takes in several arguments including args, examples, data_type, out_file, word2idx_dict, char2idx_dict, and is_test=False 
    as a default parameter. It first sets some parameters such as para_limit, ques_limit, ans_limit, and char_limit depending on whether is_test is True or False. 
    It then defines a helper function called drop_example that returns True if the example should be dropped, and False otherwise. An example is dropped if it exceeds 
    the para_limit or ques_limit, or if it is answerable and the difference between the end and start indices of the answer exceeds ans_limit.

    The function then iterates through each example, checks if it should be dropped using drop_example, and if not, converts it to indices. 
    It uses word2idx_dict and char2idx_dict to convert each token and character to its corresponding index. It then creates four arrays to hold the 
    indices of the context, context characters, question, and question characters, and populates them with the converted indices. 
    It also populates two arrays, y1s and y2s, with the start and end indices of the answer if the example is answerable, and -1 otherwise. 
    Finally, it saves all the arrays to a .npz file specified by out_file and returns a dictionary containing metadata about the conversion process, 
    including the total number of examples converted.'''

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit or \
                   (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    np.savez(out_file,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             y1s=np.array(y1s),
             y2s=np.array(y2s),
             ids=np.array(ids))
    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    # The save function is used to save a Python object as a JSON file with the given filename.
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args):
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(args.train_file, "train", word_counter, char_counter)
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=None, vec_size=args.char_dim)

    # Process dev and test sets
    dev_examples, dev_eval = process_file(args.dev_file, "dev", word_counter, char_counter)
    build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)
    if args.include_test_examples:
        test_examples, test_eval = process_file(args.test_file, "test", word_counter, char_counter)
        save(args.test_eval_file, test_eval, message="test eval")
        test_meta = build_features(args, test_examples, "test",
                                   args.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

    save(args.word_emb_file, word_emb_mat, message="word embedding")
    save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.word2idx_file, word2idx_dict, message="word dictionary")
    save(args.char2idx_file, char2idx_dict, message="char dictionary")
    save(args.dev_meta_file, dev_meta, message="dev meta")


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()
    print(f'Printing Arguments to setup.py: {args_}')

    # Download resources
    download(args_)

    # Import spacy language model
    # spacy.blank("en") creates a blank spaCy Language object with the specified language model. In this case, the language model is "en", which is the code for the English language.
    nlp = spacy.blank("en")

    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    glove_ext = f'.txt' if glove_dir.endswith('d') else f'.{args_.glove_dim}d.txt'
    args_.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    pre_process(args_)
