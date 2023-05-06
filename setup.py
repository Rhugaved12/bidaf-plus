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
# import en_core_web_md
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile

import collections
from functools import partial
import msgpack


def download_url(url, output_path, show_progress=True):
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
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
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

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])


def word_tokenize(sent, type):
    doc = nlp(sent)

    if type is "context":
        context_tokens = [token.text for token in doc]
        context_tags = [w.tag_ for w in doc]
        context_ents = [w.ent_type_ for w in doc]
        # context_ents = [w.ent_type_ if w.ent_type_ !=
        #                 "" else "NULL" for w in doc]

        # print("doc: ", doc)
        # term frequency in document
        context_tokens_lower = [w.lower() for w in context_tokens]
        counter_ = collections.Counter(context_tokens_lower)
        total = len(context_tokens_lower)
        context_tf = [counter_[w] / total for w in context_tokens_lower]
        # print("context tokens: ", context_tokens)
        # print("TAGS: ", context_tags)
        # print("ENTS-->: ", context_ents)
        # print("TF:: ", context_tf)
        # counter_tag = collections.Counter(w for row in full for w in row[3])
        # vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True)
        # counter_ent = collections.Counter(w for row in full for w in row[4])
        # vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
        # w2id = {w: i for i, w in enumerate(vocab)}
        # tag2id = {w: i for i, w in enumerate(vocab_tag)}
        # ent2id = {w: i for i, w in enumerate(vocab_ent)}
        # log.info('Vocabulary size: {}'.format(len(vocab)))
        # log.info('Found {} POS tags.'.format(len(vocab_tag)))
        # log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))

    # to_id_ = partial(to_id, w2id=w2id, tag2id=tag2id, ent2id=ent2id)
    # train = list(map(to_id_, train))

        return (context_tokens, context_tf, context_tags, context_ents)
    return [token.text for token in doc]


def convert_idx(text, tokens):
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
    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                # returns (context_tokens, context_tf, context_tags, context_ents)
                context_features = word_tokenize(context, 'context')
                context_tokens = context_features[0]
                context_features = context_features[1:]
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
                    ques_tokens = word_tokenize(ques, 'question')
                    # Figuring out exact match here
                    context_tokens_lower = [w.lower() for w in context_tokens]
                    ques_tokens_lower = [w.lower() for w in ques_tokens]
                    ques_tokens_lower_set = set(ques_tokens_lower)
                    match_lower = [1.0 if w in ques_tokens_lower_set else 0.0 for w in context_tokens_lower]
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
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_features": context_features,  # context_features: [context_tf, context_tags, context_ents]
                               "em_lower": match_lower,         #exact match for each questoin-context pair
                               "context_tokens": context_tokens,
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
        print(
            f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                      token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0

# Each row_dict in examples: ((tag_ids, ent_ids, context_tf) + em_lower + (context_tokens, context_chars, question_token, question_chars, y1s, y2s, id))
def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
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
                len(ex["ques_tokens"]) > ques_limit or \
                (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print(f"Converting {data_type} examples to indices...")
# ((tag_ids, ent_ids, context_tf) + em_lower + (context_tokens, context_chars, question_token, question_chars, y1s, y2s, id))
    total = 0
    total_ = 0
    meta = {}
    tag_idxs = []
    ent_idxs = []
    context_tfs = []
    em_lowers = []
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    print("Build features length example: ", len(examples))
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
        tag_idx = np.zeros([para_limit], dtype=np.int32)
        ent_idx = np.zeros([para_limit], dtype=np.int32)
        context_tf = np.zeros([para_limit], dtype=np.float32)
        em_lower = np.zeros([para_limit], dtype=np.int32)

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)
# ((tag_ids, ent_ids, context_tf) + em_lower + (context_tokens, context_chars, question_token, question_chars, y1s, y2s, id))
        # print(f"{n}) Build feature examples: {example.keys()}")
        for i, token in enumerate(example["tag_ids"]):
            tag_idx[i] = token
        tag_idxs.append(tag_idx)

        for i, token in enumerate(example["ent_ids"]):
            ent_idx[i] =token
        ent_idxs.append(ent_idx)

        for i, token in enumerate(example["context_tf"]):
            context_tf[i] = token
        context_tfs.append(context_tf)

        for i, token in enumerate(example["em_lower"]):
            em_lower[i] =token
        em_lowers.append(em_lower)

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
    # print("CONTEXT TFS: ", context_tfs)
# ((tag_ids, ent_ids, context_tf) + em_lower + (context_tokens, context_chars, question_token, question_chars, y1s, y2s, id))
    np.savez(out_file,
            tag_idxs = np.array(tag_idxs),
            ent_idxs = np.array(ent_idxs),
            context_tfs = np.array(context_tfs),
            em_lowers = np.array(em_lowers),
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
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args):
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(
        args.train_file, "train", word_counter, char_counter)
    '''example = {"context_features": context_features,   #context_features: (context_tf, context_tags, context_ents)
                "em_lower": match_lower,         #exact match for each questoin-context pair
                "context_tokens": context_tokens,
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
    '''
    print("train examples 1: ", train_examples[0].keys())
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=None, vec_size=args.char_dim)

    # counter_tag = collections.Counter(
    #     w for row in train_examples for w in row['context_features'][1])
    # vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True)
    # counter_ent = collections.Counter(
    #     w for row in train_examples for w in row['context_features'][2])
    # vocab_ent = sorted(counter_tag, key=counter_ent.get, reverse=True)
    # tag2idx = {w: i for i, w in enumerate(vocab_tag)}
    # ent2idx = {w: i for i, w in enumerate(vocab_ent)}
    # print('Found {} POS tags.'.format(len(vocab_tag)))
    # print('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
    #
    # to_id_ = partial(to_id, tag2id=tag2idx, ent2id=ent2idx)
    # train_examples = list(map(to_id_, train_examples))

    # Process dev and test sets
    dev_examples, dev_eval = process_file(
        args.dev_file, "dev", word_counter, char_counter)

    # CHECK-> because in the reference code, we're concatenating train and dev
    # Since both of em are list of dicts u can "+" to append them
    full_examples = train_examples+dev_examples

    counter_tag = collections.Counter(
        w for row in full_examples for w in row['context_features'][1])
    vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True)
    counter_ent = collections.Counter(
        w for row in full_examples for w in row['context_features'][2])
    vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
    # vocab_ent = counter_ent
    tag2idx = {w: i for i, w in enumerate(vocab_tag)}
    ent2idx = {w: i for i, w in enumerate(vocab_ent)}
    print('Found {} POS tags.'.format(len(vocab_tag)))
    # 373 - 382 is performed on both train and dev content i.e., the indexing is done for train+dev features
    print('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))

    to_id_ = partial(to_id, tag2id=tag2idx, ent2id=ent2idx)
    # full_examples = list(map(to_id_, full_examples)) #corresponding indexes
    train_examples = list(map(to_id_, train_examples))
    dev_examples = list(map(to_id_, dev_examples))
    # ((tag_ids, ent_ids, context_tf) + em_lower + (context_tokens, context_chars, question_token, question_chars, y1s, y2s, id))
    print("TRain examples: ", train_examples[0].keys())

    # change build_features as there is change in structure of train_examples
    build_features(args, train_examples, "train",
                   args.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(
        args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)
    if args.include_test_examples:
        test_examples, test_eval = process_file(
            args.test_file, "test", word_counter, char_counter)
        save(args.test_eval_file, test_eval, message="test eval")
        test_meta = build_features(args, test_examples, "test",
                                   args.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

    # Saving vocab of tags and ents because we need them while training the model
    meta = {
        'vocab_tag': vocab_tag,
        'vocab_ent': vocab_ent,
    }
    with open('data/meta.msgpack', 'wb') as f:
        msgpack.dump(meta, f)

    save(args.word_emb_file, word_emb_mat, message="word embedding")
    save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.word2idx_file, word2idx_dict, message="word dictionary")
    save(args.char2idx_file, char2idx_dict, message="char dictionary")
    save(args.dev_meta_file, dev_meta, message="dev meta")


'''example = {"context_features": context_features,   #context_features: (context_tf, context_tags, context_ents)
                "em_lower": match_lower,         #exact match for each questoin-context pair
                "context_tokens": context_tokens,
                "context_chars": context_chars,
                "ques_tokens": ques_tokens,
                "ques_chars": ques_chars,
                "y1s": y1s,
                "y2s": y2s,
                "id": total}
    examples.append(example)
    '''


def to_id(row, tag2id, ent2id):
    # context_features = row[0]
    # context_features
    context_tf, context_tags, context_ents = row['context_features']
    row.pop('context_features', None)
    # question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    # context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    row['tag_ids'] = [tag2id[w] for w in context_tags]
    row['ent_ids'] = [ent2id[w] for w in context_ents]
    row['context_tf'] = context_tf
    # print("TFS: ", context_tf)
# ((tag_ids, ent_ids, context_tf) + em_lower + (context_tokens, context_chars, question_token, question_chars, y1s, y2s, id))
    return row


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    download(args_)

    # Import spacy language model
    nlp = spacy.load('en_core_web_sm')
    # nlp= en_core_web_md.load()
    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    glove_ext = f'.txt' if glove_dir.endswith(
        'd') else f'.{args_.glove_dim}d.txt'
    args_.glove_file = os.path.join(
        glove_dir, os.path.basename(glove_dir) + glove_ext)
    pre_process(args_)
