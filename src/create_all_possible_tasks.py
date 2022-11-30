#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os
import logging
import pathlib
import numpy as np
from collections import defaultdict
import pandas as pd
from sys import stdout
from pathlib import Path
from itertools import product

from conllu import load_conllu_file, is_ok, find_conllu_files, read_sentences


class SampledDataset:
    def __init__(self, source_dataset):
        self.source_dataset = source_dataset
        self.vocab = set()
        self.focus_vocab = set()
        self.sentence_ids = set()

    def __len__(self):
        return len(self.sentence_ids)

    def __iter__(self):
        for sid in self.sentence_ids:
            yield self.source_dataset[sid]

    def add(self, sentence_id):
        self.sentence_ids.add(sentence_id)
        sentence = self.source_dataset[sentence_id]
        self.vocab |= set(t.form for t in sentence.sentence)
        self.focus_vocab.add(sentence.focus_word)

    def has_overlap(self, sentence):
        tokens = set(t.form for t in sentence.sentence)
        focus = sentence.focus_word
        return focus in self.vocab or len(tokens & self.focus_vocab) > 0

    def get_sentence(self, sentence_id):
        return self.source_dataset[sentence_id]

    def get_all_samples(self, max_class_ratio=None):
        if max_class_ratio is not None:
            return self.get_balanced_random_samples(
                len(self.sentence_ids), max_class_ratio
            )
        return [self.source_dataset[sid] for sid in self.sentence_ids]

    def get_random_samples(self, size):
        sampled_ids = np.random.choice(
            list(self.sentence_ids), size=size, replace=False
        )
        return [self.source_dataset[sid] for sid in sampled_ids]

    def get_balanced_random_samples(self, size, max_class_ratio):
        class_fr = defaultdict(set)
        for sid in self.sentence_ids:
            sentence = self.get_sentence(sid)
            class_fr[sentence.value].add(sid)
        samples = set()
        classes = list(class_fr.keys())
        sample_cl_cnt = defaultdict(int)
        stop = False
        while stop is False:
            for cl in classes:
                if class_fr[cl]:
                    s = np.random.choice(list(class_fr[cl]))
                    samples.add(s)
                    class_fr[cl].remove(s)
                    sample_cl_cnt[cl] += 1
                else:
                    # check if class ratio is reached
                    if (
                        float(max(sample_cl_cnt.values())) / min(sample_cl_cnt.values())
                        >= max_class_ratio
                    ):
                        stop = True
            if len(samples) >= size:
                stop = True
        return [self.source_dataset[sid] for sid in samples]


class SampleSentence:
    def __init__(self, sentence, focus_idx, name, value=None):
        self.sentence = sentence
        self.focus_idx = focus_idx
        self.focus_word = self.sentence[focus_idx].form
        self.name = name
        if value is None:
            self.value = sentence[focus_idx].get_tag_value(name)
        else:
            self.value = value


def parse_args():
    ud_root = os.path.join(os.environ["HOME"], "data", "ud-treebanks-v2.5")
    p = ArgumentParser()
    p.add_argument("--ud-dir", type=str, default=ud_root)
    p.add_argument("--tasks_file", default=None, type=str, help="TSV file with the list of tasks to generate")
    p.add_argument("-t", "--tags", type=str, nargs="*")
    p.add_argument("-l", "--languages", type=str, nargs="*")
    p.add_argument("-p", "--pos", type=str, nargs="*")
    p.add_argument(
        "--maxlen", type=int, default=100, help="Maximum number of tokens in sentence"
    )
    p.add_argument(
        "--minlen", type=int, default=0, help="Minimum number of tokens in sentence"
    )
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--dev-size", type=int, default=200)
    p.add_argument("--rare-filter", type=int, default=50)
    p.add_argument(
        "--max-class-ratio",
        type=float,
        default=100,
        help="Maximum ratio of samples in the biggest vs " "the smallest class",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-o", "--outdir", type=str)
    p.add_argument(
        "--stats-only",
        action="store_true",
        help="Compute label entropies without creating datasets",
    )
    p.add_argument("--include-pos", action="store_true", help="Add POS to each token")
    return p.parse_args()


def load_train_dev_test(dirs, maxlen, minlen):
    data = {"train": [], "dev": [], "test": []}
    for subdir in dirs:
        data_files = find_conllu_files(subdir.path)
        for fn in data_files:
            split = fn.split("-")[-1][: -len(".conllu")]
            for sentence in load_conllu_file(
                os.path.join(subdir.path, fn), skip_word_pieces=True
            ):
                if len(sentence) >= minlen and len(sentence) <= maxlen:
                    data[split].append(sentence)
    return data


def find_valid_ud_dirs(ud_root, languages):
    ud_dirs = defaultdict(set)
    for fn in os.scandir(ud_root):
        language = fn.name[3:].split("-")[0]
        if language == "Norwegian":
            if "Bokmaal" in fn.name:
                language = "Norwegian_Bokmal"
            else:
                language = "Norwegian_Nynorsk"
        if language not in languages:
            continue
        if is_ok(fn.path):
            ud_dirs[language].add(fn)
    return ud_dirs


def find_focus_words(data, name, pos=None, disambig="depth"):
    cat_sentences = {"train": [], "dev": [], "test": []}
    if pos:
        pos = set(pos.split(","))
    for split, sentences in data.items():
        for sid, sentence in enumerate(sentences):
            if disambig == "depth":
                tokens = list(sentence.get_tokens_with_tag(name))
                if pos:
                    tokens = [t for t in tokens if t.upos in pos]
                if tokens:
                    sentence.add_depth()
                    mintoken = min(tokens, key=lambda x: x[1].depth)
                    value = mintoken[1].get_tag_value(name)
                    cat_sentences[split].append(
                        SampleSentence(sentence, mintoken[0], name, value)
                    )
            elif disambig == "all":
                for ti, token in enumerate(sentence.tokens):
                    if pos and token.upos not in pos:
                        continue
                    if token.has_morph_tag(name):
                        value = token.get_tag_value(name)
                        cat_sentences[split].append(
                            SampleSentence(sentence, ti, name, value)
                        )
    return cat_sentences


def filter_rare_tags(data, min_size):
    tag_freq = defaultdict(int)
    for sentences in data.values():
        for sentence in sentences:
            tag_freq[sentence.value] += 1
    rare = set(k for k, v in tag_freq.items() if v < min_size)
    if rare:
        logging.warning("Filtering rare tags: {}".format(", ".join(rare)))
        logging.warning(
            "Remaining tags are: {}".format(", ".join(set(tag_freq.keys()) - rare))
        )
        filtered_data = {"train": [], "dev": [], "test": []}
        for split, sentences in data.items():
            for sentence in sentences:
                if sentence.value not in rare:
                    filtered_data[split].append(sentence)
        return filtered_data, set(tag_freq.keys() - rare)
    return data, set(tag_freq.keys())


def get_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        vocab |= set(t.form for t in sentence.sentence)
    return vocab


def create_disjunct_datasets(data, max_train=0, max_dev=0):
    # uniq forms
    sampled_dataset = {}
    vocab = {}
    forms = {}
    for split in data.keys():
        sampled_dataset[split] = SampledDataset(data[split])
        vocab[split] = get_vocab(data[split])
        forms[split] = set(s.focus_word for s in data[split])

    uniq = {}
    for split in data.keys():
        uniq[split] = forms[split]
        for other in data.keys():
            if other == split:
                continue
            uniq[split] -= vocab[other]

        for si, s in enumerate(data[split]):
            if s.focus_word in uniq[split]:
                sampled_dataset[split].add(si)

    remaining = gather_remaining_sentences(sampled_dataset)

    max_size = {
        "train": max_train if max_train > 0 else 100000,
        "dev": max_dev if max_dev > 0 else 100000,
        "test": max_dev if max_dev > 0 else 100000,
    }

    to_sample = ["train"] * 4 + ["dev", "test"] + ["train"] * 4
    ti = 0
    while all(len(v) > 0 for v in remaining.values()):

        if all(len(d) >= max_size[split] for split, d in sampled_dataset.items()):
            break

        split = to_sample[ti]
        ti = (ti + 1) % len(to_sample)
        sid = np.random.choice(list(remaining[split]))
        sampled_dataset[split].add(sid)
        remaining = gather_remaining_sentences(sampled_dataset)
    return sampled_dataset


def gather_remaining_sentences(dataset):
    remaining = {"train": set(), "dev": set(), "test": set()}
    for split, data in dataset.items():
        for sid, sentence in enumerate(data.source_dataset):
            if sid in data.sentence_ids:
                continue
            overlap = False
            for other in dataset.keys():
                if other == split:
                    continue
                if dataset[other].has_overlap(sentence):
                    overlap = True
            if overlap is False:
                remaining[split].add(sid)
    return remaining


def check_overlap(dataset):
    for split, data in dataset.items():
        for sent in data:
            for other in dataset.keys():
                if split == other:
                    continue
                assert dataset[other].has_overlap(sent) is False


def sample_sentence_pairs(dataset, train_size, dev_size):
    sizes = {"train": train_size, "dev": dev_size, "test": dev_size}
    pairs = {}
    for split, data in dataset.items():
        pairs[split] = sample_pairs_from_single_dataset(data, sizes[split])
    return pairs


def sample_pairs_from_single_dataset(data, size):
    tag_mapping = defaultdict(list)
    for sid in data.sentence_ids:
        sentence = data.get_sentence(sid)
        tag_mapping[sentence.value].append(sid)

    pairs = set()
    tags = list(tag_mapping.keys())

    S = len(data)
    if S * (S - 1) / 2 < size:
        logging.warning(
            "Not enough samples, reducing size {}-->{}".format(size, S * (S - 1) / 2)
        )
        size = int(S * (S - 1) / 2)

    class_no = len(tag_mapping)
    class_min = size // class_no
    mintag, minval = min(tag_mapping.items(), key=lambda x: len(x[1]))
    minval = len(minval)
    skip_class = set()
    while minval * (minval - 1) / 2 <= class_min:
        logging.info(
            "Class {} is small ({}), using all possible combinations".format(
                mintag, minval
            )
        )
        skip_class.add(mintag)
        for l in tag_mapping[mintag]:
            for r in tag_mapping[mintag]:
                if l < r:
                    pairs.add((l, r, 1))
        class_no -= 1
        class_min = size // class_no
        remaining = set(tag_mapping.keys()) - skip_class
        if not remaining:
            break
        mintag = min(remaining, key=lambda x: len(tag_mapping[x]))
        minval = len(tag_mapping[mintag])

    for _ in range(len(pairs)):
        # sample the same numbr of negative pairs
        tag1, tag2 = np.sort(np.random.choice(tags, size=2, replace=False))
        l = np.random.choice(tag_mapping[tag1])
        r = np.random.choice(tag_mapping[tag2])
        while (l, r, 0) in pairs:
            l = np.random.choice(tag_mapping[tag1])
            r = np.random.choice(tag_mapping[tag2])
        pairs.add((l, r, 0))

    while len(pairs) < size:
        for tag, sids in tag_mapping.items():
            if tag in skip_class:
                continue
            # sample positive pair
            l, r = np.sort(np.random.choice(sids, size=2, replace=False))
            # while (l, r, 1) in pairs:
            # l, r = np.sort(np.random.choice(sids, size=2, replace=False))
            pairs.add((l, r, 1))
            # sample negative pair
            if len(tags) > 1:
                tag1, tag2 = np.sort(np.random.choice(tags, size=2, replace=False))
                l = np.random.choice(tag_mapping[tag1])
                r = np.random.choice(tag_mapping[tag2])
                # while (l, r, 0) in pairs:
                # l = np.random.choice(tag_mapping[tag1])
                # r = np.random.choice(tag_mapping[tag2])
                pairs.add((l, r, 0))

    # switch pairs with .5 prob
    switched = set()
    for l, r, label in pairs:
        if np.random.random() < 0.5:
            switched.add((r, l, label))
        else:
            switched.add((l, r, label))
    assert len(pairs) == len(switched)
    return switched


def sample_sentences(dataset, train_size, dev_size, max_class_ratio):
    sizes = {"train": train_size, "dev": dev_size, "test": dev_size}
    pairs = {}
    for split, data in dataset.items():
        if len(data) < sizes[split]:
            logging.warning(
                "Not enough samples in {}, discarding task".format(split, len(data))
            )
            return None
            pairs[split] = data.get_all_samples(max_class_ratio)
        else:
            pairs[split] = data.get_balanced_random_samples(
                sizes[split], max_class_ratio
            )
    return pairs


def save_dataset(sentences, outdir, include_pos=False):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    for split, this_split in sentences.items():
        fn = os.path.join(outdir, "{}.tsv".format(split))
        np.random.shuffle(this_split)
        with open(fn, "w") as f:
            for sentence in this_split:
                if include_pos:
                    f.write(
                        "{}\t{}\t{}\t{}\n".format(
                            " ".join(
                                "{}_{}".format(t.form, t.upos)
                                for t in sentence.sentence.tokens
                            ),
                            sentence.focus_word,
                            sentence.focus_idx,
                            sentence.value,
                        )
                    )
                else:
                    f.write(
                        "{}\t{}\t{}\t{}\n".format(
                            " ".join(t.form for t in sentence.sentence.tokens),
                            sentence.focus_word,
                            sentence.focus_idx,
                            sentence.value,
                        )
                    )


def load_tasks(args):
    if args.tasks_file is not None:
        tasks = pd.read_table(args.tasks_file, sep="\t")
    else:
        tasks = []
        tasks = pd.DataFrame(product(args.languages, args.pos, args.tags),
                     columns=["language", "pos", "tag"])
        tasks["task"] = tasks["tag"] + "_" + tasks["pos"]
    return tasks


def compute_label_entropy(dataset):
    token_label_cnt = defaultdict(lambda: defaultdict(int))
    label_types = set()
    for data in dataset.values():
        for sent in data:
            tok = sent.focus_word
            label = sent.value
            label_types.add(label)
            token_label_cnt[tok][label] += 1
    lsum = sum(len(d) for d in dataset.values())
    p_token = {t: sum(d.values()) / lsum for t, d in token_label_cnt.items()}
    entropies = []
    w_ent = []
    forms = []
    H = 0
    for token, dist in token_label_cnt.items():
        for label, cnt in dist.items():
            p = cnt / lsum
            H -= p * np.log2(p / p_token[token])
        E = get_entropy(dist)
        entropies.append(E)
        w_ent.append(E * sum(dist.values()))
        forms.append(len(dist))
    entropy = sum(entropies) / len(entropies)
    w_ent = sum(w_ent) / len(w_ent)
    avg_form = sum(forms) / len(forms)
    return {
        "entropy": entropy,
        "weighted_entropy": w_ent,
        "avg_form": avg_form,
        "H": H,
        "TTR": len(token_label_cnt) / lsum,
        "label_types": len(label_types),
    }


def get_entropy(dist):
    s = sum(dist.values())
    E = 0
    for v in dist.values():
        p = v / s
        E -= p * np.log2(p)
    return E


def main():
    args = parse_args()
    tasks = load_tasks(args)
    # gather language dirs, check if data is ok
    ud_dirs = find_valid_ud_dirs(args.ud_dir, tasks.language.unique())

    cnt = 0
    stats = []
    for language, dirs in sorted(ud_dirs.items()):
        logging.info(f"====== Generating {language} tasks. ======")
        # create language dataset
        full_data = load_train_dev_test(dirs, args.maxlen, args.minlen)
        for k, v in full_data.items():
            logging.info(f"{k} size before filtering: {len(v)}")

        for idx, row in tasks[tasks.language == language].iterrows():
            if args.seed is not None:
                np.random.seed(args.seed)
            tag = row["tag"]
            pos = row["pos"]
            task = row["task"]
            cnt += 1
            logging.info(
                f"{cnt} / {len(tasks)} Creating <{language},{pos},{task}> dataset"
            )
            outdir = Path(args.outdir) / task.lower() / language.lower()
            data = find_focus_words(full_data, tag, pos=pos, disambig="all")
            data, tags = filter_rare_tags(data, args.rare_filter)
            if any(len(d) < 100 for d in data.values()):
                logging.info(
                    f"Too few samples for <{language},{pos},{task}> - skipping."
                )
                continue
            if len(tags) < 2:
                logging.info(f"Only one class left, skipping <{language},{pos},{task}>")
            if args.stats_only:
                st = compute_label_entropy(data)
                d = {
                    "language": language,
                    "pos": pos,
                    "tag": tag,
                }
                d.update(st)
                stats.append(d)
                continue
            dataset = create_disjunct_datasets(
                data, max_train=5 * args.train_size, max_dev=5 * args.dev_size
            )
            for k, v in dataset.items():
                logging.info(f"{k} size after filtering: {len(v)}.")
            check_overlap(dataset)
            sentences = sample_sentences(
                dataset, args.train_size, args.dev_size, args.max_class_ratio
            )
            if sentences:
                save_dataset(
                    sentences, outdir, include_pos=args.include_pos
                )
    if args.stats_only:
        stats = pd.DataFrame(stats)
        stats.to_csv(stdout, sep="\t", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
