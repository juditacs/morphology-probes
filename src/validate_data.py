#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import pandas as pd
import os
import logging
import shutil


def parse_args():
    p = ArgumentParser()
    p.add_argument('--train-size', default=2000, type=int,
                   help="Minimum train size")
    p.add_argument('--dev-size', default=200, type=int,
                   help="Minimum dev size")
    p.add_argument('--test-size', default=200, type=int,
                   help="Minimum test size")
    p.add_argument('--delete-small', action='store_true',
                   help="Delete tasks smaller than minimum size")
    p.add_argument('--copy-good', type=str, default=None)
    p.add_argument('data_dir', type=str)
    return p.parse_args()


class NotGoodEnough(ValueError):
    pass


def validate_directory(dirname, train_size, dev_size, test_size):
    columns = ['sentence', 'target', 'target_idx', 'label']
    sizes = {'train': train_size, 'dev': dev_size, 'test': test_size}
    labels = {}
    for split in ['train', 'dev', 'test']:
        fn = os.path.join(dirname, f'{split}.tsv')
        df = pd.read_table(fn, names=columns, na_filter=False, quoting=3)
        if len(df) < sizes[split]:
            raise NotGoodEnough(f'Not enough {split} samples ({len(df)}) {fn}')
        labels[split] = tuple(sorted(df.label.unique()))
        validate_dataframe(df, fn)
    if len(set(labels.values())) != 1:
        raise NotGoodEnough(f'Label set is different between splits: {labels} in {dirname}')



def validate_dataframe(df, fn):
    label_cnt = df.label.value_counts()
    if label_cnt.min() * 3 < label_cnt.max():
        raise NotGoodEnough(f"Smallest class * 3 < largest class {fn}")
    if len(label_cnt) < 2:
        raise NotGoodEnough(f"Only one class {fn}")
    try:
        df.apply(check_target, axis=1)
    except AssertionError:
        raise NotGoodEnough(f"Target idx incorrect {fn}")


def check_target(row):
    tgt = row.sentence.split(' ')[row.target_idx]
    # POS data removed for now. No need to remove POS.
    # tgt = tgt.rsplit("_", maxsplit=1)[0]
    assert tgt == row.target


def main():
    args = parse_args()
    good = 0
    all_ = 0
    for task_path in os.scandir(args.data_dir):
        task = task_path.name
        for lang_path in os.scandir(task_path.path):
            language = lang_path.name
            try:
                all_ += 1
                validate_directory(lang_path.path, train_size=args.train_size,
                                dev_size=args.dev_size, test_size=args.test_size)
            except NotGoodEnough as e:
                print(e)
            else:
                if args.copy_good:
                    task_dir = os.path.join(args.copy_good, task)
                    if not os.path.exists(task_dir):
                        os.makedirs(task_dir)
                    lang_dir = os.path.join(task_dir, language)
                    #if not os.path.exists(lang_dir):
                        #os.makedirs(lang_dir)
                    shutil.copytree(lang_path.path, lang_dir)
                good += 1
    print(f"Task good {good}/{all_}")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
