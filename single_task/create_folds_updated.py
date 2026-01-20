import os
import random
import sys
import re
import math
from collections import Counter
from collections import defaultdict
from sys import getsizeof
from typing import Dict, List, Union
from dataclasses import dataclass, Field

import argparse


@dataclass
class FoldConfig:
    """
   Configuration settings for the k-fold split and output.

   Attributes:
       anno_path (str): Path to the input folder containing WebAnno TSV files.
       num_folds (int): The total number of folds (k) for cross-validation.
       task_type (str): The format type for output ('BERT' or 'MULTITASK').
       tags_columns (list): The list of tags/columns to be extracted and used.
       output_path (str): Path to the output directory for the split files.
   """
    anno_path: str
    num_folds: int
    task_type: str
    tags_columns: list
    output_path: str


@dataclass
class Label:
    """
    Represents a single label of a token's annotation.

    See https://webanno.github.io/webanno/releases/3.6.11/docs/user-guide.html#_disambiguation_ids
    for explanation and examples.

    Attributes:
        tag: the tag corresponding to a class in the tag set
        disambiguation_id: an integer indicating which tokens
                in a multi-token spans belong to the same span
                or that there are stacked annotations (one token
                is annotated with multiple tags)
    """
    tag: str
    disambiguation_id: Union[int, None]

@dataclass
class Annotation:
    """
    Represents a single token's annotation from a WebAnno TSV line.

    Attributes:
        text_id (str): A unique identifier for the source document.
        sent_idx (str): The 1-based index of the sentence in the document.
        token_idx (str): The 1-based index of the token in the sentence.
        char_range (str): The character start and end index (e.g., '10-15').
        token (str): The text of the token.
        label (Label): The extracted linguistic label (e.g., 'Smell\\_Word', 'O') and
                disambiguation_ids (if any).
   """
    text_id: str
    sent_idx: str
    token_idx: str
    char_range: str
    token: str
    labels: List[Label]


def parse_label_string(label_string: str, tags: List[str]):
    """
   Parses a WebAnno label string to extract the primary label and disambiguation ID.

   WebAnno labels can include multiple annotations separated by '|' and
   disambiguation IDs in brackets (e.g., 'Label1|Label2[1]').

   Args:
       label_string (str): The raw label string from the WebAnno TSV file.
       tags (List[str]): A list of target tags to filter for.

   Returns:
       Tuple[str, Union[int, None]]: The filtered label ('O' if no match) and the
           disambiguation ID (int or None).
   """
    if label_string == '_':
        return [Label('O', None)]
    label_strings = label_string.split('|')
    labels = []
    for label_string in label_strings:
        if m := re.search(r"(.*)\[(\d+)]", label_string):
            tag = m.group(1)
            disambiguation_id = int(m.group(2))
        else:
            tag = label_string
            disambiguation_id = None
        label = Label(tag, disambiguation_id)
        if label.tag in tags:
            # only keep labels that are in the tags list
            # (ignore other categories)
            labels.append(label)
    # sort the labels by their order in the tags list
    # in the single task case, this determines which single label will be used
    labels.sort(key=lambda x: tags.index(x.tag))
    if len(labels) == 0:
        return [Label('O', None)]
    return labels


def read_anno_file(text_id: str, anno_file: str, tags: List[str]):
    """
   Reads a single WebAnno TSV file and yields Annotation objects for each token.

   Args:
       text_id (str): The identifier for the text/document.
       anno_file (str): The full path to the WebAnno TSV file.
       tags (List[str]): The list of target tags to consider.

   Yields:
       Annotation: An Annotation object for each token with a valid sentence/token index.

   """
    with open(anno_file, 'rt') as fh:
        for line in fh:
            if len(line) < 1 or line[0].isdigit() is False:
                continue
            if re.match(r"^\d+-\d+\t\d+-\d+\t\S+\t\S+\t", line.strip('\n')):
                sent_token, char_range, token, label_string, *rest = line.strip('\n').split('\t')
                sent_idx, token_idx = [int(x) for x in sent_token.split('-')]
                # extract tags and disambiguation IDs from label string
                labels = parse_label_string(label_string, tags)
                annotation = Annotation(text_id, sent_idx, token_idx, char_range, token,
                                        labels)
                yield annotation
    return None


def extract_text_id(root: str):
    """
   Generates a text identifier from the file path, typically the filename without extension.

   Args:
       root (str): The file path or a part of the path (e.g., directory or filename).

   Returns:
       str: A cleaned-up string suitable for use as a text ID.
   """
    _, fname = os.path.split(root)
    basename, _ = os.path.splitext(fname)
    text_id = re.sub(r'\W', '_', basename)
    # print(f"root: {root}\nfname: {fname}\nbasename: {basename}\ntext_id: {text_id}\n")
    return text_id


def read_anno_files(anno_path: str, tags: List[str]):
    """
   Recursively searches a directory for WebAnno TSV files and reads all annotations.

   Args:
       anno_path (str): The root directory to search for TSV files.
       tags (List[str]): The list of target tags to consider during parsing.

   Returns:
       Dict[str, List[Annotation]]: A dictionary mapping text IDs to a list of
           all Annotation objects in that document.
   """
    text_annos = {}
    for root, dirs, files in os.walk(anno_path):
        for fname in files:
            if not fname.endswith(".tsv"):
                continue
            text_id = extract_text_id(root)
            anno_file = os.path.join(root, fname)
            text_annos[text_id] = [anno for anno in read_anno_file(text_id, anno_file, tags)]
    # ensure that the annos per text have the correct number of tokens per sentence
    check_text_annos(text_annos)
    return text_annos


def check_text_annos(text_annos: Dict[str, List[Annotation]]):
    """
   Performs a sanity check on the loaded annotations to ensure sentence consistency.

   The check verifies that for every sentence, the total number of tokens (length of the
   list) equals the maximum token index found in that sentence.

   Args:
       text_annos (Dict[str, List[Annotation]]): Annotations grouped by text ID.

   Returns:
       None

   Raises:
       ValueError: If the number of tokens in a sentence does not match the max token index.
   """
    for text_id in text_annos:
        sent_annos = defaultdict(list)
        for anno in text_annos[text_id]:
            sent_annos[anno.sent_idx].append(anno)
        for sent_idx in sent_annos:
            max_token_idx = sent_annos[sent_idx][-1].token_idx
            if len(sent_annos[sent_idx]) != max_token_idx:
                for anno in text_annos[text_id]:
                    print(anno)
                raise ValueError(f"number of tokens for sent {sent_idx} in text {text_id} not equal "
                                 f"to max token_idx {max_token_idx}")
    return None


def filter_annotations(annos: List[Annotation], tag: str = 'Smell\\_Word') -> List[Annotation]:
    """
   Filters a list of annotations, keeping only those that match a specified label.

   Args:
       annos (List[Annotation]): The list of annotations to filter.
       tag (str): The target tag to keep. Defaults to 'Smell\\_Word'.

   Returns:
       List[Annotation]: The filtered list of annotations.
   """
    return [anno for anno in annos if tag in anno.label.tags]


def split_annos(num_folds: int, text_annos: Dict[str, any], seed: int):
    """
   Splits the annotations into `num_folds` partitions, aiming for a stratified split
   based on the frequency of the 'Smell\\_Word' label across texts.

   The stratification attempts to ensure each fold has an approximately equal
   number of 'Smell\\_Word' instances.

   Args:
       num_folds (int): The desired number of folds (k).
       text_annos (Dict[str, any]): Annotations grouped by text ID.
       seed (int): The random seed for shuffling text IDs.

   Returns:
       List[List[Annotation]]: A list of folds, where each fold is a list of Annotation objects.

   Raises:
       ValueError: If the final number of created folds does not equal `num_folds`.
   """
    # step 1: shuffle text_ids
    shuffled_ids = list(text_annos.keys())
    random.seed(seed)
    random.shuffle(shuffled_ids)
    folds = []
    # step 2: create fold list
    fold = []
    fold_sw_count = 0
    # step 3: count smell words per text and in total
    text_label_freq = defaultdict(Counter)
    for text_id in text_annos:
        text_tags = [label.tag for anno in text_annos[text_id] for label in anno.labels]
        text_label_freq[text_id] = Counter(text_tags)
    total_sw_count = sum(text_label_freq[text_id]['Smell\\_Word'] for text_id in text_label_freq)
    # step 4: calculate smell words per fold
    print(f"num_folds: {num_folds}")
    fold_sw_threshold = total_sw_count / num_folds
    print(f"total smell word count: {total_sw_count}, fold smell word threshold: {fold_sw_threshold}")
    # step 5: iterate over text_ids
    for text_id in shuffled_ids:
        # step 5a: count smell words in fold
        fold_sw_count += text_label_freq[text_id]['Smell\\_Word']
        # step 5b: add annos from file to fold list
        fold.extend(text_annos[text_id])
        # step 5c: if smell words in fold exceeds threshold, create new fold
        if fold_sw_count >= fold_sw_threshold:
            print(f"fold: {len(folds)}, smell words in fold: {fold_sw_count}")
            folds.append(fold)
            fold = []
            fold_sw_count = 0
    if len(fold) > 0:
        print(f"fold: {len(folds)}, smell words in fold: {fold_sw_count}")
        folds.append(fold)
    # step 6: check that we have the correct number of folds
    if len(folds) != num_folds:
        raise ValueError(f"incorrect number of folds: {len(folds)} instead of {num_folds}")
    return folds


def get_tags_columns(tags: List[str], task_type: str):
    """
   Determines the structure of the output columns based on the `task_type`.

   Args:
       tags (List[str]): The list of all target tags.
       task_type (str): The desired output format ('SINGLETASK' or 'MULTITASK').

   Returns:
       List[List[str]]: A list defining the tag structure for the output.
           - 'SINGLETASK': [['Tag1', 'Tag2', ...]] (All tags in one column).
           - 'MULTITASK': [['Tag1'], ['Tag2'], ...] (Each tag in a separate column).

   Raises:
       ValueError: If `task_type` is not 'SINGLETASK' or 'MULTITASK'.
   """
    tags_columns = []
    if task_type == 'SINGLETASK':
        tags_columns.append(tags)
    elif task_type == "MULTITASK":
        tags_columns.extend([[tag] for tag in tags])
    else:
        raise ValueError(f"task_type must be one of {'SINGLETASK', 'MULTITASK'}, not {task_type}.")
    return tags_columns


def assign_folds(test_idx: int, dev_idx: int, folds: List[List[Annotation]]):
    """
   Assigns folds to the test, development (dev), and training (train) sets
   based on their indices.

   Args:
       test_idx (int): The index of the fold to be used for testing.
       dev_idx (int): The index of the fold to be used for development/validation.
       folds (List[List[Annotation]]): The complete list of all folds.

   Returns:
       Tuple[List[Annotation], List[Annotation], List[Annotation]]: The test,
           dev, and train annotation lists, respectively.
   """
    test_fold, dev_fold = folds[test_idx], folds[dev_idx]
    train_folds = [fold for fi, fold in enumerate(folds) if fi not in {test_idx, dev_idx}]
    train_fold = [anno for fold in train_folds for anno in fold]
    return test_fold, dev_fold, train_fold


def make_anno_tsv_line(anno: Annotation, prev_labels: List[Label],
                       tags_columns: List[List[str]]):
    """
   Converts an Annotation object into a Bi-I-O tagged TSV line (IOB2 format).

   The B-I distinction is determined by checking if the current label is different
   from the previous label or if the disambiguation ID has changed within the same label.

   Args:
       anno (Annotation): The current Annotation object.
       prev_labels (List[Label]): The list of labels of the *previous* token in the sentence.
       tags_columns (List[List[str]]) A list of columns with per column the tags the should
            go in that column.

   Returns:
       str: A tab-separated string representing the IOB2-tagged annotation line.
   """
    sent_token = f"{anno.sent_idx}-{anno.token_idx}"
    tags = []
    for tag_column in tags_columns:
        for label in anno.labels:
            if label == 'O' or label.tag not in tag_column:
                tag = 'O'
            else:
                prev_token_tag = [prev_label for prev_label in prev_labels if prev_label.tag == label.tag]
                if len(prev_token_tag) == 0:
                    tag = f'B-{label.tag}'
                else:
                    prev_token_tag = prev_token_tag[0]
                    if label.disambiguation_id != prev_token_tag.disambiguation_id:
                        tag = f'B-{label.tag}'
                    else:
                        tag = f'I-{label.tag}'
            tag = tag.replace('\\', '')
            tags.append(tag)
            break
    if len(tags_columns) == 1:
        tags = tags[:1]
    if len(tags_columns) > 1:
        assert len(tags_columns) == len(tags), (f"Number of tags ({len(tags)}) is different from "
                                                f"number of tags columns ({len(tags_columns)})")
    # if any(tag != 'O' for tag in tags):
    #     print(f"tags: {tags}")
    tag_string = '\t'.join(tags)
    return "\t".join([anno.text_id, sent_token, anno.char_range, anno.token, tag_string])


def write_annos(output_path: str, fold_num: int, fold_annos: List[Annotation],
                fold_type: str, tags_columns: List[List[str]]):
    """
   Writes a list of annotations to a single output TSV file in IOB2 format.

   Args:
       output_path (str): The root directory for output files.
       fold_num (int): The current fold number (for filename).
       fold_annos (List[Annotation]): The annotations to write (train, dev, or test set).
       fold_type (str): The type of the fold ('train', 'dev', or 'test').
       tags_columns (List[List[str]]): The list of columns and per columns which tags it
            should contain.
   """
    output_file = os.path.join(output_path, f"folds_{fold_num}_{fold_type}.tsv")
    with open(output_file, 'wt') as fh:
        prev_sent_idx = None
        prev_labels = []
        for anno in fold_annos:
            if prev_sent_idx is not None and anno.sent_idx != prev_sent_idx:
                fh.write("\n")
                prev_labels = []
            line = make_anno_tsv_line(anno, prev_labels, tags_columns)
            fh.write(f"{line}\n")
            prev_sent_idx = anno.sent_idx
            prev_labels = anno.labels


def main():
    """
   Main function to execute the WebAnno annotation reading, k-fold splitting,
   and output file generation.

   1. Parses command-line arguments (input folder, output folder, task type, tags).
   2. Reads and validates all annotations from the input path.
   3. Splits the texts into k folds with approximate stratification based on a target label.
   4. Iterates through the k-fold combinations (train/dev/test) and writes the
      resulting annotation splits to TSV files in IOB2 format.
   """
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Input folder containing INCEPTION exports", metavar="FOLDER", required=True)
    parser.add_argument("--output", help="Output folders", metavar="FOLDER", required=True)
    parser.add_argument("--tasktype", choices={"BERT", "MULTITASK"}, help="BERT or MULTITASK", metavar="TASK",
                        default="BERT", type=str)
    parser.add_argument("--tags", help="List of labels comma separated", metavar="TASK",
                        default="Smell\\_Word,Smell\\_Source,Quality", type=str)
    args = parser.parse_args()

    anno_path = args.folder
    task_type = args.tasktype
    if task_type == 'BERT':
        task_type = 'SINGLETASK'
    folds_number = 5

    tags = args.tags.split(",")
    tags_columns = get_tags_columns(tags, task_type)
    print(f"tags: {tags_columns}")

    output_path = args.output

    fold_config = FoldConfig(anno_path=anno_path, num_folds=folds_number,
                             task_type=task_type, tags_columns=tags_columns,
                             output_path=output_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    text_annos = read_anno_files(fold_config.anno_path, tags)
    print(f"num files: {len(text_annos)}")
    print(f"num annotations: {sum(len(text_annos[text_id]) for text_id in text_annos)}")

    seed = 36437
    folds = split_annos(num_folds=folds_number, text_annos=text_annos, seed=seed)
    # testdev = [[9, 0], [8, 9], [7, 8], [6, 7], [5, 6], [4, 5], [3, 4], [2, 3], [1, 2], [0, 1]]
    testdev = [[4, 0], [3, 4], [2, 3], [1, 2], [0, 1]]

    for fold_num, pair in enumerate(testdev):
        test_idx, dev_idx = pair
        test_annos, dev_annos, train_annos = assign_folds(test_idx, dev_idx, folds)
        write_annos(output_path, fold_num, test_annos, fold_type='test', tags_columns=tags_columns)
        write_annos(output_path, fold_num, dev_annos, fold_type='dev', tags_columns=tags_columns)
        write_annos(output_path, fold_num, train_annos, fold_type='train', tags_columns=tags_columns)


if __name__ == "__main__":
    main()
