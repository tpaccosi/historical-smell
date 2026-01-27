import os
import re
from collections import defaultdict
from typing import Dict, List, Union
from dataclasses import dataclass


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
        sent_idx (int): The 1-based index of the sentence in the document.
        token_idx (int): The 1-based index of the token in the sentence.
        char_range (str): The character start and end index (e.g., '10-15').
        token (str): The text of the token.
        label (Label): The extracted linguistic label (e.g., 'Smell\\_Word', 'O') and
                disambiguation_ids (if any).
   """
    text_id: str
    sent_idx: int
    token_idx: int
    char_range: str
    token: str
    labels: List[Label]


def parse_label_string(label_string: str, tags: List[str] = None):
    """
   Parses a WebAnno label string to extract the primary label and disambiguation ID.

   WebAnno labels can include multiple annotations separated by '|' and
   disambiguation IDs in brackets (e.g., 'Label1|Label2[1]').

   Args:
       label_string (str): The raw label string from the WebAnno TSV file.
       tags (List[str]): A list of target tags to filter for (default is None which means all tags are included).

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
        if tags is None or label.tag in tags:
            # only keep labels that are in the tags list
            # (ignore other categories)
            labels.append(label)
    # sort the labels by their order in the tags list
    # in the single task case, this determines which single label will be used
    if tags is not None:
        labels.sort(key=lambda x: tags.index(x.tag))
    if len(labels) == 0:
        return [Label('O', None)]
    return labels


def read_tsv_anno_file(anno_file: str, tags: List[str] = None):
    """
   Reads a single Anno TSV file and yields Annotation objects for each token.

   Args:
       text_id (str): The identifier for the text/document.
       anno_file (str): The full path to the WebAnno TSV file.
       tags (List[str]): The list of target tags to consider.

   Yields:
       Annotation: An Annotation object for each token with a valid sentence/token index.

   """
    with open(anno_file, 'rt') as fh:
        for line in fh:
            parts = line.strip('\n').split('\t')
            if len(parts) == 1:
                continue
            elif len(parts) != 5:
                print(f"line: {line}")
                print(f"parts: {parts}")
            text_id, sent_token, char_range, token, label_string = parts
            sent_idx, token_idx = [int(x) for x in sent_token.split('-')]
            # extract tags and disambiguation IDs from label string
            labels = parse_label_string(label_string, tags)
            annotation = Annotation(text_id, sent_idx, token_idx, char_range, token,
                                    labels)
            yield annotation
    return None


def read_web_anno_file(text_id: str, anno_file: str, tags: List[str] = None):
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


def read_web_anno_files(anno_path: str, tags: List[str] = None):
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
            text_annos[text_id] = [anno for anno in read_web_anno_file(text_id, anno_file, tags)]
    # ensure that the annos per text have the correct number of tokens per sentence
    check_text_annos(text_annos)
    return text_annos


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
    return [anno for anno in annos if tag in [label.tag for label in anno.labels]]


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


def write_annos(output_file: str, fold_annos: List[Annotation],
                tags_columns: List[List[str]]):
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


