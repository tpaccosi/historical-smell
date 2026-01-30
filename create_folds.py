import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List

from hist_smell.utils.annotation import Annotation, FoldConfig, read_web_anno_files, write_annos


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
    print(f"output path: {output_path}")

    text_annos = read_web_anno_files(fold_config.anno_path, tags)
    print(f"num files: {len(text_annos)}")
    print(f"num annotations: {sum(len(text_annos[text_id]) for text_id in text_annos)}")

    seed = 36437
    folds = split_annos(num_folds=folds_number, text_annos=text_annos, seed=seed)
    # testdev = [[9, 0], [8, 9], [7, 8], [6, 7], [5, 6], [4, 5], [3, 4], [2, 3], [1, 2], [0, 1]]
    testdev = [[4, 0], [3, 4], [2, 3], [1, 2], [0, 1]]

    for fold_num, pair in enumerate(testdev):
        test_idx, dev_idx = pair
        test_annos, dev_annos, train_annos = assign_folds(test_idx, dev_idx, folds)
        output_file = os.path.join(output_path, f"folds_{fold_num}_test.tsv")
        write_annos(output_file, test_annos, tags_columns=tags_columns)
        output_file = os.path.join(output_path, f"folds_{fold_num}_dev.tsv")
        write_annos(output_file, dev_annos, tags_columns=tags_columns)
        output_file = os.path.join(output_path, f"folds_{fold_num}_train.tsv")
        write_annos(output_file, train_annos, tags_columns=tags_columns)


if __name__ == "__main__":
    main()
