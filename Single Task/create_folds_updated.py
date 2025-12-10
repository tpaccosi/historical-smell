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

	anno_path: str
	num_folds: int
	task_type: str
	tags_columns: list
	output_path: str


@dataclass
class Annotation:

	text_id: str
	sent_idx: str
	token_idx: str
	char_range: str
	token: str
	label: str
	disambiguation_id: int


def parse_label(label: str, tags: List[str]):
	if label == '_':
		return 'O', None
	labels = label.split('|')
	parsed_labels = []
	for label in labels:
		if m := re.search(r"(.*)\[(\d+)]", label):
			label = m.group(1)
			disambiguation_id = int(m.group(2))
		else:
			disambiguation_id = None
		parsed_labels.append((label, disambiguation_id))
	for tag in tags:
		for label, disambiguation_id in parsed_labels:
			if tag == label:
				return label, disambiguation_id
	return 'O', None


def read_anno_file(text_id: str, anno_file: str, tags: List[str]):
	with open(anno_file, 'rt') as fh:
		for line in fh:
			if len(line) < 1 or line[0].isdigit() is False:
				continue
			if re.match(r"^\d+-\d+\t\d+-\d+\t\S+\t\S+\t", line.strip('\n')):
				sent_token, char_range, token, label_string, *rest = line.strip('\n').split('\t')
				sent_idx, token_idx = [int(x) for x in sent_token.split('-')]
				# remove specification from label
				label, disambiguation_id = parse_label(label_string, tags)
				if label is None:
					raise ValueError(f"Extracted labels do not match tags set in file {anno_file}: {label_string}")
				annotation = Annotation(text_id, sent_idx, token_idx, char_range, token, label, disambiguation_id)
				yield annotation
	return None


def extract_text_id(root: str):
	_, fname = os.path.split(root)
	basename, _ = os.path.splitext(fname)
	return re.sub(r'\W', '_', basename)


def read_anno_files(anno_path: str, tags: List[str]):
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


def filter_annotations(annos: List[Annotation], label: str = 'Smell\\_Word') -> List[Annotation]:
	return [anno for anno in annos if anno.label == label]


def split_annos(num_folds: int, text_annos: Dict[str, any], seed: int):
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
		text_label_freq[text_id] = Counter([anno.label  for anno in text_annos[text_id]])
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


#############
# print files
#############
def get_tags_columns(tags: List[str], task_type: str):
	tags_columns = []
	if task_type == "BERT":
		tags_columns.append(tags)

	elif task_type == "MULTITASK":
		tags_columns.extend([[tag] for tag in tags])
	else:
		raise ValueError(f"task_type must be one of {'BERT', 'MULTITASK'}, not {task_type}.")
	return tags_columns


def assign_folds(test_idx: int, dev_idx: int, folds: List[List[Annotation]]):
	test_fold, dev_fold = folds[test_idx], folds[dev_idx]
	train_folds = [fold for fi, fold in enumerate(folds) if fi not in {test_idx, dev_idx}]
	train_fold = [anno for fold in train_folds for anno in fold]
	return test_fold, dev_fold, train_fold


def make_anno_tsv_line(anno: Annotation, prev_label: str, prev_dis_id: Union[int, None]):
	sent_token = f"{anno.sent_idx}-{anno.token_idx}"
	if anno.label == 'O':
		tag = anno.label
	elif anno.label != prev_label:
		tag = f'B-{anno.label}'
	elif anno.label == prev_label and anno.disambiguation_id != prev_dis_id:
		tag = f'B-{anno.label}'
	else:
		tag = f'I-{anno.label}'
	tag = tag.replace('\\', '')
	return "\t".join([anno.text_id, sent_token, anno.char_range, anno.token, tag])


def write_annos(output_path: str, fold_num: int,
				fold_annos: List[Annotation], fold_type: str):
	output_file = os.path.join(output_path, f"folds_{fold_num}_{fold_type}.tsv")
	with open(output_file, 'wt') as fh:
		prev_sent_idx = None
		prev_label = None
		prev_dis_id = None
		for anno in fold_annos:
			if prev_sent_idx is not None and anno.sent_idx != prev_sent_idx:
				fh.write("\n")
				prev_label = None
				prev_dis_id = None
			line = make_anno_tsv_line(anno, prev_label, prev_dis_id)
			fh.write(f"{line}\n")
			prev_sent_idx = anno.sent_idx
			prev_label = anno.label
			prev_dis_id = anno.disambiguation_id


def main():
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
		write_annos(output_path, fold_num, test_annos, fold_type='test')
		write_annos(output_path, fold_num, dev_annos, fold_type='dev')
		write_annos(output_path, fold_num, train_annos, fold_type='train')


if __name__ == "__main__":
	main()
