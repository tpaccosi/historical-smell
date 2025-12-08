import os
import sys
import re
import math
from sys import getsizeof


import argparse


def create_tags_dict(anno_path):
	# creation dictionary of frame elements
	tags_dict = dict()
	for root, dirs, files in os.walk(anno_path):
		for name in files:
			if name.endswith(".tsv"):

				with open(os.path.join(root, name), 'r') as f:
					for line in f:
						line = line.strip("\n")
						parts = line.split("\t")
						if len(parts) < 5:
							continue

						this_tag = parts[3].split("|")
						for t in this_tag:
							t = re.sub(r'\[[0-9]+]', '', t)
							tags_dict[t] = True

	tag_list = []
	for t in tags_dict.keys():
		if len(t) <2:
			continue
		tag_list.append(t)
	tag_list.sort()

	tag_index = dict()

	for n in range(0,len(tag_list)):
		tag_index[tag_list[n]] = n
	return tag_index


def add_BIO_to_buffer(bufferArray, tlist):
	for n in range(0,len(tag_list)):

		if bufferArray[0][n] == "_" and bufferArray[1][n] == "_":
			continue

		if bufferArray[0][n] != "B-"+bufferArray[1][n] and  bufferArray[0][n] != "I-"+bufferArray[1][n] and bufferArray[0][n] != "_" and  bufferArray[1][n] != "_" :
			bufferArray[0][n] = "B-" + bufferArray[0][n]
			bufferArray[1][n] = "I-" + bufferArray[1][n]
			# print("aaa")
			continue

		if bufferArray[0][n] == "_" and bufferArray[1][n] != "_":
			bufferArray[1][n] = "B-" + bufferArray[1][n]
		if bufferArray[0][n] != "_" and bufferArray[1][n] != "_":
			if bufferArray[0][n] == "B-"+bufferArray[1][n]:
				bufferArray[1][n] = "I-" + bufferArray[1][n]
			if bufferArray[0][n] == "I-"+bufferArray[1][n]:
				bufferArray[1][n] = "I-" + bufferArray[1][n]


	for x in range(0,len(bufferArray[0])):
		if bufferArray[0][x].startswith("B-") or  bufferArray[0][x].startswith("I-") or bufferArray[0][x] == "_":
			continue
		else:
			bufferArray[0][x] = "B-"+bufferArray[0][x]

	bufferToPrint = re.sub('\[[0-9]+\]', '', "\t".join(bufferArray[0]))
	printableSting = ("\t".join(lineArray[0])+"\t"+bufferToPrint)
	return(printableSting)


def make_folds_dict(fold_path):
	for root, dirs, files in os.walk(fold_path):
		for name in files:
			if name.endswith(".tsv"):

				with open(os.path.join(root,name), 'r') as f:

					smellTotal = 0
					for line in f:
						if "Smell\_Word" in line:
							smellTotal +=1
					foldSize = round(smellTotal/folds_number)

				buffer = []
				thisDocLinesList = []
				smellCount = 0

				foldIndex = 0

				with open(os.path.join(root,name), 'r') as f:
					fileNameToPrint = os.path.join(root,name).replace(path,"")
					fileNameToPrint = re.sub(' .+/', '/', fileNameToPrint)
					for line in f:
						line = line.strip("\n")
						if line == "":
							thisDocLinesList.append(line)
							continue

						parts = line.split("\t")

						if len(parts)<5:
							continue

						thisTag = parts[3].split("|")

						newLine = "\t".join(parts[0:3])

						for t1 in tag_list:
							for t2 in thisTag:
								if t1 in t2:
									newLine= newLine+"\t"+t2
								else:
									newLine= newLine+"\t_"
						thisDocLinesList.append(newLine)

					bufferArray = []
					lineArray = []

					bufferToPrintag_list = []

					for l in thisDocLinesList:

						if l == "":
							if foldSize == 0:
								foldIndex = 0
							else:
								foldIndex = math.floor(smellCount/foldSize)

							if foldIndex > folds_number-1: foldIndex = folds_number-1
							for item in bufferToPrintag_list:
								foldsDict[foldIndex].append(item)

							foldsDict[foldIndex].append(l)
							bufferToPrintag_list = []
							bufferArray = []
							lineArray = []
							continue

						parts = l.split("\t")[3:]
						lineBegin = l.split("\t")[:3]
						lineBegin[0] = fileNameToPrint+"\t"+lineBegin[0]

						bufferArray.append(parts)
						lineArray.append(lineBegin)

						if len(bufferArray) < 2:
							continue

						bio_string = add_BIO_to_buffer(bufferArray, tag_list)
						bufferToPrintag_list.append(bio_string)

						if "Smell\\_Word" in bio_string:
							smellCount += 1

						del(bufferArray[0])
						del(lineArray[0])

					# check if there is an annotation on the last word/line of the document:
					try:

						for x in range(0,len(bufferArray[0])):
							if bufferArray[0][x].startswith("B-") or  bufferArray[0][x].startswith("I-") or bufferArray[0][x] == "_":
								continue
							else:
								bufferArray[0][x] = "B-"+bufferArray[0][x]

						bufferToPrint = re.sub('\[[0-9]+\]', '', "\t".join(bufferArray[0]))

						if "Smell\_Word" in bufferToPrint:
							smellCount +=1

						try:
							foldIndex = math.floor(smellCount/foldSize)
						except:
							foldIndex
						if foldIndex > folds_number-1: foldIndex = folds_number-1
						foldsDict[foldIndex].append("\t".join(lineArray[0])+"\t"+bufferToPrint) #####

					except:
						emptyLine = True


def write_fold(fold_id, filename):
	# number of columns until the word (included)
	colums = 4
	array_to_print = []

	for l in folds_dict[fold_id]:
		string_to_print = ""

		if len(l) < 1:
			array_to_print.append("\n")
			continue

		parts = l.split("\t")

		string_to_print = string_to_print + "\t".join(parts[:colums]) + "\t"

		for my_tags in tags_columns:

			if len(parts) <2:
				array_to_print.append("\n")
				continue
			tags_to_add = []
			for t in my_tags:
				tags_to_add.append(parts[colums+tagIndex[t]])
			tags_to_add2 = []
			if "B-" in " ".join(tags_to_add) or "I-" in " ".join(tags_to_add):
				for t in tags_to_add:
					if t != "_":
						tags_to_add2.append(t)
			else:
				tags_to_add2.append("O")
			string_to_print = string_to_print + "|".join(tags_to_add2)
			string_to_print = string_to_print + "\t"

		array_to_print.append(string_to_print)

	while array_to_print[0] == "" or  array_to_print[0] == "\n" :
		del(array_to_print[0])

	index = array_to_print[0].split("\t")[1].split("-")[0]

	with open(filename, 'a') as fh:
		for l in array_to_print:
			if l == "\n":
				continue
			if l.split("\t")[1].split("-")[0] != index:
				index = l.split("\t")[1].split("-")[0]
				fh.write("\n")
			fh.write(l.replace("\\", ""))
			fh.write("\n")


#############
# print files
#############


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
	tasktype = args.tasktype
	folds_number = 5

	tags_columns = []
	if tasktype == "BERT":
		my_list = []
		for l in args.tags.split(","):
			my_list.append(l)
		tags_columns.append(my_list)

	if tasktype == "MULTITASK":
		for l in args.tags.split(","):
			my_list = [l]
			tags_columns.append(my_list)

	folds_dict = dict()
	for n in range(folds_number):
		folds_dict[n] = []
	output_path = args.output

	folder_exist = os.path.exists(output_path)

	if not folder_exist:
		os.mkdir(output_path)

	for i in range(folds_number):
		folds_file = open(os.path.join(output_path, "folds_"+str(i)+"_train.tsv"), 'w')
		folds_file.close()
		folds_file = open(os.path.join(output_path, "folds_"+str(i)+"_dev.tsv"), 'w')
		folds_file.close()
		folds_file = open(os.path.join(output_path, "folds_"+str(i)+"_test.tsv"), 'w')
		folds_file.close()

	# testdev = [[9, 0], [8, 9], [7, 8], [6, 7], [5, 6], [4, 5], [3, 4], [2, 3], [1, 2], [0, 1]]
	testdev = [[4, 0], [3, 4], [2, 3], [1, 2], [0, 1]]

	counter = -1
	for pair in testdev:
		counter += 1
		train = []
		for fold_num in range(folds_number):
			if fold_num not in pair:
				train.append(fold_num)
		dev = []
		test = []
		dev.append(pair[0])
		test.append(pair[1])

		for fold in train:
			write_fold(fold, os.path.join(output_path, "folds_"+str(counter)+"_train.tsv"))
		for fold in dev:
			write_fold(fold, os.path.join(output_path, "folds_"+str(counter)+"_dev.tsv"))
		for fold in test:
			write_fold(fold, os.path.join(output_path, "folds_"+str(counter)+"_test.tsv"))


if __name__ == "__main__":
	main()
