prepare: download unzip make_data create_folds_single create_venv

download:
	git clone git@github.com:Odeuropa/benchmarks_and_corpora.git

unzip: unzip_de unzip_en unzip_fr unzip_it unzip_nl

unzip_en:
	cd benchmarks_and_corpora/benchmarks/EN/;unzip webanno.zip;rm -rf __MACOSX

unzip_de:
	cd benchmarks_and_corpora/benchmarks/DE/;unzip webanno.zip;rm -rf __MACOSX

unzip_nl:
	cd benchmarks_and_corpora/benchmarks/NL/;unzip webanno.zip;rm -rf __MACOSX

unzip_fr:
	cd benchmarks_and_corpora/benchmarks/FR/;unzip webanno.zip;rm -rf __MACOSX

unzip_it:
	cd benchmarks_and_corpora/benchmarks/IT/;unzip webanno.zip;rm -rf __MACOSX

make_data:
	mkdir data

create_folds_single:
	cd Single\ Task; python3 create_folds.py --folder ..//benchmarks_and_corpora/benchmarks/DE/webanno/ --output ../data/single/data_german --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	cd Single\ Task; python3 create_folds.py --folder ..//benchmarks_and_corpora/benchmarks/EN/webanno/ --output ../data/single/data_english --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	cd Single\ Task; python3 create_folds.py --folder ..//benchmarks_and_corpora/benchmarks/FR/webanno/ --output ../data/single/data_french --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	cd Single\ Task; python3 create_folds.py --folder ..//benchmarks_and_corpora/benchmarks/IT/webanno/ --output ../data/single/data_italian --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	cd Single\ Task; python3 create_folds.py --folder ..//benchmarks_and_corpora/benchmarks/NL/webanno/ --output ../data/single/data_dutch --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier

create_folds_updated_single:
	#cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/DE/webanno/ --output ../data/single/data_german --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	#cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/EN/webanno/ --output ../data/single/data_english --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	#cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/FR/webanno/ --output ../data/single/data_french --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	#cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/IT/webanno/ --output ../data/single/data_italian --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/NL/webanno/ --output ../data/single/data_dutch --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier

create_folds_updated_multi:
	cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/DE/webanno/ --output ../data/multi/data_german --tasktype MULTITASK --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/EN/webanno/ --output ../data/multi/data_english --tasktype MULTITASK --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/FR/webanno/ --output ../data/multi/data_french --tasktype MULTITASK --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/IT/webanno/ --output ../data/multi/data_italian --tasktype MULTITASK --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier
	cd Single\ Task; python3 create_folds_updated.py --folder ..//benchmarks_and_corpora/benchmarks/NL/webanno/ --output ../data/multi/data_dutch --tasktype MULTITASK --tags Smell\\_Word,Smell\\_Source,Quality,Circumstances,Location,Perceiver,Time,Evoked\\_Odorant,Effect,Odour\\_Carrier

create_venv:
	python3 -m venv . venv

