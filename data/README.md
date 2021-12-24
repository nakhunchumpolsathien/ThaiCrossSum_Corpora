#TH-ZH Machine Translation Test Set
`th-zh_mt_test.csv` contains approximately 600 pairs of Thai and Chinese (simp) sentences. It is a small test set I created to initially evaluate machine translation task in XLS and NCLS-CLS+MT. These sentences are translated by human. They are collected from [jeenmix.com](http://www.jeenmix.com) and [pasajeen.com](https://pasajeen.com/%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B9%82%E0%B8%A2%E0%B8%84%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B2%E0%B8%88%E0%B8%B5%E0%B8%99-1000-%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B9%82%E0%B8%A2%E0%B8%84%E0%B8%A0%E0%B8%B2%E0%B8%A9%E0%B8%B2%E0%B8%88%E0%B8%B5%E0%B8%99%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B9%83%E0%B8%8A%E0%B9%89%E0%B8%9A%E0%B9%88%E0%B8%AD%E0%B8%A2/). 

[test.MT.source.TH(ZH).txt](test.MT.source.TH(ZH).txt) and [test.MT.target.ZH.txt](test.MT.target.ZH.txt) are ready-to-be-used files to evaluate Chinese MT task.

#TH-EN Machine Translation Test Set
[test.MT.source.TH(EN).txt](test.MT.source.TH(EN).txt) and [test.MT.target.EN.txt](test.MT.target.EN.txt) are small test sets for evaluating TH-EN MT task. I randomly selected 3000 rows from `generated_reviews_crowd.csv ` to create these test sets.

`generated_reviews_crowd.csv` is part of [scb-mt-en-th-2020](https://arxiv.org/abs/2007.03541), A Large English-Thai Parallel Corpus, and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.th).

#Create MT Test Files for NCLS and XLS

```
python src/tools/create_mt_test_manifest.py \
--mode {th2en/th2zh} \
--input_csv_path path/to/inputcsv.csv \
--number_of_samples 3000 \
--lowercase True \
--output_dir path/to/directory/to/save/these_precessed_test_files
```
* `mode` can be `th2en` or `th2zh`, depends on input csv.
* `input_csv_path` is path to input csv file. For TH2EN, any csv file from scb-mt-en-th-2020 will do.
* `number_of_samples` is number of sentence (or utterances) pairs you want. Default is `3000`. If desired number is larger than the actual number of rows in input csv, it will switch to number of total rows in the csv instead.
* `lowercase` whether to convert English characters to lowercase. Default is `True`.

