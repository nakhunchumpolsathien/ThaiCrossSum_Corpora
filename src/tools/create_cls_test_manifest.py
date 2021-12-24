import os
import codecs
import stanza
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from pythainlp import word_tokenize

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

zh_nlp = stanza.Pipeline('zh', processors='tokenize', verbose=False, use_gpu=True)
en_nlp = stanza.Pipeline('en', processors='tokenize', verbose=False, use_gpu=True)

'''
if found 'stanza.pipeline.core.LanguageNotDownloadedError' error:
please download en and zh modes first using below command

stanza.download('en')
stanza.download('zh') 
'''


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_file(text, output_path):
    with codecs.open(output_path, mode='a', encoding="utf-8") as txtfile:
        txtfile.write(f'{text}\n')


def delete_if_exists(file_path):
    if os.path.exists(file_path):
        logging.warning(f"{file_path} already exists. It is now deleted.")
        os.remove(file_path)


def clean_text(text):
    text = text.replace('"', '').replace("'", '')
    text = text.replace('!', '').replace('(', '')
    text = text.replace(')', '').replace('-', '')
    text = text.replace('#', '')
    text = text.lower()
    return text


def tokenize_zh(text):
    zh_doc = zh_nlp(text)
    tokenized_words = []
    for i, sent in enumerate(zh_doc.sentences):
        for word in sent.words:
            tokenized_words.append(word.text)
    return {'num_words': zh_doc.num_words, 'tokenized_text': tokenized_words}


def tokenize_en(text):
    en_doc = en_nlp(text)
    tokenized_words = []
    for i, sent in enumerate(en_doc.sentences):
        for word in sent.words:
            tokenized_words.append(word.text)
    return {'num_words': en_doc.num_words, 'tokenized_text': tokenized_words}


def tokenize_th(text):
    th_text = clean_text(text)
    th_doc = word_tokenize(th_text, keep_whitespace=False)
    return {'num_words': len(th_doc), 'tokenized_text': th_doc}


def main(csv_path, max_words, output_dir, use_gg_sum, ms_ref_is_enabled):
    th_src_path = os.path.join(output_dir, 'test.CLS.source.TH.txt')
    en_tgt_path = os.path.join(output_dir, 'test.CLS.target.EN.txt')
    zh_tgt_path = os.path.join(output_dir, 'test.CLS.target.ZH.txt')
    delete_if_exists(th_src_path)
    delete_if_exists(en_tgt_path)
    delete_if_exists(zh_tgt_path)

    if ms_ref_is_enabled:
        th_tgt_path = os.path.join(output_dir, 'test.MS.target.TH.txt')
        delete_if_exists(th_tgt_path)

    logging.info(f'Read test set from {csv_path}.')
    df = pd.read_csv(csv_path, encoding='utf-8')

    if use_gg_sum:
        en_sum_ref = 'en_gg_sum'
        zh_sum_ref = 'zh_gg_sum'
        logging.info("Use Google Translation summary references")

    else:
        en_sum_ref = 'en_sum'
        zh_sum_ref = 'zh_sum'
        logging.info("Use main summary references")

    logging.info(f'Input article will be truncated to {max_words} words.')
    logging.info(f'Start tokenizing...')

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        th_body = ' '.join(tokenize_th(row['th_body'])['tokenized_text'][:max_words])
        en_sum = ' '.join(tokenize_en(row[en_sum_ref])['tokenized_text']).lower()
        zh_sum = ' '.join(tokenize_zh(row[zh_sum_ref])['tokenized_text']).lower()

        write_file(th_body, th_src_path)
        write_file(en_sum, en_tgt_path)
        write_file(zh_sum, zh_tgt_path)

        if ms_ref_is_enabled:
            th_sum = ' '.join(tokenize_th(row['th_sum'])['tokenized_text'])
            write_file(th_sum, th_tgt_path)

    logging.info(f'Done. Output files are saved at {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv_path', type=str, help='Path to test_set.csv')
    parser.add_argument('--output_dir', type=str, help='Path to output directory',
                        default='test_set')
    parser.add_argument('--use_google_sum', type=str2bool, help='Use Google translation summary reference',
                        default=False)
    parser.add_argument('--create_ms_ref', type=str2bool, help='Whether to create reference file for MS '
                                                                            'task.',
                        default=False)
    parser.add_argument('--max_tokens', type=int, help='Maximum number of words in input articles', default=500)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f'{args.output_dir} is created.')

    main(args.test_csv_path, args.max_tokens, args.output_dir, args.use_google_sum, args.create_ms_ref)
