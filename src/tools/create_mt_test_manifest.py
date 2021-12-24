import os
import codecs
import stanza
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from pythainlp import word_tokenize

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def tokenize_zh(text):
    zh_doc = nlp(clean_text(text))
    tokenized_words = []
    for i, sent in enumerate(zh_doc.sentences):
        for word in sent.words:
            tokenized_words.append(word.text)
    return {'num_words': zh_doc.num_words, 'tokenized_text': tokenized_words}


def tokenize_en(text):
    en_doc = nlp(clean_text(text))
    tokenized_words = []
    for i, sent in enumerate(en_doc.sentences):
        for word in sent.words:
            tokenized_words.append(word.text)
    return {'num_words': en_doc.num_words, 'tokenized_text': tokenized_words}


def clean_text(text):
    text = text.replace('"', '').replace(':', '')
    text = text.replace('!', '').replace('(', '')
    text = text.replace(')', '').replace('-', '')
    text = text.replace(',', '').replace('.', '') # remove period
    text = text.replace("'ll ", ' will ').replace('。', '')
    text = text.replace('、', '').replace('，', '')
    return text


def tokenize_th(text):
    th_text = clean_text(text)
    th_doc = word_tokenize(th_text, keep_whitespace=False)
    return {'num_words': len(th_doc), 'tokenized_text': th_doc}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_file(text, output_path, is_lower):
    with codecs.open(output_path, mode='a', encoding="utf-8") as txtfile:
        if is_lower:
            text = text.lower()

        txtfile.write(f'{text}\n')


def delete_if_exists(file_path):
    if os.path.exists(file_path):
        logging.warning(f"{file_path} already exists. It is now deleted.")
        os.remove(file_path)


def main(mode, input_csv, output_dir, is_lowercase, no_of_samples):
    logging.info(f'Mode {mode.upper()}')
    if mode == 'th2en':
        tgt = 'en'
        tokenizer = tokenize_en
    else:
        tgt = 'zh'
        tokenizer = tokenize_en

    src_text_path = os.path.join(output_dir, f'test.MT.source.TH({tgt.upper()}).txt')
    tgt_text_path = os.path.join(output_dir, f'test.MT.target.{tgt.upper()}.txt')

    delete_if_exists(src_text_path)
    delete_if_exists(tgt_text_path)

    logging.info(f'Read CSV from: {input_csv}.')
    df = pd.read_csv(input_csv, encoding='utf-8')

    if len(df) > no_of_samples:
        df = df.sample(n=no_of_samples)

    logging.info(f'Number of sample: {len(df)} pairs')
    logging.info(f'Start tokenizing...')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        src_text = ' '.join(tokenize_th(row['th_text'])['tokenized_text'])
        tgt_text = ' '.join(tokenizer(row[f'{tgt.lower()}_text'])['tokenized_text'])

        write_file(src_text, src_text_path, is_lowercase)
        write_file(tgt_text, tgt_text_path, is_lowercase)
    logging.info(f'Done. Output files are saved at {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        choices=['th2en', 'th2zh'],
                        help='Can be th2en or th2zh')
    parser.add_argument('--number_of_samples', type=int, help='Number of samples', default=3000)
    parser.add_argument('--input_csv_path', type=str, help='Path to input CSV')
    parser.add_argument('--lowercase', type=str2bool, help='Lower-case English character', default=True)
    parser.add_argument('--output_dir', type=str, help='Path to output directory', default='test_set')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f'{args.output_dir} is created.')

    nlp = stanza.Pipeline(args.mode.split('2')[-1].lower(), processors='tokenize', verbose=False, use_gpu=True)

    main(args.mode, args.input_csv_path, args.output_dir, args.lowercase, args.number_of_samples)
