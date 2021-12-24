import os
import argparse
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from pythainlp import word_tokenize


def get_rouge(hypo, ref):
    rouge = Rouge()
    hypothesis = " ".join(word_tokenize(hypo, keep_whitespace=False))
    reference = " ".join(word_tokenize(ref, keep_whitespace=False))
    scores = rouge.get_scores(hypothesis, reference)[0]
    r1 = scores['rouge-1']['f']
    r2 = scores['rouge-2']['f']
    return {'r1': round(r1, 3), 'r2': round(r2, 3)}


def check_dataset(dataset_name):
    dataset = dataset_name.lower()
    if dataset not in ['th2en', 'th2zh']:
        raise ValueError("Please choose 'th2en' or 'th2zh'.")
    else:
        if dataset == 'th2en':
            return 'en'
        else:
            return 'zh'


def main(dataset, r1_threshold, r2_threshold, input_csv, output_csv):
    if os.path.isfile(output_csv):
        os.remove(output_csv)

    if not os.path.exists(input_csv):
        raise FileNotFoundError('Input CSV does not exist.')

    df = pd.read_csv(input_csv, encoding='utf-8')
    target_lang = check_dataset(dataset)

    filtered = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        rouge_scores = get_rouge(row['th_sum'], row[f'{target_lang}2th'])

        if rouge_scores['r1'] < r1_threshold or rouge_scores['r2'] < r2_threshold:
            filtered = filtered + 1
            continue

        filtered_df = pd.DataFrame()
        filtered_df.loc[index, 'th_body'] = row['th_body']
        filtered_df.loc[index, 'th_sum'] = row['th_sum']

        filtered_df.loc[index, f'{target_lang}_body'] = row[f'{target_lang}_body']
        filtered_df.loc[index, f'{target_lang}_sum'] = row[f'{target_lang}_sum']

        filtered_df.loc[index, 'url'] = row['url']

        filtered_df.loc[index, 'r1'] = rouge_scores['r1']
        filtered_df.loc[index, 'r2'] = rouge_scores['r2']

        if not os.path.isfile(output_csv):
            filtered_df.to_csv(output_csv, index=False, encoding='utf-8-sig', header=filtered_df.columns)
        else:
            filtered_df.to_csv(output_csv, index=False, encoding='utf-8-sig', mode='a', header=False)

    print(f'Filtered dataset contains {len(df) - filtered} articles.')
    print(f'With R1 and R2 at {r1_threshold}, {r2_threshold}, {filtered} articles are filtered out.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Choose 'th2en' or 'th2zh'")
    parser.add_argument('--input_csv', type=str, help='Path to full dataset file')
    parser.add_argument('--output_csv', type=str, help='Path to save filtered CSV')
    parser.add_argument('--r1', type=float, help='ROUGE-1 threshold', default=0.45)
    parser.add_argument('--r2', type=float, help='ROUGE-2 threshold', default=0.2)
    args = parser.parse_args()
    main(args.dataset, args.r1, args.r2, args.input_csv, args.output_csv)
