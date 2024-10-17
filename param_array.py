import json
import os
import re

# from itertools import product

queues = [
    'gpu@@nlp-a10',
    'gpu@@nlp-gpu',
    'gpu@@csecri',
    'gpu@@crc_gpu',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pair', required=True, help='source-target language pair')
    parser.add_argument(
        '--train-data', metavar='FILE_PATH', required=True, help='parallel training data'
    )
    parser.add_argument(
        '--val-data', metavar='FILE_PATH', required=True, help='parallel validation data'
    )
    parser.add_argument('--lem-train', metavar='FILE_PATH', help='lemmatized training data')
    parser.add_argument('--lem-val', metavar='FILE_PATH', help='lemmatized validation data')
    parser.add_argument('--dict', metavar='FILE_PATH', help='bilingual dictionary')
    parser.add_argument('--freq', metavar='FILE_PATH', help='frequency statistics')
    parser.add_argument('--sw-vocab', metavar='FILE_PATH', required=True, help='subword vocab')
    parser.add_argument('--sw-model', metavar='FILE_PATH', required=True, help='subword model')
    parser.add_argument('--model', required=True, help='translation model')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--conda', metavar='ENV', required=True, help='conda environment')
    parser.add_argument('--start', metavar='INDEX', type=int, default=1, help='starting index')
    parser.add_argument('--email', required=True, help='email address')
    parser.add_argument(
        '--test-data', nargs='+', metavar='FILE_PATH', required=True, help='detokenized test data'
    )
    parser.add_argument('--metric', nargs='+', required=True, help='evaluation metric')
    args = parser.parse_args()

    param_array = []
    with open('param_array.json') as json_file:
        for option, values in json.load(json_file).items():
            param_array.append([(option, value) for value in values])
    qf_submit = 'qf submit --queue ' + ' --queue '.join(queues)
    email = f'-M {args.email} -m abe' if args.email else ''
    src_lang, tgt_lang = args.lang_pair.split('-')

    # for i, params in enumerate(product(*param_array), start=args.start):
    for i in range(1, 6):
        os.system(f'mkdir -p {args.model}')
        job_name = f"{args.model}_{str(i).rjust(3, '0')}"
        with open(f'{args.model}/{job_name}.sh', 'w') as job_file:
            job_file.write('#!/bin/bash\n\n')
            job_file.write(f'touch {args.model}/{job_name}.log\n')
            job_file.write(f'fsync -d 30 {args.model}/{job_name}.log &\n')

            job_file.write(f'\nconda activate {args.conda}\n')
            job_file.write('export PYTHONPATH="${PYTHONPATH}:${pwd}"\n')
            job_file.write('export SACREBLEU_FORMAT=text\n')

            job_file.write('\npython translation/main.py  \\\n')
            job_file.write(f'  --lang-pair {args.lang_pair} \\\n')
            job_file.write(f'  --train-data {args.train_data} \\\n')
            job_file.write(f'  --val-data {args.val_data} \\\n')
            if args.lem_train:
                job_file.write(f'  --lem-train {args.lem_train} \\\n')
            if args.lem_val:
                job_file.write(f'  --lem-val {args.lem_val} \\\n')
            if args.dict:
                job_file.write(f'  --dict {args.dict} \\\n')
            if args.freq:
                job_file.write(f'  --freq {args.freq} \\\n')
            job_file.write(f'  --sw-vocab {args.sw_vocab} \\\n')
            job_file.write(f'  --sw-model {args.sw_model} \\\n')
            job_file.write(f'  --model {args.model}/{job_name}.pt \\\n')
            job_file.write(f'  --log {args.model}/{job_name}.log \\\n')
            if args.seed:
                job_file.write(f'  --seed {args.seed} \\\n')
            # for option, value in params:
            #     job_file.write(f'  --{option} {value} \\\n')
            if 'dpe_' in args.model:
                job_file.write('  --dpe-embed 1 \\\n')

            for test_data in args.test_data:
                wmt_set = None
                if re.match(r'wmt[0-9]{2}', test_data):
                    wmt_set, test_data = test_data.split(':')
                test_set = test_data.split('/')[-1]
                job_file.write('\npython translation/translate.py  \\\n')
                if args.dict:
                    job_file.write(f'  --dict {args.dict} \\\n')
                if args.freq:
                    job_file.write(f'  --freq {args.freq} \\\n')
                job_file.write(f'  --sw-vocab {args.sw_vocab} \\\n')
                job_file.write(f'  --sw-model {args.sw_model} \\\n')
                job_file.write(f'  --model {args.model}/{job_name}.pt \\\n')
                job_file.write(f'  --input {test_data}.{src_lang} \\\n')
                job_file.write(f'  > {args.model}/{job_name}.{test_set}.hyp \n')

                if wmt_set:
                    job_file.write(f'\necho "\\n{test_data}\\n" >> {args.model}/{job_name}.log \n')
                    job_file.write(f'sacrebleu -t {wmt_set} -l {args.lang_pair} -w 4 \\\n')
                    job_file.write(f'  -i {args.model}/{job_name}.{test_set}.hyp \\\n')
                    job_file.write(f"  -m {' '.join(args.metric)} \\\n")
                    job_file.write(f'  >> {args.model}/{job_name}.log \n')
                else:
                    job_file.write(f'\necho "\\n{test_data}\\n" >> {args.model}/{job_name}.log \n')
                    job_file.write(f'sacrebleu {test_data}.{tgt_lang} -w 4 \\\n')
                    job_file.write(f'  -i {args.model}/{job_name}.{test_set}.hyp \\\n')
                    job_file.write(f"  -m {' '.join(args.metric)} \\\n")
                    job_file.write(f'  >> {args.model}/{job_name}.log \n')

        os.system(
            f"{qf_submit} --name {job_name} --deferred -- {email} -l gpu_card=1 {args.model}/{job_name}.sh"
        )
    os.system('qf check')


if __name__ == '__main__':
    import argparse

    main()
