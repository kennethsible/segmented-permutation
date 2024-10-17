import os
import re
from argparse import ArgumentParser, Namespace

QUEUES = [
    'gpu@@nlp-a10',
    'gpu@@nlp-gpu',
    'gpu@@csecri',
    'gpu@@crc_gpu',
]


def generate_header(job_name: str, args: Namespace) -> str:
    string = '#!/bin/bash\n\n'
    string += f'touch {args.output_dir}/{job_name}.log\n'
    string += f'fsync -d 30 {args.output_dir}/{job_name}.log &\n'
    string += f'\nconda activate {args.conda}\n'
    string += 'export PYTHONPATH="${PYTHONPATH}:${pwd}"\n'
    string += 'export SACREBLEU_FORMAT=text\n'
    return string


def generate_main(job_name: str, args: Namespace, unknown: list[str]) -> str:
    string = 'python translation/main.py  \\\n'
    # string += f'  --lang-pair {args.lang_pair} \\\n'
    string += f'  --train-data {args.train_data} \\\n'
    string += f'  --val-data {args.val_data} \\\n'
    string += f'  --sw-vocab {args.sw_vocab} \\\n'
    # string += f'  --sw-model {args.sw_model} \\\n'
    string += f'  --model {args.output_dir}/{job_name}.pt \\\n'
    string += f'  --log {args.output_dir}/{job_name}.log \\\n'
    if args.seed:
        string += f'  --seed {args.seed} \\\n'
    for i, arg in enumerate(unknown):
        if arg.startswith('--') and len(unknown) > i:
            string += f'  {arg} {unknown[i + 1]} \\\n'
    return string


def generate_translate(job_name: str, test_data: str, args: Namespace) -> str:
    src_lang, _ = args.lang_pair.split('-')
    if re.match(r'wmt[0-9]{2}', test_data):
        _, test_data = test_data.split(':')
    test_set = test_data.split('/')[-1]
    string = 'python translation/translate.py  \\\n'
    string += f'  --sw-vocab {args.sw_vocab} \\\n'
    string += f'  --sw-model {args.sw_model} \\\n'
    string += f'  --model {args.output_dir}/{job_name}.pt \\\n'
    string += f'  --input {test_data}.{src_lang} \\\n'
    string += f'  > {args.output_dir}/{job_name}.{test_set}.hyp \n'
    return string


def generate_sacrebleu(job_name: str, test_data: str, args: Namespace) -> str:
    _, tgt_lang = args.lang_pair.split('-')
    wmt_set = ''
    if re.match(r'wmt[0-9]{2}', test_data):
        wmt_set, test_data = test_data.split(':')
    test_set = test_data.split('/')[-1]
    string = ''
    if wmt_set:
        string += f'echo "\\n{test_data}\\n" >> {args.output_dir}/{job_name}.log \n'
        string += f'sacrebleu -t {wmt_set} -l {args.lang_pair} -w 4 \\\n'
        string += f'  -i {args.output_dir}/{job_name}.{test_set}.hyp \\\n'
        string += f"  -m {' '.join(args.metric)} \\\n"
        string += f'  >> {args.output_dir}/{job_name}.log \n'
    else:
        string += f'echo "\\n{test_data}\\n" >> {args.output_dir}/{job_name}.log \n'
        string += f'sacrebleu {test_data}.{tgt_lang} -w 4 \\\n'
        string += f'  -i {args.output_dir}/{job_name}.{test_set}.hyp \\\n'
        string += f"  -m {' '.join(args.metric)} \\\n"
        string += f'  >> {args.output_dir}/{job_name}.log \n'
    return string


def generate_job_script(job_name: str, args: Namespace, unknown: list[str]) -> str:
    string = generate_header(job_name, args)
    string += '\n' + generate_main(job_name, args, unknown)
    # for test_data in args.test_data:
    #     string += '\n' + generate_translate(job_name, test_data, args)
    #     string += '\n' + generate_sacrebleu(job_name, test_data, args)
    return string


def qf_submit(job_name: str, args: Namespace) -> str:
    string = 'qf submit --queue ' + ' --queue '.join(QUEUES)
    string += f' --name {job_name} --deferred --'
    if args.email:
        string += f' -M {args.email} -m abe'
    string += f' -l gpu_card=1 {args.output_dir}/{job_name}.sh'
    return string


def main():
    parser = ArgumentParser()
    # parser.add_argument('--lang-pair', required=True, help='language pair')
    parser.add_argument('--train-data', metavar='FILE_PATH', required=True, help='training data')
    parser.add_argument('--val-data', metavar='FILE_PATH', required=True, help='validation data')
    parser.add_argument('--sw-vocab', metavar='FILE_PATH', required=True, help='subword vocab')
    # parser.add_argument('--sw-model', metavar='FILE_PATH', required=True, help='subword model')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--model', required=True, help='translation model')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--conda', metavar='ENV', required=True, help='conda environment')
    parser.add_argument('--email', required=True, help='email address')
    # parser.add_argument(
    #     '--test-data', nargs='+', metavar='FILE_PATH', required=True, help='detokenized test data'
    # )
    # parser.add_argument('--metric', nargs='+', required=True, help='evaluation metric')
    args, unknown = parser.parse_known_args()

    os.system(f'mkdir -p {args.output_dir}')
    job_name = args.model
    with open(f'{args.output_dir}/{job_name}.sh', 'w') as job_file:
        job_file.write(generate_job_script(job_name, args, unknown))
    os.system(qf_submit(job_name, args))
    os.system('qf check')


if __name__ == '__main__':
    main()
