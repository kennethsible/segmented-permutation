import torch

from translation.decoder import greedy_search
from translation.manager import Manager


def tokenize(w: list[str], k: int) -> list[str]:
    return [s[i : i + k] for s in w for i in range(0, len(s), k)]


def translate(string: str, manager: Manager) -> str:
    model, vocab, device = manager.model, manager.vocab, manager.device
    # tokenizer = Tokenizer(manager.src_lang, manager.tgt_lang, manager.sw_model)
    N = 8 // manager.config['k']
    src_words = tokenize(string.split(), manager.config['k'])
    src_words = N * ['<BOS>'] + src_words + N * ['<EOS>']

    model.eval()
    with torch.no_grad():
        src_nums = torch.tensor(vocab.numberize(src_words), device=device)
        src_encs = model.encode(src_nums.unsqueeze(0))
        out_nums = greedy_search(manager, src_encs, manager.max_length * 2)

    return ' '.join(vocab.denumberize(out_nums.tolist()))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sw-vocab', metavar='FILE_PATH', required=True, help='subword vocab')
    # parser.add_argument('--sw-model', metavar='FILE_PATH', required=True, help='subword model')
    parser.add_argument('--model', metavar='FILE_PATH', required=True, help='translation model')
    parser.add_argument('--input', type=str, help='detokenized input')
    args, unknown = parser.parse_known_args()

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model_state = torch.load(args.model, map_location=device)

    config = model_state['config']
    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:].replace('-', '_'), unknown[i + 1]
            try:
                config[option] = (int if value.isdigit() else float)(value)
            except ValueError:
                config[option] = value

    manager = Manager(
        config,
        device,
        # model_state['src_lang'],
        # model_state['tgt_lang'],
        args.model,
        args.sw_vocab,
        # args.sw_model,
    )
    manager.model.load_state_dict(model_state['state_dict'])

    if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')

    print(translate(args.input, manager))


if __name__ == '__main__':
    main()
