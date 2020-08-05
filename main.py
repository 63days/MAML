import torch
from sinusoid import Sinusoid
from meta import Meta
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    meta = Meta(inner_lr=args.inner_lr, outer_lr=args.outer_lr)
    meta.to(device)
    if not args.test:
        train_ds = Sinusoid(k_shot=args.k_shot, q_query=15, num_tasks=1000000)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)

        best_loss = float('inf')
        losses = []
        pbar = tqdm(range(args.epochs))
        for epoch in pbar:
            k_i, k_o, q_i, q_o = next(iter(train_dl))
            k_i, k_o, q_i, q_o = k_i.to(device), k_o.to(device), q_i.to(device), q_o.to(device)
            loss = meta(k_i, k_o, q_i, q_o)
            losses.append(loss)
            pbar.set_description('| loss: {:.5f} |'.format(loss))

            if best_loss > loss:
                best_loss = loss
                meta.save_weights()

        plt.plot(losses)
        plt.savefig('loss_final.png', dpi=300)


    else:
        print('testing... k_shot:', args.k_shot)
        meta.test(k=args.k_shot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML')
    parser.add_argument(
        '--epochs',
        type=int,
        default=70000
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10
    )
    parser.add_argument(
        '--k_shot',
        type=int,
        default=10
    )
    parser.add_argument(
        '--test',
        action='store_true'
    )
    parser.add_argument(
        '--inner_lr',
        type=float,
        default=1e-2
    )
    parser.add_argument(
        '--outer_lr',
        type=float,
        default=1e-3
    )
    args = parser.parse_args()

    main(args)