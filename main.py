#!/usr/bin/env python -W ignore::DeprecationWarning
import argparse
import torch
from data.dataset import Pataka_Dataset
from train.train_vae import train_vae
import warnings ; warnings.warn = lambda *args,**kwargs: None


def main():
    args = get_arguments()
    SEED = args.seed
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    vae_dataset = Pataka_Dataset(DBs = ['Gita','Neurovoz'], train_size=0.8, mode = 'train', seed=SEED)
    vae_dataset = torch.utils.data.DataLoader(vae_dataset, batch_size=args.batch_size, shuffle=True)
    vae = train_vae(vae_dataset, x_dim=1, z_dim=32, epochs=args.nEpochs, lr=args.lr, device=device)
    torch.save({
            'epoch': args.nEpochs,
            'model_state_dict': vae.state_dict(),
            }, 'vae.pth')
    print('Model saved as vae.pth')
    return


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_training', action='store_true', default=False,
                        help='load saved_model as initial model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nEpochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', default=1e-7, type=float,
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()