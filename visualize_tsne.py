# -*- coding:utf-8 -*-
"""
作者：张少杰
日期：2022年03月12日
"""
import typing
import io
import os
import argparse
import logging
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from models.modeling import VisionTransformer, CONFIGS, AdversarialNetwork
from data.data_list_image import Normalize, ImageList
from sklearn.manifold import TSNE
from utils.transform import get_transform

logger = logging.getLogger(__name__)


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result


def visualize_tsne(args):
    # Prepare Model
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, num_classes=args.num_classes,
                              zero_head=False, img_size=args.img_size, vis=True)

    model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
    model.load_state_dict(torch.load(model_checkpoint))
    model.to(args.device)
    model.eval()

    ad_net = AdversarialNetwork(config.hidden_size // 12, config.hidden_size // 12)
    ad_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv.bin" % args.name)
    ad_net.load_state_dict(torch.load(ad_checkpoint))
    ad_net.to(args.device)
    ad_net.eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        Normalize(meanfile='./data/ilsvrc_2012_mean.npy')
    ])

    # 初始化一个t-sne模型
    tsne = TSNE(n_components=2, init="pca", learning_rate='auto', perplexity=30)

    # Prepare dataset
    transform_source, transform_target, _ = get_transform(args.dataset, args.img_size)
    source_loader = torch.utils.data.DataLoader(
        ImageList(open(args.source_list).readlines(), transform=transform_source, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    target_loader = torch.utils.data.DataLoader(
        ImageList(open(args.target_list).readlines(), transform=transform_target, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    len_source = len(source_loader)
    len_target = len(target_loader)

    # 整个数据集遍历过一遍
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)

    x_s_bank = np.ones((1,config.hidden_size))
    y_s_bank = np.ones((1,))
    x_t_bank = np.ones((1,config.hidden_size))
    y_t_bank = np.ones((1,))

    for data_source in iter_source:
        x_s, y_s = tuple(t.to(args.device) for t in data_source)
        _, _, _, embedding_s = model(x_s, ad_net=ad_net)
        y_s = y_s.cpu().detach().numpy()
        embedding_s_cls = embedding_s[:, 0, :].squeeze(1).cpu().detach().numpy()
        x_s_bank = np.vstack((x_s_bank, embedding_s_cls))
        y_s_bank = np.hstack((y_s_bank, y_s))
    print(x_s_bank.shape, y_s_bank.shape)
    embedding_tsne_s = tsne.fit_transform(x_s_bank[1:, :])
    plt.scatter(embedding_tsne_s[:, 0], embedding_tsne_s[:, 1], c='r')

    for data_target in iter_target:
        x_t, y_t = tuple(t.to(args.device) for t in data_target)
        _, _, _, embedding_t = model(x_t, ad_net=ad_net)
        y_t = y_t.cpu().detach().numpy()
        embedding_t_cls = embedding_t[:, 0, :].squeeze(1).cpu().detach().numpy()
        x_t_bank = np.vstack((x_t_bank, embedding_t_cls))
        y_t_bank = np.hstack((y_t_bank, y_t))
    embedding_tsne_t = tsne.fit_transform(x_t_bank[1:, :])
    plt.scatter(embedding_tsne_t[:, 0], embedding_tsne_t[:, 1], c='b')

    save_name = "t-SNE" + args.method + ".jpg"
    save_path = os.path.join(args.save_dir, args.dataset, args.name)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default="svhn2mnist",
                        help="Which downstream task.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--num_classes", default=10, type=int,
                        help="Number of classes in the dataset.")
    parser.add_argument("--source_list", help="Path of the source image.")
    parser.add_argument("--target_list", help="Path of the target image.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--save_dir", default="attention_visual", type=str,
                        help="The directory where attention maps will be saved.")
    parser.add_argument("--method", type=str, default="source_only",
                        help=" the method of training the model")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    visualize_tsne(args)


if __name__ == "__main__":
    main()

