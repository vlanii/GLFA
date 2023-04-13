import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image,make_grid
from tqdm import tqdm
import itertools
from torchinfo import summary
from torchvision.utils import save_image

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0'

import net as net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True   # 加�?Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def style_transfer(vgg, decoder, LCT, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = LCT(content_f, style_f)
    return decoder(feat)
def sample_image(vgg, decoder, LCT, content_images, style_images, output_file, iter):
    batch_size = content_images.shape[0]
    decoder.eval()
    LCT.eval()

    temp_img = style_transfer(vgg, decoder, LCT, content_images, style_images, alpha=1.0)

    cont = make_grid(content_images, nrow=batch_size, normalize=True)
    style = make_grid(style_images, nrow=batch_size, normalize=True)
    out = make_grid(temp_img, nrow=batch_size, normalize=True)
    image_grid = torch.cat((cont, style, out), 1)

    save_image(image_grid, output_file + 'output'+str(iter)+'.jpg', normalize=False)
    decoder.train()
    LCT.train()
    return


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    # device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    output_file = Path(args.temp_image_save_dir)
    output_file.mkdir(exist_ok=True, parents=True)
    output_file_name = args.temp_image_save_dir
    writer = SummaryWriter(log_dir=str(log_dir))

    decoder = net.decoder if args.training_mode == 'art' else nn.Sequential(*list(net.decoder.children())[10:])
    vgg = net.vgg

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31]) if args.training_mode == 'art' else nn.Sequential(*list(vgg.children())[:18])
    network = net.Net(vgg, decoder, args.training_mode)
    network.train()
    network.to(device)
    

    network = nn.DataParallel(network, device_ids=[0])
    # print(network)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    optimizer = torch.optim.Adam(itertools.chain(network.module.decoder.parameters(), network.module.LCT.parameters(), network.module.mlp.parameters()), lr=args.lr)
    

    for i in tqdm(range(args.max_iter)):
        adjust_learning_rate(optimizer, iteration_count=i)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)

        loss_c, loss_s, loss_ccp = network(content_images, style_images, args.tau, args.num_s, args.num_l)

        loss_c = args.content_weight * torch.mean(loss_c)
        loss_s = args.style_weight * torch.mean(loss_s)
        loss_ccp = args.ccp_weight * torch.mean(loss_ccp)

        loss = loss_c + loss_s + loss_ccp


        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)
        writer.add_scalar('loss_ccp', loss_ccp.item(), i + 1)


        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'decoder_iter_{:d}.pth.tar'.format(i + 1))
            #save temp image
            sample_image(vgg, decoder, network.module.LCT, content_images, style_images, output_file=output_file_name,
                         iter=i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = network.module.LCT.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'lct_iter_{:d}.pth.tar'.format(i + 1))


    writer.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default='',
                        help='Directory path to COCO2014 data-set')
    parser.add_argument('--style_dir', type=str, default='',
                        help='Directory path to Wikiart data-set')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

    # training options
    parser.add_argument('--training_mode', default='Photo-realistic')
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--temp_image_save_dir', default='./temp_images/',
                        help='Directory to save the model')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--ccp_weight', type=float, default=5.0)
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--num_s', type=int, default=8, help='number of sampled anchor vectors')
    parser.add_argument('--num_l', type=int, default=3, help='number of layers to calculate CCPL')
    args = parser.parse_args()

    train(args)


