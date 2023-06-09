import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import os
import pathlib
import numpy as np
import lpips
import net
from function import nor_mean_std, nor_mean

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, LCT, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = LCT(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = LCT(content_f, style_f)
    return decoder(feat)


def pytest(args):
    torch.cuda.set_device(1)
    do_interpolation = False
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Either --content or --contentDir should be given.
    assert (args.content or args.content_dir)
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    # Either --style or --styleDir should be given.
    assert (args.style or args.style_dir)
    if args.style:
        style_paths = args.style.split(',')
        if len(style_paths) == 1:
            style_paths = [Path(args.style)]
        else:
            do_interpolation = True
            assert (args.style_interpolation_weights != ''), \
                'Please specify interpolation weights'
            weights = [int(i) for i in args.style_interpolation_weights.split(',')]
            interpolation_weights = [w / sum(weights) for w in weights]
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    decoder = net.decoder
    vgg = net.vgg
    network = net.Net(vgg, decoder, args.testing_mode)
    LCT = network.LCT

    LCT.eval()
    decoder.eval()
    vgg.eval()

    # remove 'module.'
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    load_decoder = torch.load(args.decoder)
    for k, v in load_decoder.items():
        # namekey = k[7:] # remove `module.`
        # print(k)
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    load_vgg = torch.load(args.vgg)
    for k, v in load_vgg.items():
        # namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    vgg.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    load_LCT = torch.load(args.LCT)
    for k, v in load_LCT.items():
        # namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    LCT.load_state_dict(new_state_dict)

    vgg = nn.Sequential(*list(vgg.children())[:18])
    decoder = nn.Sequential(*list(net.decoder.children())[10:])

    vgg.to(device)
    decoder.to(device)
    LCT.to(device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    for content_path in tqdm(content_paths):
        if do_interpolation:  # one content image, N style image
            style = torch.stack([style_tf(Image.open(str(p)).convert('RGB')) for p in style_paths])
            content = content_tf(Image.open(str(content_path)).convert('RGB')) \
                .unsqueeze(0).expand_as(style)
            style = style.to(device)
            content = content.to(device)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, LCT, content, style,
                                        args.alpha, interpolation_weights)
            output = output.cpu()
            output_name = output_dir / '{:s}_interpolation{:s}'.format(
                content_path.stem, args.save_ext)
            save_image(output, str(output_name))

        else:  # process one content and one style
            for style_path in style_paths:
                # print(content_path, style_path)
                content = content_tf(Image.open(str(content_path)).convert('RGB'))
                style = style_tf(Image.open(str(style_path)).convert('RGB'))
                if args.preserve_color:
                    style = coral(style, content)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, LCT, content, style,
                                            args.alpha)
                output = output.cpu()

                output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                    content_path.stem, style_path.stem, args.save_ext)
                save_image(output, str(output_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str,
                        help='File path to the content image')
    parser.add_argument('--content_dir', type=str,default='',
                        help='Directory path to a batch of content images')
    parser.add_argument('--testing_mode', default='Photo-realistic')
    parser.add_argument('--style', type=str,
                        help='File path to the style image, or multiple style \
                        images separated by commas if you want to do style \
                        interpolation or spatial control')
    parser.add_argument('--style_dir', type=str,default='',
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='')
    parser.add_argument('--LCT', type=str, default='')

    # Additional options
    parser.add_argument('--content_size', type=int, default=512,
                        help='New (minimum) size for the content image, \
                        keeping the original size if set to 0')
    parser.add_argument('--style_size', type=int, default=512,
                        help='New (minimum) size for the style image, \
                        keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output/',
                        help='Directory to save the output image(s)')

    # Advanced options
    parser.add_argument('--preserve_color', action='store_true',
                        help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The weight that controls the degree of \
                                 stylization. Should be between 0 and 1')
    parser.add_argument(
        '--style_interpolation_weights', type=str,
        help='The weight for blending the style of multiple style images')



    args = parser.parse_args()
    pytest(args)


