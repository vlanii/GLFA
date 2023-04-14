import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from function import calc_mean_std, nor_mean_std, nor_mean, calc_cov
import random
from moNCE_sinkhorn import OT
from packaging import version


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

mlp = nn.ModuleList([nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)]) 

class LCT(nn.Module):
    def __init__(self, training_mode='art'):
        super(LCT, self).__init__()
        self.cnet = nn.Sequential(nn.Conv2d(256,128,1,1,0),nn.ReLU(inplace=True),nn.Conv2d(128,32,1,1,0))
        self.snet = nn.Sequential(nn.Conv2d(256,128,3,1,0),nn.ReLU(inplace=True),nn.Conv2d(128,32,1,1,0))
        self.uncompress = nn.Conv2d(32,256,1,1,0)

    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    def forward(self, content, style):
        cF_nor = nor_mean_std(content)
        sF_nor, smean = nor_mean(style)
        cF = self.cnet(cF_nor)
        sF = self.snet(sF_nor)
        b, c, w, h = cF.size()
        s_cov = calc_cov(sF)
        gF = torch.bmm(s_cov, cF.flatten(2, 3)).view(b,c,w,h)
        gF = self.uncompress(gF)
        gF = gF + smean.expand(cF_nor.size())

        # skip connection
        gFadain = self.adaptive_instance_normalization(content, style)
        gF = (gF + gFadain) / 2.0

        return gF                                             

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class CCPL(nn.Module):
    def __init__(self, mlp):
        super(CCPL, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp

    def NeighborSample(self, feat, layer, num_s, sample_ids=None):
        b, c, h, w = feat.size()   
        if sample_ids is None:    
            sample_ids = []
            while len(sample_ids) < num_s:
                h_id = random.randint(0, h-3) # upper left corner
                w_id = random.randint(0, w-3)
                if [h_id, w_id] not in sample_ids:
                    sample_ids += [[h_id, w_id]]
            sample_ids = torch.tensor(sample_ids)        
        h_ids = sample_ids[:,0]
        w_ids = sample_ids[:,1]              
        ft = torch.ones((b,c,8*num_s)).to(feat.device) # b, c, 32 
        for i in range(num_s):
            f_c = feat[:,:,h_ids[i]+1,w_ids[i]+1].view(b,c,1) # centor

            f = feat[:,:,h_ids[i]:h_ids[i]+3,w_ids[i]:w_ids[i]+3].flatten(2, 3) - f_c

            ft[:,:,8*i:8*i+8] = torch.cat([f[:,:,:4], f[:,:,5:]], 2)
            f_nine = feat[:,:,h_ids[i]:h_ids[i]+3,w_ids[i]:w_ids[i]+3].flatten(2, 3)
        ft = ft.permute(0,2,1) 
        for i in range(3):
            ft = self.mlp[3*layer+i](ft)
        ft = Normalize(2)(ft.permute(0,2,1))

        f_nine = Normalize(2)(f_nine)

        return ft, sample_ids, f_nine

    def MoNCELoss(self, feat_q, feat_k, tau=0.07):

        batchSize = feat_q.shape[0]    
        dim = feat_q.shape[1]   
        gpu_ids = [2]
        nce_includes_all_negatives_from_minibatch = False

        if nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batchSize // len(gpu_ids)

        ot_q = feat_q.contiguous().view(batch_dim_for_bmm, -1, dim)   
        ot_k = feat_k.contiguous().view(batch_dim_for_bmm, -1, dim).detach()   
        f = OT(ot_q, ot_k, eps=1, max_iter=50, cost_type='hard')   
        f = f.permute(0, 2, 1) * (9 - 1) * 1.0 + 1e-8

        feat_k = feat_k.detach()
        l_pos = (feat_k * feat_q).sum(dim=1)[:, :, None]   

        feat_q = feat_q.contiguous().view(batch_dim_for_bmm, -1, dim)   
        feat_k = feat_k.contiguous().view(batch_dim_for_bmm, -1, dim)   
        npatches = feat_q.size(1)  
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))   
        # if self.loss_type == 'MoNCE':
        l_neg_curbatch = l_neg_curbatch + torch.log(f) * tau   

        mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch

        out = torch.cat((l_pos, l_neg), dim=2) / tau
        predictions = out.flatten(0, 1)

        target = torch.zeros(batchSize * feat_q.shape[1], dtype=torch.long).to(feat_q.device)
        loss = self.cross_entropy_loss(predictions, target)
        return loss

    def kl_divergence(self, feat_s, feat_t, eps=1e-6):


        feat_s = feat_s.permute(0, 2, 1)   
        feat_t = feat_t.permute(0, 2, 1)
        b, c, num_s = feat_s.shape
        feat_s = feat_s.contiguous().view(b * c, num_s).contiguous()
        feat_t = feat_t.contiguous().view(b * c, num_s).contiguous()
        feat_s_norm = torch.sqrt(torch.sum(feat_s ** 2, dim=1, keepdim=True))
        feat_s = feat_s / (feat_s_norm + eps)
        feat_s[feat_s != feat_s] = 0

        feat_t_norm = torch.sqrt(torch.sum(feat_t ** 2, dim=1, keepdim=True))
        feat_t = feat_t / (feat_t_norm + eps)
        feat_t[feat_t != feat_t] = 0

        # Calculate the cosine similarity
        feat_s_cos_sim = torch.mm(feat_s, feat_s.transpose(0, 1))
        feat_t_cos_sim = torch.mm(feat_t, feat_t.transpose(0, 1))

        # Scale cosine similarity to [0,1]
        feat_s_cos_sim = (feat_s_cos_sim + 1.0) / 2.0
        feat_t_cos_sim = (feat_t_cos_sim + 1.0) / 2.0

        # Transform them into probabilities
        feat_s_cond_prob = feat_s_cos_sim / torch.sum(
            feat_s_cos_sim, dim=1, keepdim=True
        )
        feat_t_cond_prob = feat_t_cos_sim / torch.sum(
            feat_t_cos_sim, dim=1, keepdim=True
        )

        # Calculate the KL-divergence
        divergence = torch.mean(
            feat_t_cond_prob
            * torch.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps))
        )
        return divergence

    def forward(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.07):
        loss_ccp = 0.0
        for i in range(start_layer, end_layer):
            f_q, sample_ids, f_q_nine = self.NeighborSample(feats_q[i], i, num_s)
            f_k, _, f_k_nine = self.NeighborSample(feats_k[i], i, num_s, sample_ids)
            loss_ccp += self.MoNCELoss(f_q, f_k, tau)
            kl_divergence = self.kl_divergence(f_q_nine, f_k_nine)
            loss_ccp += kl_divergence
        return loss_ccp    

class Net(nn.Module):
    def __init__(self, encoder, decoder, training_mode='art'):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.LCT = LCT(training_mode)
        self.mlp = mlp if training_mode == 'art' else mlp[:9]
        
        self.CCPL = CCPL(self.mlp)
        self.mse_loss = nn.MSELoss()
        self.end_layer = 4 if training_mode == 'art' else 3
        self.mode = training_mode

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(self.end_layer):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def feature_compress(self, feat):
        feat = feat.flatten(2,3)
        feat = self.mlp(feat)
        feat = feat.flatten(1,2)
        feat = Normalize(2)(feat)
        return feat      

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, tau, num_s, num_layer):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)

        gF = self.LCT(content_feats[-1], style_feats[-1])

        gimage = self.decoder(gF)
        g_t_feats = self.encode_with_intermediate(gimage)

        end_layer = self.end_layer
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feats[-1]) 
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0]) 
        for i in range(1, end_layer):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i]) 

        start_layer = end_layer - num_layer
        loss_ccp = self.CCPL(g_t_feats, content_feats, num_s, start_layer, end_layer)

        return loss_c, loss_s, loss_ccp
