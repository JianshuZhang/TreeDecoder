import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import DenseNet
from decoder import Gru_cond_layer, Gru_prob
import math

# create gru init state
class FcLayer(nn.Module):
    def __init__(self, nin, nout):
        super(FcLayer, self).__init__()
        self.fc = nn.Linear(nin, nout)

    def forward(self, x):
        out = torch.tanh(self.fc(x))
        return out


# Embedding
class My_Embedding(nn.Module):
    def __init__(self, params):
        super(My_Embedding, self).__init__()
        self.embedding = nn.Embedding(params['K'], params['m'])
        self.pos_embedding = torch.zeros(params['maxlen'], params['m']).cuda()
        nin = params['maxlen']
        nout = params['m']
        d_model = nout
        for pos in range(nin):
            for i in range(nout//2):
                self.pos_embedding[pos, 2*i] = math.sin(1.*pos/(10000**(2.*i/d_model)))
                self.pos_embedding[pos, 2*i+1] = math.cos(1.*pos/(10000**(2.*i/d_model)))
    def forward(self, params, ly, lp, ry):
        if ly.sum() < 0.:  # <bos>
            lemb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            lemb = self.embedding(ly)  # (seqs_y,batch,m)  |  (batch,m)
            if len(lemb.shape) == 3:  # only for training stage
                lemb_shifted = torch.zeros([lemb.shape[0], lemb.shape[1], params['m']], dtype=torch.float32).cuda()
                lemb_shifted[1:] = lemb[:-1]
                lemb = lemb_shifted

        if lp.sum() < 1.:  # pos=0
            Pemb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            Pemb = self.pos_embedding[lp]  # (seqs_y,batch,m)  |  (batch,m)
            if len(Pemb.shape) == 3:  # only for training stage
                Pemb_shifted = torch.zeros([Pemb.shape[0], Pemb.shape[1], params['m']], dtype=torch.float32).cuda()
                Pemb_shifted[1:] = Pemb[:-1]
                Pemb = Pemb_shifted

        if ry.sum() < 0.:  # <bos>
            remb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            remb = self.embedding(ry)  # (seqs_y,batch,m)  |  (batch,m)
            if len(remb.shape) == 3:  # only for training stage
                remb_shifted = torch.zeros([remb.shape[0], remb.shape[1], params['m']], dtype=torch.float32).cuda()
                remb_shifted[1:] = remb[1:]
                remb = remb_shifted
        return lemb, Pemb, remb
    def word_emb(self, params, y):
        if y.sum() < 0.:  # <bos>
            emb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            emb = self.embedding(y)  # (seqs_y,batch,m)  |  (batch,m)
        return emb
    def pos_emb(self, params, p):
        if p.sum() < 1.:  # <bos>
            Pemb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            Pemb = self.pos_embedding[p]  # (seqs_y,batch,m)  |  (batch,m)
        return Pemb

class Encoder_Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder, self).__init__()
        self.encoder = DenseNet(growthRate=params['growthRate'], reduction=params['reduction'],
                                bottleneck=params['bottleneck'], use_dropout=params['use_dropout'])
        self.init_GRU_model = FcLayer(params['D'], params['n'])
        self.emb_model = My_Embedding(params)
        self.gru_model = Gru_cond_layer(params)
        self.gru_prob_model = Gru_prob(params)
        self.fc_Uamem = nn.Linear(params['n'], params['dim_attention'])
        self.fc_Wamem = nn.Linear(params['n'], params['dim_attention'], bias=False)
        # self.conv_Qmem = nn.Conv2d(1, 512, kernel_size=(3,1), bias=False, padding=(1,0))
        # self.fc_Ufmem = nn.Linear(512, params['dim_attention'])
        self.fc_vamem = nn.Linear(params['dim_attention'], 1)
        self.criterion = torch.nn.CrossEntropyLoss(reduce=False)

    def forward(self, params, x, x_mask, ly, ly_mask, ry, ry_mask, re, re_mask, ma, ma_mask, lp, rp, one_step=False):
        # recover permute
        # ly = ly.permute(1, 0)
        # ly_mask = ly_mask.permute(1, 0)
        # ly = ly.permute(1, 0)
        # ly_mask = ly_mask.permute(1, 0)

        ma = ma.permute(2, 1, 0) # SeqY * Matt * batch
        ma_mask = ma_mask.permute(2, 1, 0)

        ctx, ctx_mask = self.encoder(x, x_mask)

        # init state
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) / ctx_mask.sum(2).sum(1)[:, None]  # (batch,D)
        init_state = self.init_GRU_model(ctx_mean)  # (batch,n)

        # two GRU layers
        lemb, Pemb, remb = self.emb_model(params, ly, lp, ry)  # (seqs_y,batch,m)
        # h2ts: (seqs_y,batch,n),  cts: (seqs_y,batch,D),  alphas: (seqs_y,batch,H,W)
        h2ts, h1ts, h01ts, ctCs, ctPs, cts, calphas, calpha_pasts, palphas, palpha_pasts= \
                                self.gru_model(params, lemb, remb, ly_mask, ctx, ctx_mask, init_state=init_state)

        word_pos_memory = torch.cat((init_state[None, :, :], h2ts[:-1]), 0)
        # word_pos_memory = lemb + Pemb
        mempctx_ = self.fc_Uamem(word_pos_memory)
        memquery = self.fc_Wamem(h01ts)
        memattention_score = torch.tanh(mempctx_[None, :, :, :] + memquery[:, None, :, :])
        memalpha = self.fc_vamem(memattention_score)
        memalpha = memalpha - memalpha.max()
        memalpha = memalpha.view(memalpha.shape[0], memalpha.shape[1], memalpha.shape[2]) # SeqY * Matt * batch
        memalpha = torch.exp(memalpha)
        memalpha = memalpha * ma_mask # SeqY * Matt * batch
        memalpha = memalpha / (memalpha.sum(1)[:, None, :] + 1e-10)
        memalphas = memalpha + 1e-10
        cost_memalphas = - ma.float() * torch.log(memalphas) * ma_mask
        loss_memalphas = cost_memalphas.sum((0,1))

        # compute KL alpha
        calpha_sort_ = torch.cat((torch.zeros(1, calphas.shape[1], calphas.shape[2], calphas.shape[3]).cuda(), calphas), 0)
        n_gaps = calpha_sort_.shape[0]
        n_batch = calpha_sort_.shape[1]
        n_H = calpha_sort_.shape[2]
        n_W = calpha_sort_.shape[3]
        rp = rp.permute(1,0) # batch * SeqY
        rp_shape = rp.shape
        rp = rp + n_gaps * torch.arange(n_batch)[:, None].cuda()

        calpha_sort = calpha_sort_.permute(1,0,2,3)
        calpha_sort = torch.reshape(calpha_sort, (calpha_sort.shape[0]*calpha_sort.shape[1],calpha_sort.shape[2],calpha_sort.shape[3]))
        calpha_sort = calpha_sort[rp.flatten()]
        calpha_sort = torch.reshape(calpha_sort, (rp_shape[0], rp_shape[1], n_H, n_W))
        calpha_sort = calpha_sort.permute(1,0,2,3)

        calpha_sort = calpha_sort + 1e-10
        palphas = palphas + 1e-10
        cost_KL_alpha = calpha_sort * (torch.log(calpha_sort)-torch.log(palphas)) * ctx_mask[None, :, :, :]
        loss_KL = cost_KL_alpha.sum((0,2,3))

        cscores, pscores, rescores = self.gru_prob_model(ctCs, ctPs, cts, h2ts, remb, use_dropout=params['use_dropout'])  # (seqs_y x batch,K)

        cscores = cscores.contiguous()
        cscores = cscores.view(-1, cscores.shape[2])
        ly = ly.contiguous()
        lpred_loss = self.criterion(cscores, ly.view(-1))  # (seqs_y x batch,)
        lpred_loss = lpred_loss.view(ly.shape[0], ly.shape[1])  # (seqs_y,batch)
        lpred_loss = (lpred_loss * ly_mask).sum(0) / (ly_mask.sum(0)+1e-10)
        lpred_loss = lpred_loss.mean()

        pscores = pscores.contiguous()
        pscores = pscores.view(-1, pscores.shape[2])
        ry = ry.contiguous()
        rpred_loss = self.criterion(pscores, ry.view(-1))
        rpred_loss = rpred_loss.view(ry.shape[0], ry.shape[1])
        rpred_loss = (rpred_loss * ry_mask).sum(0) / (ry_mask.sum(0)+1e-10)
        rpred_loss = rpred_loss.mean()

        rescores = rescores.contiguous()
        rescores = rescores.view(-1, rescores.shape[2])
        re = re.contiguous()
        repred_loss = self.criterion(rescores, re.view(-1))
        repred_loss = repred_loss.view(re.shape[0], re.shape[1])
        repred_loss = (repred_loss * re_mask).sum(0) / (re_mask.sum(0)+1e-10)
        repred_loss = repred_loss.mean()

        mem_loss = loss_memalphas / (ly_mask.sum(0)+1e-10)
        mem_loss = mem_loss.mean()

        KL_loss = loss_KL / (ly_mask.sum(0)+1e-10)
        KL_loss = KL_loss.mean()

        loss = params['ly_lambda']*lpred_loss + params['ry_lambda']*rpred_loss + \
            params['re_lambda']*repred_loss + params['rpos_lambda']*mem_loss + params['KL_lambda']*KL_loss

        return loss, lpred_loss, rpred_loss, repred_loss, mem_loss, KL_loss

    # decoding: encoder part
    def f_init(self, x, x_mask=None):
        if x_mask is None:  # x_mask is actually no use here
            shape = x.shape
            x_mask = torch.ones(shape).cuda()
        ctx, _ctx_mask = self.encoder(x, x_mask)
        ctx_mean = ctx.mean(dim=3).mean(dim=2)
        init_state = self.init_GRU_model(ctx_mean)  # (1,n)
        return init_state, ctx

    def f_next_parent(self, params, ly, lp, ctx, init_state, h1t, palpha_past, nextemb_memory, nextePmb_memory, initIdx):
        emb = self.emb_model.word_emb(params, ly)
        # Pemb = self.emb_model.pos_emb(params, lp)
        nextemb_memory[initIdx, :, :] = emb
        # ePmb_memory_ = emb + Pemb
        nextePmb_memory[initIdx, :, :] = init_state

        h01, ctP, palpha, next_palpha_past = self.gru_model.parent_forward(params, emb, context=ctx, init_state=init_state, palpha_past=palpha_past)

        mempctx_ = self.fc_Uamem(nextePmb_memory)
        memquery = self.fc_Wamem(h01)
        memattention_score = torch.tanh(mempctx_ + memquery[None, :, :])
        memalpha = self.fc_vamem(memattention_score)
        memalpha = memalpha - memalpha.max()
        memalpha = memalpha.view(memalpha.shape[0], memalpha.shape[1]) # Matt * batch
        memalpha = torch.exp(memalpha)
        mem_mask = torch.zeros(nextePmb_memory.shape[0], nextePmb_memory.shape[1]).cuda()
        mem_mask[:(initIdx+1), :] = 1
        memalpha = memalpha * mem_mask # Matt * batch
        memalpha = memalpha / (memalpha.sum(0) + 1e-10)

        Pmemalpha = memalpha.view(-1, memalpha.shape[1])
        Pmemalpha = Pmemalpha.permute(1, 0) # batch * Matt
        return h01, Pmemalpha, ctP, palpha, next_palpha_past, nextemb_memory, nextePmb_memory

    # decoding: decoder part
    def f_next_child(self, params, remb, ctP, ctx, init_state, calpha_past):

        next_state, h1t, ctC, ctP, ct, calpha, next_calpha_past = \
                self.gru_model.child_forward(params, remb, ctP, context=ctx, init_state=init_state, calpha_past=calpha_past)

        # reshape to suit GRU step code
        h2te = next_state.view(1, next_state.shape[0], next_state.shape[1])
        ctC = ctC.view(1, ctC.shape[0], ctC.shape[1])
        ctP = ctP.view(1, ctP.shape[0], ctP.shape[1])
        ct = ct.view(1, ct.shape[0], ct.shape[1])

        # calculate probabilities
        cscores, pscores, rescores = self.gru_prob_model(ctC, ctP, ct, h2te, remb, use_dropout=params['use_dropout'])
        cscores = cscores.view(-1, cscores.shape[2])
        next_lprobs = F.softmax(cscores, dim=1)
        rescores = rescores.view(-1, rescores.shape[2])
        next_reprobs = F.softmax(rescores, dim=1)
        next_re = torch.argmax(next_reprobs, dim=1)

        return next_lprobs, next_reprobs, next_state, h1t, calpha, next_calpha_past, next_re
