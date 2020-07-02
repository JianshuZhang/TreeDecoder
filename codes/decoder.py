import torch
import torch.nn as nn


# two layers of GRU
class Gru_cond_layer(nn.Module):
    def __init__(self, params):
        super(Gru_cond_layer, self).__init__()
        self.fc_Wyz0 = nn.Linear(params['m'], params['n'])
        self.fc_Wyr0 = nn.Linear(params['m'], params['n'])
        self.fc_Wyh0 = nn.Linear(params['m'], params['n'])
        self.fc_Uhz0 = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhr0 = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhh0 = nn.Linear(params['n'], params['n'], bias=False)

        # attention for parent symbol
        self.conv_UaP = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.fc_WaP = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.conv_QP = nn.Conv2d(1, 512, kernel_size=11, bias=False, padding=5)
        self.fc_UfP = nn.Linear(512, params['dim_attention'])
        self.fc_vaP = nn.Linear(params['dim_attention'], 1)

        # attention for memory
        # self.fc_Uamem = nn.Linear(params['m'], params['dim_attention'])
        # self.fc_Wamem = nn.Linear(params['D'], params['dim_attention'], bias=False)
        # # self.conv_Qmem = nn.Conv2d(1, 512, kernel_size=(3,1), bias=False, padding=(1,0))
        # # self.fc_Ufmem = nn.Linear(512, params['dim_attention'])
        # self.fc_vamem = nn.Linear(params['dim_attention'], 1)

        self.fc_Wyz1 = nn.Linear(params['D'], params['n'])
        self.fc_Wyr1 = nn.Linear(params['D'], params['n'])
        self.fc_Wyh1 = nn.Linear(params['D'], params['n'])
        self.fc_Uhz1 = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhr1 = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhh1 = nn.Linear(params['n'], params['n'], bias=False)

        # the first GRU layer
        self.fc_Wyz = nn.Linear(params['m'], params['n'])
        self.fc_Wyr = nn.Linear(params['m'], params['n'])
        self.fc_Wyh = nn.Linear(params['m'], params['n'])

        self.fc_Uhz = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhr = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhh = nn.Linear(params['n'], params['n'], bias=False)

        # attention for child symbol
        self.conv_UaC = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.fc_WaC = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.conv_QC = nn.Conv2d(1, 512, kernel_size=11, bias=False, padding=5)
        self.fc_UfC = nn.Linear(512, params['dim_attention'])
        self.fc_vaC = nn.Linear(params['dim_attention'], 1)

        # the second GRU layer
        self.fc_Wcz = nn.Linear(params['D'], params['n'], bias=False)
        self.fc_Wcr = nn.Linear(params['D'], params['n'], bias=False)
        self.fc_Wch = nn.Linear(params['D'], params['n'], bias=False)

        self.fc_Uhz2 = nn.Linear(params['n'], params['n'])
        self.fc_Uhr2 = nn.Linear(params['n'], params['n'])
        self.fc_Uhh2 = nn.Linear(params['n'], params['n'])

    def forward(self, params, lembedding, rembedding, ly_mask=None, 
        context=None, context_mask=None, init_state=None):
        
        n_steps = lembedding.shape[0]  # seqs_y
        n_samples = lembedding.shape[1]  # batch

        pctx_ = self.conv_UaC(context)  # (batch,n',H,W)
        pctx_ = pctx_.permute(2, 3, 0, 1)  # (H,W,batch,n')
        repctx_ = self.conv_UaP(context)  # (batch,n',H,W)
        repctx_ = repctx_.permute(2, 3, 0, 1)  # (H,W,batch,n')
        state_below_lz = self.fc_Wyz0(lembedding)
        state_below_lr = self.fc_Wyr0(lembedding)
        state_below_lh = self.fc_Wyh0(lembedding)
        state_below_z = self.fc_Wyz(rembedding)
        state_below_r = self.fc_Wyr(rembedding)
        state_below_h = self.fc_Wyh(rembedding)
            
        calpha_past = torch.zeros(n_samples, context.shape[2], context.shape[3]).cuda()  # (batch,H,W)
        palpha_past = torch.zeros(n_samples, context.shape[2], context.shape[3]).cuda()
        h2t = init_state
        h2ts = torch.zeros(n_steps, n_samples, params['n']).cuda()
        h1ts = torch.zeros(n_steps, n_samples, params['n']).cuda()
        h01ts = torch.zeros(n_steps, n_samples, params['n']).cuda()
        ctCs = torch.zeros(n_steps, n_samples, params['D']).cuda()
        ctPs = torch.zeros(n_steps, n_samples, params['D']).cuda()
        cts = torch.zeros(n_steps, n_samples, 2*params['D']).cuda()
        calphas = (torch.zeros(n_steps, n_samples, context.shape[2], context.shape[3])).cuda()
        calpha_pasts = torch.zeros(n_steps, n_samples, context.shape[2], context.shape[3]).cuda()
        palphas = (torch.zeros(n_steps, n_samples, context.shape[2], context.shape[3])).cuda()
        palpha_pasts = torch.zeros(n_steps, n_samples, context.shape[2], context.shape[3]).cuda()
            
        for i in range(n_steps):
            h2t, h1t, h01t, ctC, ctP, ct, calpha, calpha_past, palpha, palpha_past = self._step_slice(ly_mask[i], 
                                                            context_mask, h2t, palpha_past, calpha_past, 
                                                            pctx_, repctx_, context, state_below_lz[i],
                                                            state_below_lr[i], state_below_lh[i], state_below_z[i],
                                                            state_below_r[i], state_below_h[i])

            h2ts[i] = h2t  # (seqs_y,batch,n)
            h1ts[i] = h1t
            h01ts[i] = h01t
            ctCs[i] = ctC
            ctPs[i] = ctP
            cts[i] = ct  # (seqs_y,batch,D)
            calphas[i] = calpha  # (seqs_y,batch,H,W)
            calpha_pasts[i] = calpha_past  # (seqs_y,batch,H,W)
            palphas[i] = palpha
            palpha_pasts[i] = palpha_past
        return h2ts, h1ts, h01ts, ctCs, ctPs, cts, calphas, calpha_pasts, palphas, palpha_pasts

    def parent_forward(self, params, lembedding, ly_mask=None, context=None, context_mask=None, init_state=None, palpha_past=None):
        state_below_lz = self.fc_Wyz0(lembedding)
        state_below_lr = self.fc_Wyr0(lembedding)
        state_below_lh = self.fc_Wyh0(lembedding)
        repctx_ = self.conv_UaP(context)  # (batch,n',H,W)
        repctx_ = repctx_.permute(2, 3, 0, 1)  # (H,W,batch,n')
        # if ly_mask is None:
        #     ly_mask = torch.ones(embedding.shape[0]).cuda()
        if ly_mask is None:
            ly_mask = torch.ones(lembedding.shape[0]).cuda()
        h0s, ctPs, palphas, palpha_pasts = self._step_slice_parent(ly_mask, context_mask, init_state, palpha_past, 
                                                            repctx_, context, state_below_lz, state_below_lr, state_below_lh)                                                   
        return h0s, ctPs, palphas, palpha_pasts

    def child_forward(self, params, rembedding, ctxP, ly_mask=None, 
        context=None, context_mask=None, init_state=None, calpha_past=None):

        pctx_ = self.conv_UaC(context)  # (batch,n',H,W)
        pctx_ = pctx_.permute(2, 3, 0, 1)  # (H,W,batch,n')

        state_below_z = self.fc_Wyz(rembedding)
        state_below_r = self.fc_Wyr(rembedding)
        state_below_h = self.fc_Wyh(rembedding)

        if ly_mask is None:
            ly_mask = torch.ones(rembedding.shape[0]).cuda()
        h2ts, h1ts, ctCs, ctPs, cts, calphas, calpha_pasts = \
                                self._step_slice_child(ly_mask, context_mask, init_state, calpha_past, 
                                    pctx_, context, state_below_z, state_below_r, state_below_h, ctxP)
        return h2ts, h1ts, ctCs, ctPs, cts, calphas, calpha_pasts

    # one step of two GRU layers
    def _step_slice(self, ly_mask, ctx_mask, h_, palpha_past_, calpha_past_, 
        pctx_, repctx_, cc_, state_below_lz, state_below_lr, state_below_lh, state_below_z, state_below_r, state_below_h):
        
        z0 = torch.sigmoid(self.fc_Uhz0(h_) + state_below_lz)  # (batch,n)
        r0 = torch.sigmoid(self.fc_Uhr0(h_) + state_below_lr)  # (batch,n)
        h0_p = torch.tanh(self.fc_Uhh0(h_) * r0 + state_below_lh)  # (batch,n)
        h0 = z0 * h_ + (1. - z0) * h0_p  # (batch,n)
        h0 = ly_mask[:, None] * h0 + (1. - ly_mask)[:, None] * h_

        # attention for parent symbol
        query_parent = self.fc_WaP(h0)
        palpha_past__ = palpha_past_[:, None, :, :]
        cover_FP = self.conv_QP(palpha_past__).permute(2, 3, 0, 1)  # (H,W,batch,n')
        pcover_vector = self.fc_UfP(cover_FP)
        pattention_score = torch.tanh(repctx_ + query_parent[None, None, :, :] + pcover_vector)
        palpha = self.fc_vaP(pattention_score)
        palpha = palpha - palpha.max()
        palpha = palpha.view(palpha.shape[0], palpha.shape[1], palpha.shape[2])
        palpha = torch.exp(palpha)
        if (ctx_mask is not None):
            palpha = palpha * ctx_mask.permute(1, 2, 0)
        palpha = palpha / (palpha.sum(1).sum(0)[None, None, :] + 1e-10)  # (H,W,batch)
        palpha_past = palpha_past_ + palpha.permute(2, 0, 1)  # (batch,H,W)
        ctP = (cc_ * palpha.permute(2, 0, 1)[:, None, :, :]).sum(3).sum(2)

        z01 = torch.sigmoid(self.fc_Uhz1(h0) + self.fc_Wyz1(ctP))  # zt  (batch,n)
        r01 = torch.sigmoid(self.fc_Uhr1(h0) + self.fc_Wyr1(ctP))  # rt  (batch,n)
        h01_p = torch.tanh(self.fc_Uhh1(h0) * r01 + self.fc_Wyh1(ctP))  # (batch,n)
        h01 = z01 * h0 + (1. - z01) * h01_p  # (batch,n)
        h01 = ly_mask[:, None] * h01 + (1. - ly_mask)[:, None] * h0
        # the first GRU layer
        z1 = torch.sigmoid(self.fc_Uhz(h01) + state_below_z)  # (batch,n)
        r1 = torch.sigmoid(self.fc_Uhr(h01) + state_below_r)  # (batch,n)
        h1_p = torch.tanh(self.fc_Uhh(h01) * r1 + state_below_h)  # (batch,n)
        h1 = z1 * h01 + (1. - z1) * h1_p  # (batch,n)
        h1 = ly_mask[:, None] * h1 + (1. - ly_mask)[:, None] * h01

        # attention for child symbol
        query_child = self.fc_WaC(h1)
        calpha_past__ = calpha_past_[:, None, :, :]  # (batch,1,H,W)
        cover_FC = self.conv_QC(calpha_past__).permute(2, 3, 0, 1)  # (H,W,batch,n')
        ccover_vector = self.fc_UfC(cover_FC)  # (H,W,batch,n')
        cattention_score = torch.tanh(pctx_ + query_child[None, None, :, :] + ccover_vector)  # (H,W,batch,n')
        calpha = self.fc_vaC(cattention_score)  # (H,W,batch,1)
        calpha = calpha - calpha.max()
        calpha = calpha.view(calpha.shape[0], calpha.shape[1], calpha.shape[2])  # (H,W,batch)
        calpha = torch.exp(calpha)  # exp
        if (ctx_mask is not None):
            calpha = calpha * ctx_mask.permute(1, 2, 0)
        calpha = (calpha / calpha.sum(1).sum(0)[None, None, :] + 1e-10)  # (H,W,batch)
        calpha_past = calpha_past_ + calpha.permute(2, 0, 1)  # (batch,H,W)
        ctC = (cc_ * calpha.permute(2, 0, 1)[:, None, :, :]).sum(3).sum(2)  # current context, (batch,D)

        # the second GRU layer
        ct = torch.cat((ctC, ctP), 1)
        z2 = torch.sigmoid(self.fc_Uhz2(h1) + self.fc_Wcz(ctC))  # zt  (batch,n)
        r2 = torch.sigmoid(self.fc_Uhr2(h1) + self.fc_Wcr(ctC))  # rt  (batch,n)
        h2_p = torch.tanh(self.fc_Uhh2(h1) * r2 + self.fc_Wch(ctC))  # (batch,n)
        h2 = z2 * h1 + (1. - z2) * h2_p  # (batch,n)
        h2 = ly_mask[:, None] * h2 + (1. - ly_mask)[:, None] * h1

        return h2, h1, h01, ctC, ctP, ct, calpha.permute(2, 0, 1), calpha_past, palpha.permute(2, 0, 1), palpha_past

    def _step_slice_parent(self, ly_mask, ctx_mask, h_, palpha_past_, repctx_, cc_, state_below_lz, state_below_lr, state_below_lh):
        z0 = torch.sigmoid(self.fc_Uhz0(h_) + state_below_lz)  # (batch,n)
        r0 = torch.sigmoid(self.fc_Uhr0(h_) + state_below_lr)  # (batch,n)
        h0_p = torch.tanh(self.fc_Uhh0(h_) * r0 + state_below_lh)  # (batch,n)
        h0 = z0 * h_ + (1. - z0) * h0_p  # (batch,n)
        h0 = ly_mask[:, None] * h0 + (1. - ly_mask)[:, None] * h_
        # attention for parent symbol
        query_parent = self.fc_WaP(h0)
        palpha_past__ = palpha_past_[:, None, :, :]
        cover_FP = self.conv_QP(palpha_past__).permute(2, 3, 0, 1)  # (H,W,batch,n')
        pcover_vector = self.fc_UfP(cover_FP)
        pattention_score = torch.tanh(repctx_ + query_parent[None, None, :, :] + pcover_vector)
        palpha = self.fc_vaP(pattention_score)
        palpha = palpha - palpha.max()
        palpha = palpha.view(palpha.shape[0], palpha.shape[1], palpha.shape[2])
        palpha = torch.exp(palpha)
        if (ctx_mask is not None):
            palpha = palpha * ctx_mask.permute(1, 2, 0)
        palpha = palpha / (palpha.sum(1).sum(0)[None, None, :] + 1e-10)  # (H,W,batch)
        palpha_past = palpha_past_ + palpha.permute(2, 0, 1)  # (batch,H,W)
        ctP = (cc_ * palpha.permute(2, 0, 1)[:, None, :, :]).sum(3).sum(2)

        z01 = torch.sigmoid(self.fc_Uhz1(h0) + self.fc_Wyz1(ctP))  # zt  (batch,n)
        r01 = torch.sigmoid(self.fc_Uhr1(h0) + self.fc_Wyr1(ctP))  # rt  (batch,n)
        h01_p = torch.tanh(self.fc_Uhh1(h0) * r01 + self.fc_Wyh1(ctP))  # (batch,n)
        h01 = z01 * h0 + (1. - z01) * h01_p  # (batch,n)
        h01 = ly_mask[:, None] * h01 + (1. - ly_mask)[:, None] * h0

        return h01, ctP, palpha.permute(2, 0, 1), palpha_past

    def _step_slice_child(self, ly_mask, ctx_mask, h_, calpha_past_, 
        pctx_, cc_, state_below_z, state_below_r, state_below_h, ctP):

        # the first GRU layer
        z1 = torch.sigmoid(self.fc_Uhz(h_) + state_below_z)  # (batch,n)
        r1 = torch.sigmoid(self.fc_Uhr(h_) + state_below_r)  # (batch,n)
        h1_p = torch.tanh(self.fc_Uhh(h_) * r1 + state_below_h)  # (batch,n)
        h1 = z1 * h_ + (1. - z1) * h1_p  # (batch,n)
        h1 = ly_mask[:, None] * h1 + (1. - ly_mask)[:, None] * h_

        # attention for child symbol
        query_child = self.fc_WaC(h1)
        calpha_past__ = calpha_past_[:, None, :, :]  # (batch,1,H,W)
        cover_FC = self.conv_QC(calpha_past__).permute(2, 3, 0, 1)  # (H,W,batch,n')
        ccover_vector = self.fc_UfC(cover_FC)  # (H,W,batch,n')
        cattention_score = torch.tanh(pctx_ + query_child[None, None, :, :] + ccover_vector)  # (H,W,batch,n')
        calpha = self.fc_vaC(cattention_score)  # (H,W,batch,1)
        calpha = calpha - calpha.max()
        calpha = calpha.view(calpha.shape[0], calpha.shape[1], calpha.shape[2])  # (H,W,batch)
        calpha = torch.exp(calpha)  # exp
        if (ctx_mask is not None):
            calpha = calpha * ctx_mask.permute(1, 2, 0)
        calpha = (calpha / calpha.sum(1).sum(0)[None, None, :] + 1e-10)  # (H,W,batch)
        calpha_past = calpha_past_ + calpha.permute(2, 0, 1)  # (batch,H,W)
        ctC = (cc_ * calpha.permute(2, 0, 1)[:, None, :, :]).sum(3).sum(2)  # current context, (batch,D)

        # the second GRU layer
        ct = torch.cat((ctC, ctP), 1)
        z2 = torch.sigmoid(self.fc_Uhz2(h1) + self.fc_Wcz(ctC))  # zt  (batch,n)
        r2 = torch.sigmoid(self.fc_Uhr2(h1) + self.fc_Wcr(ctC))  # rt  (batch,n)
        h2_p = torch.tanh(self.fc_Uhh2(h1) * r2 + self.fc_Wch(ctC))  # (batch,n)
        h2 = z2 * h1 + (1. - z2) * h2_p  # (batch,n)
        h2 = ly_mask[:, None] * h2 + (1. - ly_mask)[:, None] * h1

        return h2, h1, ctC, ctP, ct, calpha.permute(2, 0, 1), calpha_past

# calculate probabilities
class Gru_prob(nn.Module):
    def __init__(self, params):
        super(Gru_prob, self).__init__()
        self.fc_WctC = nn.Linear(params['D'], params['m'])
        self.fc_WhtC = nn.Linear(params['n'], params['m'])
        self.fc_WytC = nn.Linear(params['m'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_W0C = nn.Linear(int(params['m'] / 2), params['K'])
        # self.fc_WctP = nn.Linear(params['D'], params['m'])
        self.fc_W0P = nn.Linear(int(params['m'] / 2), params['K'])
        self.fc_WctRe = nn.Linear(2*params['D'], params['mre'])
        self.fc_W0Re = nn.Linear(int(params['mre']), params['Kre'])

    def forward(self, ctCs, ctPs, cts, htCs, prevC, use_dropout):
        clogit = self.fc_WctC(ctCs) + self.fc_WhtC(htCs) + self.fc_WytC(prevC)  # (seqs_y,batch,m)
        # maxout
        cshape = clogit.shape  # (seqs_y,batch,m)
        cshape2 = int(cshape[2] / 2)  # m/2
        cshape3 = 2
        clogit = clogit.view(cshape[0], cshape[1], cshape2, cshape3)  # (seqs_y,batch,m) -> (seqs_y,batch,m/2,2)
        clogit = clogit.max(3)[0]  # (seqs_y,batch,m/2)
        if use_dropout:
            clogit = self.dropout(clogit)
        cprob = self.fc_W0C(clogit)  # (seqs_y,batch,K)

        plogit = self.fc_WctC(ctPs)
        # maxout
        pshape = plogit.shape  # (seqs_y,batch,m)
        pshape2 = int(pshape[2] / 2)  # m/2
        pshape3 = 2
        plogit = plogit.view(pshape[0], pshape[1], pshape2, pshape3)  # (seqs_y,batch,m) -> (seqs_y,batch,m/2,2)
        plogit = plogit.max(3)[0]  # (seqs_y,batch,m/2)
        if use_dropout:
            plogit = self.dropout(plogit)
        pprob = self.fc_W0P(plogit)  # (seqs_y,batch,K)
        
        relogit = self.fc_WctRe(cts)
        if use_dropout:
            relogit = self.dropout(relogit)
        reprob = self.fc_W0Re(relogit)

        return cprob, pprob, reprob
