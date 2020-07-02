import random
import numpy as np
import copy
import sys
import pickle as pkl
import torch
from torch import nn


class BatchBucket():
    def __init__(self, max_h, max_w, max_l, max_img_size, max_batch_size, feature_file, label_file, dictionary,
                 use_all=True):
        self._max_img_size = max_img_size
        self._max_batch_size = max_batch_size
        self._fea_file = feature_file
        self._label_file = label_file
        self._dictionary_file = dictionary
        self._use_all = use_all
        self._dict_load()
        self._data_load()
        self.keys = self._calc_keys(max_h, max_w, max_l)
        self._make_plan()
        self._reset()

    def _dict_load(self):
        fp = open(self._dictionary_file)
        stuff = fp.readlines()
        fp.close()
        self._lexicon = {}
        for l in stuff:
            w = l.strip().split()
            self._lexicon[w[0]] = int(w[1])

    def _data_load(self):
        fp_fea = open(self._fea_file, 'rb')
        self._features = pkl.load(fp_fea)
        fp_fea.close()
        fp_label = open(self._label_file, 'r')
        labels = fp_label.readlines()
        fp_label.close()
        self._targets = {}
        for l in labels:
            tmp = l.strip().split()
            uid = tmp[0]
            w_list = []
            for w in tmp[1:]:
                if self._lexicon.__contains__(w):
                    w_list.append(self._lexicon[w])
                else:
                    print('a word not in the dictionary !! sentence ', uid, 'word ', w)
                    sys.exit()
            self._targets[uid] = w_list
        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[1], fea.shape[2], len(label)) for (uid, fea), (_, label) in
                             zip(self._features.items(), self._targets.items())]

    def _calc_keys(self, max_h, max_w, max_l):
        mh = mw = ml = 0
        for _, h, w, l in self._data_parser:
            if h > mh:
                mh = h
            if w > mw:
                mw = w
            if l > ml:
                ml = l
        max_h = min(max_h, mh)
        max_w = min(max_w, mw)
        max_l = min(max_l, ml)
        keys = []
        init_h = 64 if 64 < max_h else max_h
        init_w = 64 if 64 < max_w else max_w
        init_l = 20 if 20 < max_l else max_l
        h_step = 64
        w_step = 64
        l_step = 30
        h = init_h
        while h <= max_h:
            w = init_w
            while w <= max_w:
                l = init_l
                while l <= max_l:
                    keys.append([h, w, l, h * w * l, 0])
                    if l < max_l and l + l_step > max_l:
                        l = max_l
                        continue
                    l += l_step
                if w < max_w and w + w_step > max_w:
                    w = max_w
                    continue
                w += w_step
            if h < max_h and h + h_step > max_h:
                h = max_h
                continue
            h += h_step
        keys = sorted(keys, key=lambda area: area[3])
        for _, h, w, l in self._data_parser:
            for i in range(len(keys)):
                hh, ww, ll, _, _ = keys[i]
                if h <= hh and w <= ww and l <= ll:
                    keys[i][-1] += 1
                    break
        new_keys = []
        n_samples = len(self._data_parser)
        th = n_samples * 0.01
        if self._use_all:
            th = 1
        num = 0
        for key in keys:
            hh, ww, ll, _, n = key
            num += n
            if num >= th:
                new_keys.append((hh, ww, ll))
                num = 0
        return new_keys

    def _make_plan(self):
        self._bucket_keys = []
        for h, w, l in self.keys:
            batch_size = int(self._max_img_size / (h * w))
            if batch_size > self._max_batch_size:
                batch_size = self._max_batch_size
            if batch_size == 0:
                continue
            self._bucket_keys.append((batch_size, h, w, l))
        self._data_buckets = [[] for key in self._bucket_keys]
        unuse_num = 0
        for item in self._data_parser:
            flag = 0
            for key, bucket in zip(self._bucket_keys, self._data_buckets):
                _, h, w, l = key
                if item[1] <= h and item[2] <= w and item[3] <= l:
                    bucket.append(item)
                    flag = 1
                    break
            if flag == 0:
                unuse_num += 1
        print('The number of unused samples: ', unuse_num)
        all_sample_num = 0
        for key, bucket in zip(self._bucket_keys, self._data_buckets):
            sample_num = len(bucket)
            all_sample_num += sample_num
            print('bucket {}, sample number={}'.format(key, len(bucket)))
        print('All samples number={}, raw samples number={}'.format(all_sample_num, len(self._data_parser)))

    def _reset(self):
        # shuffle data in each bucket
        for bucket in self._data_buckets:
            random.shuffle(bucket)
        self._batches = []
        for id, (key, bucket) in enumerate(zip(self._bucket_keys, self._data_buckets)):
            batch_size, _, _, _ = key
            bucket_len = len(bucket)
            batch_num = (bucket_len + batch_size - 1) // batch_size
            for i in range(batch_num):
                start = i * batch_size
                end = start + batch_size if start + batch_size < bucket_len else bucket_len
                if start != end:  # remove empty batch
                    self._batches.append(bucket[start:end])

    def get_batches(self):
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            label_batch = []
            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                label = self._targets[uid]
                fea_batch.append(feature)
                label_batch.append(label)
                uid_batches.append(uid)
            batches.append((fea_batch, label_batch))
        return batches, uid_batches


# load dictionary
def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon


# create batch
def prepare_data(params, images_x, seqs_ly, seqs_ry, seqs_re, seqs_ma, seqs_lp, seqs_rp):
    heights_x = [s.shape[1] for s in images_x]
    widths_x = [s.shape[2] for s in images_x]
    lengths_ly = [len(s) for s in seqs_ly]
    lengths_ry = [len(s) for s in seqs_ry]

    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_ly = np.max(lengths_ly)
    maxlen_ry = np.max(lengths_ry)

    x = np.zeros((n_samples, params['input_channels'], max_height_x, max_width_x)).astype(np.float32)
    ly = np.zeros((maxlen_ly, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
    ry = np.zeros((maxlen_ry, n_samples)).astype(np.int64)
    re = np.zeros((maxlen_ly, n_samples)).astype(np.int64)
    ma = np.zeros((n_samples, maxlen_ly, maxlen_ly)).astype(np.int64)
    lp = np.zeros((maxlen_ly, n_samples)).astype(np.int64)
    rp = np.zeros((maxlen_ry, n_samples)).astype(np.int64)

    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
    ly_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
    ry_mask = np.zeros((maxlen_ry, n_samples)).astype(np.float32)
    re_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
    ma_mask = np.zeros((n_samples, maxlen_ly, maxlen_ly)).astype(np.float32)

    for idx, [s_x, s_ly, s_ry, s_re, s_ma, s_lp, s_rp] in enumerate(zip(images_x, seqs_ly, seqs_ry, seqs_re, seqs_ma, seqs_lp, seqs_rp)):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x / 255.
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        ly[:lengths_ly[idx], idx] = s_ly
        ly_mask[:lengths_ly[idx], idx] = 1.
        ry[:lengths_ry[idx], idx] = s_ry
        ry_mask[:lengths_ry[idx], idx] = 1.
        ry_mask[0, idx] = 0. # remove the <s>
        re[:lengths_ly[idx], idx] = s_re
        re_mask[:lengths_ly[idx], idx] = 1.
        re_mask[0, idx] = 0. # remove the Start relation
        re_mask[lengths_ly[idx]-1, idx] = 0. # remove the End relation
        ma[idx, :lengths_ly[idx], :lengths_ly[idx]] = s_ma
        for ma_idx in range(lengths_ly[idx]):
            ma_mask[idx, :(ma_idx+1), ma_idx] = 1.
        lp[:lengths_ly[idx], idx] = s_lp
        # lp_mask[:lengths_ly[idx], idx] = 1
        rp[:lengths_ry[idx], idx] = s_rp

    return x, x_mask, ly, ly_mask, ry, ry_mask, re, re_mask, ma, ma_mask, lp, rp

def gen_sample(model, x, params, gpu_flag, k=1, maxlen=30, rpos_beam=3):
    
    sample = []
    sample_score = []
    rpos_sample = []
    # rpos_sample_score = []
    relation_sample = []

    live_k = 1
    dead_k = 0  # except init, live_k = k - dead_k

    # current living paths and corresponding scores(-log)
    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)
    hyp_rpos_samples = [[]] * live_k
    hyp_relation_samples = [[]] * live_k
    # get init state, (1,n) and encoder output, (1,D,H,W)
    next_state, ctx0 = model.f_init(x)
    next_h1t = next_state
    # -1 -> My_embedding -> 0 tensor(1,m)
    next_lw = -1 * torch.ones(1, dtype=torch.int64).cuda()
    next_calpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()  # (live_k,H,W)
    next_palpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()
    nextemb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()
    nextePmb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()    

    for ii in range(maxlen):
        ctxP = ctx0.repeat(live_k, 1, 1, 1)  # (live_k,D,H,W)
        next_lpos = ii * torch.ones(live_k, dtype=torch.int64).cuda()
        next_h01, next_ma, next_ctP, next_pa, next_palpha_past, nextemb_memory, nextePmb_memory = \
                    model.f_next_parent(params, next_lw, next_lpos, ctxP, next_state, next_h1t, next_palpha_past, nextemb_memory, nextePmb_memory, ii)
        next_ma = next_ma.cpu().numpy()
        # next_ctP = next_ctP.cpu().numpy()
        next_palpha_past = next_palpha_past.cpu().numpy()
        nextemb_memory = nextemb_memory.cpu().numpy()
        nextePmb_memory = nextePmb_memory.cpu().numpy()

        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2)) # batch * Matt * dim
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        
        next_rpos = next_ma.argsort(axis=1)[:,-rpos_beam:] # topK parent index; batch * topK
        n_gaps = nextemb_memory.shape[1]
        n_batch = nextemb_memory.shape[0]
        next_rpos_gap = next_rpos + n_gaps * np.arange(n_batch)[:, None]
        next_remb_memory = nextemb_memory.reshape([n_batch*n_gaps, nextemb_memory.shape[-1]])
        next_remb = next_remb_memory[next_rpos_gap.flatten()] # [batch*rpos_beam, emb_dim]
        rpos_scores = next_ma.flatten()[next_rpos_gap.flatten()] # [batch*rpos_beam,]

        # next_ctPC = next_ctP.repeat(1, 1, rpos_beam)
        # next_ctPC = torch.reshape(next_ctPC, (-1, next_ctP.shape[1]))
        ctxC = ctx0.repeat(live_k*rpos_beam, 1, 1, 1)
        next_ctPC = torch.zeros(next_ctP.shape[0]*rpos_beam, next_ctP.shape[1]).cuda()
        next_h01C = torch.zeros(next_h01.shape[0]*rpos_beam, next_h01.shape[1]).cuda()
        next_calpha_pastC = torch.zeros(next_calpha_past.shape[0]*rpos_beam, next_calpha_past.shape[1], next_calpha_past.shape[2]).cuda()
        for bidx in range(next_calpha_past.shape[0]):
            for ridx in range(rpos_beam):
                next_ctPC[bidx*rpos_beam+ridx] = next_ctP[bidx]
                next_h01C[bidx*rpos_beam+ridx] = next_h01[bidx]
                next_calpha_pastC[bidx*rpos_beam+ridx] = next_calpha_past[bidx]
        next_remb = torch.from_numpy(next_remb).cuda()

        next_lp, next_rep, next_state, next_h1t, next_ca, next_calpha_past, next_re = \
                    model.f_next_child(params, next_remb, next_ctPC, ctxC, next_h01C, next_calpha_pastC)

        next_lp = next_lp.cpu().numpy()
        next_state = next_state.cpu().numpy()
        next_h1t = next_h1t.cpu().numpy()
        next_calpha_past = next_calpha_past.cpu().numpy()
        next_re = next_re.cpu().numpy()

        hyp_scores = np.tile(hyp_scores[:, None], [1, rpos_beam]).flatten()
        cand_scores = hyp_scores[:, None] - np.log(next_lp+1e-10)- np.log(rpos_scores+1e-10)[:,None]
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        voc_size = next_lp.shape[1]
        trans_indices = ranks_flat // voc_size
        trans_indicesP = ranks_flat // (voc_size*rpos_beam)
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        # update paths
        new_hyp_samples = []
        new_hyp_scores = np.zeros(k-dead_k).astype('float32')
        new_hyp_rpos_samples = []
        new_hyp_relation_samples = []
        new_hyp_states = []
        new_hyp_h1ts = []
        new_hyp_calpha_past = []
        new_hyp_palpha_past = []
        new_hyp_emb_memory = []
        new_hyp_ePmb_memory = []
        
        for idx, [ti, wi, tPi] in enumerate(zip(trans_indices, word_indices, trans_indicesP)):
            new_hyp_samples.append(hyp_samples[tPi]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_rpos_samples.append(hyp_rpos_samples[tPi]+[next_rpos.flatten()[ti]])
            new_hyp_relation_samples.append(hyp_relation_samples[tPi]+[next_re[ti]])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_h1ts.append(copy.copy(next_h1t[ti]))
            new_hyp_calpha_past.append(copy.copy(next_calpha_past[ti]))
            new_hyp_palpha_past.append(copy.copy(next_palpha_past[tPi]))
            new_hyp_emb_memory.append(copy.copy(nextemb_memory[tPi]))
            new_hyp_ePmb_memory.append(copy.copy(nextePmb_memory[tPi]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_rpos_samples = []
        hyp_relation_samples = []
        hyp_states = []
        hyp_h1ts = []
        hyp_calpha_past = []
        hyp_palpha_past = []
        hyp_emb_memory = []
        hyp_ePmb_memory = []

        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0: # <eol>
                sample_score.append(new_hyp_scores[idx])
                sample.append(new_hyp_samples[idx])
                rpos_sample.append(new_hyp_rpos_samples[idx])
                relation_sample.append(new_hyp_relation_samples[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_scores.append(new_hyp_scores[idx])
                hyp_samples.append(new_hyp_samples[idx])
                hyp_rpos_samples.append(new_hyp_rpos_samples[idx])
                hyp_relation_samples.append(new_hyp_relation_samples[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_h1ts.append(new_hyp_h1ts[idx])
                hyp_calpha_past.append(new_hyp_calpha_past[idx])
                hyp_palpha_past.append(new_hyp_palpha_past[idx])
                hyp_emb_memory.append(new_hyp_emb_memory[idx])
                hyp_ePmb_memory.append(new_hyp_ePmb_memory[idx])   
                    
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_lw = np.array([w[-1] for w in hyp_samples])  # each path's final symbol, (live_k,)
        next_state = np.array(hyp_states)  # h2t, (live_k,n)
        next_h1t = np.array(hyp_h1ts)
        next_calpha_past = np.array(hyp_calpha_past)  # (live_k,H,W)
        next_palpha_past = np.array(hyp_palpha_past)
        nextemb_memory = np.array(hyp_emb_memory)
        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2))
        nextePmb_memory = np.array(hyp_ePmb_memory)
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        next_lw = torch.from_numpy(next_lw).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_h1t = torch.from_numpy(next_h1t).cuda()
        next_calpha_past = torch.from_numpy(next_calpha_past).cuda()
        next_palpha_past = torch.from_numpy(next_palpha_past).cuda()
        nextemb_memory = torch.from_numpy(nextemb_memory).cuda()
        nextePmb_memory = torch.from_numpy(nextePmb_memory).cuda()

    return sample_score, sample, rpos_sample, relation_sample


# init model params
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

# compute metric
def cmp_result(rec,label):
    dist_mat = np.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def compute_wer(rec_mat, label_mat):
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        # label = list(map(int,label))
        # rec = list(map(int,rec))
        dist, llen = cmp_result(rec, label)
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
    wer = float(total_dist)/total_label
    sacc = float(total_line_rec)/total_line
    return wer, sacc

def cmp_sacc_result(rec_list,label_list,rec_ridx_list,label_ridx_list,rec_re_list,label_re_list,chdict,redict):
    rec = True
    out_sym_pdict = {}
    label_sym_pdict = {}
    out_sym_pdict['0'] = '<s>'
    label_sym_pdict['0'] = '<s>'
    for idx, sym in enumerate(rec_list):
        out_sym_pdict[str(idx+1)] = chdict[sym]
    for idx, sym in enumerate(label_list):
        label_sym_pdict[str(idx+1)] = chdict[sym]

    if len(rec_list) != len(label_list):
        rec = False
    else:
        for idx in range(len(rec_list)):
            out_sym = chdict[rec_list[idx]]
            label_sym = chdict[label_list[idx]]
            out_repos = int(rec_ridx_list[idx])
            label_repos = int(label_ridx_list[idx])
            out_re = redict[rec_re_list[idx]]
            label_re = redict[label_re_list[idx]]
            if out_repos in out_sym_pdict:
                out_resym_s = out_sym_pdict[out_repos]
            else:
                out_resym_s = 'unknown'
            if label_repos in label_sym_pdict:
                label_resym_s = label_sym_pdict[label_repos]
            else:
                label_resym_s = 'unknown'

            # post-processing only for math recognition
            if (out_resym_s == '\lim' and label_resym_s == '\lim') or \
            (out_resym_s == '\int' and label_resym_s == '\int') or \
            (out_resym_s == '\sum' and label_resym_s == '\sum'):
                if out_re == 'Above':
                    out_re = 'Sup'
                if out_re == 'Below':
                    out_re = 'Sub'
                if label_re == 'Above':
                    label_re = 'Sup'
                if label_re == 'Below':
                    label_re = 'Sub'

            # if out_sym != label_sym or out_pos != label_pos or out_repos != label_repos or out_re != label_re:
            # if out_sym != label_sym or out_repos != label_repos:
            if out_sym != label_sym or out_repos != label_repos or out_re != label_re:
                rec = False
                break
    return rec

def compute_sacc(rec_mat, label_mat, rec_ridx_mat, label_ridx_mat, rec_re_mat, label_re_mat, chdict, redict):
    total_num = len(rec_mat)
    correct_num = 0
    for key_rec in rec_mat:
        rec_list = rec_mat[key_rec]
        label_list = label_mat[key_rec]
        rec_ridx_list = rec_ridx_mat[key_rec]
        label_ridx_list = label_ridx_mat[key_rec]
        rec_re_list = rec_re_mat[key_rec]
        label_re_list = label_re_mat[key_rec]
        rec_result = cmp_sacc_result(rec_list,label_list,rec_ridx_list,label_ridx_list,rec_re_list,label_re_list,chdict,redict)
        if rec_result:
            correct_num += 1
    correct_rate = 1. * correct_num / total_num
    return correct_rate
