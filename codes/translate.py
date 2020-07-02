import argparse
import copy
import numpy as np
import os
import re
import time
import sys

import torch

from data_iterator import dataIterator, dataIterator_test
from encoder_decoder import Encoder_Decoder


# Note:
#   here model means Encoder_Decoder -->  WAP_model
#   x means a sample not a batch(or batch_size = 1),and x's shape should be (1,1,H,W),type must be Variable
#   live_k is just equal to k -dead_k(except the begin of sentence:live_k = 1,dead_k = 0,so use k-dead_k to represent the number of alive paths in beam search)

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


def main(model_path, dictionary_target, dictionary_retarget, fea, output_path, k=5):
    # set parameters
    params = {}
    params['n'] = 256
    params['m'] = 256
    params['dim_attention'] = 512
    params['D'] = 684
    params['K'] = 106
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    params['input_channels'] = 1
    params['Kre'] = 7
    params['mre'] = 256

    maxlen = 300
    params['maxlen'] = maxlen

    # load model
    model = Encoder_Decoder(params)
    model.load_state_dict(torch.load(model_path,map_location=lambda storage,loc:storage))
    # enable CUDA
    model.cuda()

    # load source dictionary and invert
    worddicts = load_dict(dictionary_target)
    print ('total chars',len(worddicts))
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    reworddicts = load_dict(dictionary_retarget)
    print ('total relations',len(reworddicts))
    reworddicts_r = [None] * len(reworddicts)
    for kk, vv in reworddicts.items():
        reworddicts_r[vv] = kk

    valid,valid_uid_list = dataIterator_test(fea, worddicts, reworddicts,
                         batch_size=8, batch_Imagesize=800000,maxImagesize=800000)

    # change model's mode to eval
    model.eval()

    valid_out_path = output_path + 'symbol_relation/'
    valid_malpha_path = output_path + 'memory_alpha/'
    if not os.path.exists(valid_out_path):
        os.mkdir(valid_out_path)
    if not os.path.exists(valid_malpha_path):
        os.mkdir(valid_malpha_path)
    valid_count_idx = 0
    print('Decoding ... ')
    ud_epoch = time.time()
    model.eval()
    with torch.no_grad():
        for x in valid:
            for xx in x:  # xx：当前batch中的一个数据,numpy
                print('%d : %s' % (valid_count_idx + 1, valid_uid_list[valid_count_idx]))
                xx_pad = np.zeros((xx.shape[0], xx.shape[1], xx.shape[2]), dtype='float32')  # (1,height,width)
                xx_pad[:, :, :] = xx / 255.
                xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda()
                score, sample, malpha_list, relation_sample = \
                            gen_sample(model, xx_pad, params, gpu_flag=False, k=k, maxlen=maxlen)
                # sys.exit()
                if len(score) != 0:
                    score = score / np.array([len(s) for s in sample])
                    # relation_score = relation_score / np.array([len(r) for r in relation_sample])
                    min_score_index = score.argmin()
                    ss = sample[min_score_index]
                    rs = relation_sample[min_score_index]
                    mali = malpha_list[min_score_index]
                    fpp_sample = open(valid_out_path+valid_uid_list[valid_count_idx]+'.txt','w')
                    file_malpha_sample = valid_malpha_path+valid_uid_list[valid_count_idx]+'_malpha.txt'
                    for i, [vv, rv] in enumerate(zip(ss, rs)):
                        if vv == 0:
                            string = worddicts_r[vv] + '\tEnd\n'
                            fpp_sample.write(string)
                            break
                        else:
                            if i == 0:
                                string = worddicts_r[vv] + '\tStart\n'
                            else:
                                string = worddicts_r[vv] + '\t' + reworddicts_r[rv] + '\n'
                            fpp_sample.write(string)
                    np.savetxt(file_malpha_sample, np.array(mali))
                    fpp_sample.close()
                valid_count_idx=valid_count_idx+1
    print('test set decode done')
    ud_epoch = (time.time() - ud_epoch) / 60.
    print('epoch cost time ... ', ud_epoch)

    # valid_result = [result_wer, result_exprate]
    # os.system('python compute_sym_re_ridx_cer.py ' + valid_out_path + ' ' + valid_malpha_path + ' ' + label_path + ' ' + valid_result[0])
    # fpp=open(valid_result[0])
    # lines = fpp.readlines()
    # fpp.close()
    # part1 = lines[-3].split()
    # if part1[0] == 'CER':
    #     valid_cer=100. * float(part1[1])
    # else:
    #     print ('no CER result')
    # part2 = lines[-2].split()
    # if part2[0] == 'reCER':
    #     valid_recer=100. * float(part2[1])
    # else:
    #     print ('no reCER result')
    # part3 = lines[-1].split()
    # if part3[0] == 'ridxCER':
    #     valid_ridxcer=100. * float(part3[1])
    # else:
    #     print ('no ridxCER result')
    # os.system('python evaluate_ExpRate2.py ' + valid_out_path + ' ' + valid_malpha_path + ' ' + label_path + ' ' + valid_result[1])
    # fpp=open(valid_result[1])
    # exp_lines = fpp.readlines()
    # fpp.close()
    # parts = exp_lines[0].split()
    # if parts[0] == 'ExpRate':
    #     valid_exprate = float(parts[1])
    # else:
    #     print ('no ExpRate result')
    # print ('ExpRate: %.2f%%' % (valid_exprate))
    # print ('Valid CER: %.2f%%, relation_CER: %.2f%%, rpos_CER: %.2f%%, ExpRate: %.2f%%' % (valid_cer,valid_recer,valid_ridxcer,valid_exprate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('model_path', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('dictionary_retarget', type=str)
    parser.add_argument('fea', type=str)
    parser.add_argument('output_path', type=str)

    args = parser.parse_args()

    main(args.model_path, args.dictionary_target, args.dictionary_retarget, args.fea, 
        args.output_path, k=args.k)
