import time
import os
import re
import numpy as np
import random
import torch
from torch import optim, nn
from utils import BatchBucket, load_dict, prepare_data, gen_sample, weight_init, compute_wer, compute_sacc
from encoder_decoder import Encoder_Decoder
from data_iterator import dataIterator

# whether use multi-GPUs
multi_gpu_flag = False
# whether init params
init_param_flag = True
# whether reload params
reload_flag = False

# load configurations
# root_paths
bfs2_path = ''
work_path = ''

model_idx = 7
# paths
dictionaries = [bfs2_path + '106_dictionary.txt', bfs2_path + '7relation_dictionary.txt']
datasets = [bfs2_path + 'jiaming-train-py3.pkl', bfs2_path + 'train_data_v1_label_gtd.pkl', bfs2_path + 'train_data_v1_label_align_gtd.pkl']
valid_datasets = [bfs2_path + 'jiaming-valid-py3.pkl', bfs2_path + 'valid_data_v1_label_gtd.pkl', bfs2_path + 'valid_data_v1_label_align_gtd.pkl']
valid_output = [work_path+'results'+str(model_idx)+'/symbol_relation/', work_path+'results'+str(model_idx)+'/memory_alpha/']
valid_result = [work_path+'results'+str(model_idx)+'/valid.cer', work_path+'results'+str(model_idx)+'/valid.exprate']
saveto = work_path+'models'+str(model_idx)+'/WAP_params.pkl'
last_saveto = work_path+'models'+str(model_idx)+'/WAP_params_last.pkl'

# training settings
if multi_gpu_flag:
    batch_Imagesize = 500000
    valid_batch_Imagesize = 500000
    batch_size = 24
    valid_batch_size = 24
else:
    batch_Imagesize = 500000
    valid_batch_Imagesize = 500000
    batch_size = 8
    valid_batch_size = 8
    maxImagesize = 500000
maxlen = 200
max_epochs = 5000
lrate = 1.0
my_eps = 1e-6
decay_c = 1e-4
clip_c = 100.

# early stop
estop = False
halfLrFlag = 0
bad_counter = 0
patience = 15
validStart = 10
finish_after = 10000000

# model architecture
params = {}
params['n'] = 256
params['m'] = 256
params['dim_attention'] = 512
params['D'] = 684
params['K'] = 106

params['Kre'] = 7
params['mre'] = 256
params['maxlen'] = maxlen

params['growthRate'] = 24
params['reduction'] = 0.5
params['bottleneck'] = True
params['use_dropout'] = True
params['input_channels'] = 1

params['ly_lambda'] = 1.
params['ry_lambda'] = 0.1
params['re_lambda'] = 1.
params['rpos_lambda'] = 1.
params['KL_lambda'] = 0.1

# load dictionary
worddicts = load_dict(dictionaries[0])
print ('total chars',len(worddicts))
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk

reworddicts = load_dict(dictionaries[1])
print ('total relations',len(reworddicts))
reworddicts_r = [None] * len(reworddicts)
for kk, vv in reworddicts.items():
    reworddicts_r[vv] = kk

train,train_uid_list = dataIterator(datasets[0], datasets[1], datasets[2], worddicts, reworddicts,
                         batch_size=batch_size, batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize)
valid,valid_uid_list = dataIterator(valid_datasets[0], valid_datasets[1], valid_datasets[2], worddicts, reworddicts,
                         batch_size=valid_batch_size, batch_Imagesize=valid_batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize)
# display
uidx = 0  # count batch
lpred_loss_s = 0.  # count loss
rpred_loss_s = 0.
repred_loss_s = 0.
mem_loss_s = 0.
KL_loss_s = 0.
loss_s = 0.
ud_s = 0  # time for training an epoch
validFreq = -1
saveFreq = -1
sampleFreq = -1
dispFreq = 100
if validFreq == -1:
    validFreq = len(train)
if saveFreq == -1:
    saveFreq = len(train)
if sampleFreq == -1:
    sampleFreq = len(train)

# initialize model
WAP_model = Encoder_Decoder(params)
if init_param_flag:
    WAP_model.apply(weight_init)
if multi_gpu_flag:
    WAP_model = nn.DataParallel(WAP_model, device_ids=[0, 1, 2, 3])
if reload_flag:
    WAP_model.load_state_dict(torch.load(saveto,map_location=lambda storage,loc:storage))
WAP_model.cuda()

# print model's parameters
model_params = WAP_model.named_parameters()
for k, v in model_params:
    print(k)

# loss function
# criterion = torch.nn.CrossEntropyLoss(reduce=False)
# optimizer
optimizer = optim.Adadelta(WAP_model.parameters(), lr=lrate, eps=my_eps, weight_decay=decay_c)

print('Optimization')

# statistics
history_errs = []

for eidx in range(max_epochs):
    n_samples = 0
    ud_epoch = time.time()
    random.shuffle(train)
    for x, ly, ry, re, ma, lp, rp in train:
        WAP_model.train()
        ud_start = time.time()
        n_samples += len(x)
        uidx += 1
        x, x_mask, ly, ly_mask, ry, ry_mask, re, re_mask, ma, ma_mask, lp, rp = \
                                prepare_data(params, x, ly, ry, re, ma, lp, rp)

        x = torch.from_numpy(x).cuda()  # (batch,1,H,W)
        x_mask = torch.from_numpy(x_mask).cuda()  # (batch,H,W)
        ly = torch.from_numpy(ly).cuda()  # (seqs_y,batch)
        ly_mask = torch.from_numpy(ly_mask).cuda()  # (seqs_y,batch)
        ry = torch.from_numpy(ry).cuda()  # (seqs_y,batch)
        ry_mask = torch.from_numpy(ry_mask).cuda()  # (seqs_y,batch)
        re = torch.from_numpy(re).cuda()  # (seqs_y,batch)
        re_mask = torch.from_numpy(re_mask).cuda()  # (seqs_y,batch)
        ma = torch.from_numpy(ma).cuda()  # (batch,seqs_y,seqs_y)
        ma_mask = torch.from_numpy(ma_mask).cuda()  # (batch,seqs_y,seqs_y)
        lp = torch.from_numpy(lp).cuda()  # (seqs_y,batch)
        rp = torch.from_numpy(rp).cuda()  # (seqs_y,batch)

        # permute for multi-GPU training
        # ly = ly.permute(1, 0)
        # ly_mask = ly_mask.permute(1, 0)
        # ry = ry.permute(1, 0)
        # ry_mask = ry_mask.permute(1, 0)
        # lp = lp.permute(1, 0)
        # rp = rp.permute(1, 0)

        # forward
        loss, lpred_loss, rpred_loss, repred_loss, mem_loss, KL_loss = \
            WAP_model(params, x, x_mask, ly, ly_mask, ry, ry_mask, re, re_mask, ma, ma_mask, lp, rp)

        # recover from permute
        lpred_loss_s += lpred_loss.item()
        rpred_loss_s += rpred_loss.item()
        repred_loss_s += repred_loss.item()
        mem_loss_s += mem_loss.item()
        KL_loss_s += KL_loss.item()
        loss_s += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        if clip_c > 0.:
            torch.nn.utils.clip_grad_norm_(WAP_model.parameters(), clip_c)

        # update
        optimizer.step()

        ud = time.time() - ud_start
        ud_s += ud

        # display
        if np.mod(uidx, dispFreq) == 0:
            ud_s /= 60.
            loss_s /= dispFreq
            lpred_loss_s /= dispFreq
            rpred_loss_s /= dispFreq
            repred_loss_s /= dispFreq
            mem_loss_s /= dispFreq
            KL_loss_s /= dispFreq
            print ('Epoch', eidx, ' Update', uidx, ' Cost_lpred %.7f, Cost_rpred %.7f, Cost_re %.7f, Cost_matt %.7f, Cost_kl %.7f' % \
                (np.float(lpred_loss_s),np.float(rpred_loss_s),np.float(repred_loss_s),np.float(mem_loss_s),np.float(KL_loss_s)), \
                ' UD %.3f' % ud_s, ' lrate', lrate, ' eps', my_eps, ' bad_counter', bad_counter)
            ud_s = 0
            loss_s = 0.
            lpred_loss_s = 0.
            rpred_loss_s = 0.
            repred_loss_s = 0.
            mem_loss_s = 0.
            KL_loss_s = 0.

        # validation
        if np.mod(uidx, sampleFreq) == 0 and eidx >= validStart:
            print('begin sampling')
            ud_epoch_train = (time.time() - ud_epoch) / 60.
            print('epoch training cost time ... ', ud_epoch_train)
            WAP_model.eval()
            valid_out_path = valid_output[0]
            valid_malpha_path = valid_output[1]
            if not os.path.exists(valid_out_path):
                os.mkdir(valid_out_path)
            if not os.path.exists(valid_malpha_path):
                os.mkdir(valid_malpha_path)
            rec_mat = {}
            label_mat = {}
            rec_re_mat = {}
            label_re_mat = {}
            rec_ridx_mat = {}
            label_ridx_mat = {}
            with torch.no_grad():
                valid_count_idx = 0
                for x, ly, ry, re, ma, lp, rp in valid:
                    for xx, lyy, ree, rpp in zip(x, ly, re, rp):
                        xx_pad = xx.astype(np.float32) / 255.
                        xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda()  # (1,1,H,W)
                        score, sample, malpha_list, relation_sample = \
                            gen_sample(WAP_model, xx_pad, params, multi_gpu_flag, k=3, maxlen=maxlen, rpos_beam=3)
                        
                        key = valid_uid_list[valid_count_idx]
                        rec_mat[key] = []
                        label_mat[key] = lyy
                        rec_re_mat[key] = []
                        label_re_mat[key] = ree
                        rec_ridx_mat[key] = []
                        label_ridx_mat[key] = rpp
                        if len(score) == 0:
                            rec_mat[key].append(0)
                            rec_re_mat[key].append(0) # End
                            rec_ridx_mat[key].append(0)
                        else:
                            score = score / np.array([len(s) for s in sample])
                            min_score_index = score.argmin()
                            ss = sample[min_score_index]
                            rs = relation_sample[min_score_index]
                            mali = malpha_list[min_score_index]
                            for i, [vv, rv] in enumerate(zip(ss, rs)):
                                if vv == 0:
                                    rec_mat[key].append(vv)
                                    rec_re_mat[key].append(0) # End
                                    break
                                else:
                                    if i == 0:
                                        rec_mat[key].append(vv)
                                        rec_re_mat[key].append(6) # Start
                                    else:
                                        rec_mat[key].append(vv)
                                        rec_re_mat[key].append(rv)
                            ma_idx_list = np.array(mali).astype(np.int64)
                            ma_idx_list[-1] = int(len(ma_idx_list)-1)
                            rec_ridx_mat[key] = ma_idx_list
                        valid_count_idx=valid_count_idx+1

            print('valid set decode done')
            ud_epoch = (time.time() - ud_epoch) / 60.
            print('epoch cost time ... ', ud_epoch)

        if np.mod(uidx, saveFreq) == 0:
            print('Saving latest model params ... ')
            torch.save(WAP_model.state_dict(), last_saveto)

        # calculate wer and expRate
        if np.mod(uidx, validFreq) == 0 and eidx >= validStart:
            valid_cer_out = compute_wer(rec_mat, label_mat)
            valid_cer = 100. * valid_cer_out[0]
            valid_recer_out = compute_wer(rec_re_mat, label_re_mat)
            valid_recer = 100. * valid_recer_out[0]
            valid_ridxcer_out = compute_wer(rec_ridx_mat, label_ridx_mat)
            valid_ridxcer = 100. * valid_ridxcer_out[0]
            valid_exprate = compute_sacc(rec_mat, label_mat, rec_ridx_mat, label_ridx_mat, rec_re_mat, label_re_mat, worddicts_r, reworddicts_r)
            valid_exprate = 100. * valid_exprate
            valid_err=valid_cer+valid_ridxcer
            history_errs.append(valid_err)

            # the first time validation or better model
            if uidx // validFreq == 0 or valid_err <= np.array(history_errs).min():
                bad_counter = 0
                print('Saving best model params ... ')
                if multi_gpu_flag:
                    torch.save(WAP_model.module.state_dict(), saveto)
                else:
                    torch.save(WAP_model.state_dict(), saveto)

            # worse model
            if uidx / validFreq != 0 and valid_err > np.array(history_errs).min():
                bad_counter += 1
                if bad_counter > patience:
                    if halfLrFlag == 2:
                        print('Early Stop!')
                        estop = True
                        break
                    else:
                        print('Lr decay and retrain!')
                        bad_counter = 0
                        lrate = lrate / 10.
                        params['KL_lambda'] = params['KL_lambda'] * 0.5
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lrate
                        halfLrFlag += 1
            print ('Valid CER: %.2f%%, relation_CER: %.2f%%, rpos_CER: %.2f%%, ExpRate: %.2f%%' % (valid_cer,valid_recer,valid_ridxcer,valid_exprate))
        # finish after these many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            estop = True
            break

    print('Seen %d samples' % n_samples)

    # early stop
    if estop:
        break
