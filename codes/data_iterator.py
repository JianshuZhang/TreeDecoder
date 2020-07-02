import numpy

import pickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def dataIterator(feature_file,label_file,align_file,dictionary,redictionary,batch_size,batch_Imagesize,maxlen,maxImagesize):
    
    fp_feature=open(feature_file,'rb')
    features=pkl.load(fp_feature)
    fp_feature.close()

    fp_label=open(label_file,'rb')
    labels=pkl.load(fp_label)
    fp_label.close()

    fp_align=open(align_file,'rb')
    aligns=pkl.load(fp_align)
    fp_align.close()

    ltargets = {}
    rtargets = {}
    relations = {}
    lpositions = {}
    rpositions = {}

    # map word to int with dictionary
    for uid, label_lines in labels.items():
        lchar_list = []
        rchar_list = []
        relation_list = []
        lpos_list = []
        rpos_list = []
        for line_idx, line in enumerate(label_lines):
            parts = line.strip().split('\t')
            lchar = parts[0]
            lpos = parts[1]
            rchar = parts[2]
            rpos = parts[3]
            relation = parts[4]
            if dictionary.__contains__(lchar):
                lchar_list.append(dictionary[lchar])
            else:
                print ('a symbol not in the dictionary !! formula',uid ,'symbol', lchar)
                sys.exit()
            if dictionary.__contains__(rchar):
                rchar_list.append(dictionary[rchar])
            else:
                print ('a symbol not in the dictionary !! formula',uid ,'symbol', rchar)
                sys.exit()
            
            lpos_list.append(int(lpos))
            rpos_list.append(int(rpos))  

            if line_idx != len(label_lines)-1:
                if redictionary.__contains__(relation):
                    relation_list.append(redictionary[relation])
                else:
                    print ('a relation not in the redictionary !! formula',uid ,'relation', relation)
                    sys.exit()
            else:
                relation_list.append(0) # whatever which one to replace End relation
        ltargets[uid]=lchar_list
        rtargets[uid]=rchar_list
        relations[uid]=relation_list
        lpositions[uid] = lpos_list
        rpositions[uid] = rpos_list

    imageSize={}
    for uid,fea in features.items():
        if uid in ltargets:
            imageSize[uid]=fea.shape[1]*fea.shape[2]
        else:
            continue

    imageSize= sorted(imageSize.items(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element

    feature_batch=[]
    llabel_batch=[]
    rlabel_batch=[]
    relabel_batch=[]
    align_batch=[]
    lpos_batch=[]
    rpos_batch=[]

    feature_total=[]
    llabel_total=[]
    rlabel_total=[]
    relabel_total=[]
    align_total=[]
    lpos_total=[]
    rpos_total=[]

    uidList=[]

    batch_image_size=0
    biggest_image_size=0
    i=0
    for uid,size in imageSize:
        if uid not in ltargets:
            continue
        if size>biggest_image_size:
            biggest_image_size=size
        fea=features[uid]
        llab=ltargets[uid]
        rlab=rtargets[uid]
        relab=relations[uid]
        ali=aligns[uid]
        lp=lpositions[uid]
        rp=rpositions[uid]
        batch_image_size=biggest_image_size*(i+1)
        if len(llab)>maxlen:
            print ('this sentence length bigger than', maxlen, 'ignore')
        elif size>maxImagesize:
            print ('this image size bigger than', maxImagesize, 'ignore')
        else:
            uidList.append(uid)
            if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full
                feature_total.append(feature_batch)
                llabel_total.append(llabel_batch)
                rlabel_total.append(rlabel_batch)
                relabel_total.append(relabel_batch)
                align_total.append(align_batch)
                lpos_total.append(lpos_batch)
                rpos_total.append(rpos_batch)

                i=0
                biggest_image_size=size
                feature_batch=[]
                llabel_batch=[]
                rlabel_batch=[]
                relabel_batch=[]
                align_batch=[]
                lpos_batch=[]
                rpos_batch=[]
                feature_batch.append(fea)
                llabel_batch.append(llab)
                rlabel_batch.append(rlab)
                relabel_batch.append(relab)
                align_batch.append(ali)
                lpos_batch.append(lp)
                rpos_batch.append(rp)
                batch_image_size=biggest_image_size*(i+1)
                i+=1
            else:
                feature_batch.append(fea)
                llabel_batch.append(llab)
                rlabel_batch.append(rlab)
                relabel_batch.append(relab)
                align_batch.append(ali)
                lpos_batch.append(lp)
                rpos_batch.append(rp)
                i+=1

    # last batch
    feature_total.append(feature_batch)
    llabel_total.append(llabel_batch)
    rlabel_total.append(rlabel_batch)
    relabel_total.append(relabel_batch)
    align_total.append(align_batch)
    lpos_total.append(lpos_batch)
    rpos_total.append(rpos_batch)

    print ('total ',len(feature_total), 'batch data loaded')

    return list(zip(feature_total,llabel_total,rlabel_total,relabel_total,align_total,lpos_total,rpos_total)),uidList


def dataIterator_test(feature_file,dictionary,redictionary,batch_size,batch_Imagesize,maxImagesize):
    
    fp_feature=open(feature_file,'rb')
    features=pkl.load(fp_feature)
    fp_feature.close()

    imageSize={}
    for uid,fea in features.items():
        imageSize[uid]=fea.shape[1]*fea.shape[2]

    imageSize= sorted(imageSize.items(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element

    feature_batch=[]

    feature_total=[]

    uidList=[]

    batch_image_size=0
    biggest_image_size=0
    i=0
    for uid,size in imageSize:
        if size>biggest_image_size:
            biggest_image_size=size
        fea=features[uid]
        batch_image_size=biggest_image_size*(i+1)
        if size>maxImagesize:
            print ('this image size bigger than', maxImagesize, 'ignore')
        elif uid == '34_em_225':
            print ('this image ignore', uid)
        else:
            uidList.append(uid)
            if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full
                feature_total.append(feature_batch)

                i=0
                biggest_image_size=size
                feature_batch=[]
                feature_batch.append(fea)
                batch_image_size=biggest_image_size*(i+1)
                i+=1
            else:
                feature_batch.append(fea)
                i+=1

    # last batch
    feature_total.append(feature_batch)

    print ('total ',len(feature_total), 'batch data loaded')

    return feature_total, uidList
