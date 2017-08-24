import sys
import os
import numpy as np
import pdb
#sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/Keras-2.0.4-py2.7.egg')
from keras.models import Sequential
import keras.layers.core as core
import keras.layers.convolutional as conv
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization, Lambda, GlobalMaxPooling2D
from keras.layers import LSTM, Bidirectional, Reshape
from keras.layers.embeddings import Embedding
#from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from keras.layers import merge, Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import WeightRegularizer, l1, l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
#from customlayers import Recalc, ReRank, ExtractDim, SoftReRank, ActivityRegularizerOneDim, RecalcExpand, Softmax4D
from keras.constraints import maxnorm
from attention import Attention,myFlatten

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import objectives
from keras import backend as K
#from keras.utils import np_utils, plot_model
from sklearn.metrics import roc_curve, auc, roc_auc_score
_EPSILON = K.epsilon()
import random
import gzip
import pickle
import timeit

def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[1:] #discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    
    return seq_dict

def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq
def read_rna_dict(rna_dict = 'rna_dict'):
    odr_dict = {}
    with open(rna_dict, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict

def get_6_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**6
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        n=n/base
        ch3=chars[n%base]
        n=n/base
        ch4=chars[n%base]
        n=n/base
        ch5=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    return  nucle_com 

def split_training_validation(classes, validation_size = 0.2, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label 

def split_overlap_seq(seq):
    window_size = 101
    overlap_size = 20
    #pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - 101)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - 101)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(num_ins):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence(seq1)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            #pdb.set_trace()
            #start = len(seq) -window_size
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence(seq1)
            bag_seqs.append(pad_seq)
    return bag_seqs
            
def get_6_nucleotide_composition(tris, seq, ordict):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    #tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer)
            tri_feature.append(ordict[str(ind)])
        else:
            tri_feature.append(-1)
    #tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()        
    return np.asarray(tri_feature)

def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    
    return seq_list, labels

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        #if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        #else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def get_RNA_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def load_graphprot_data(protein, train = True, path = '../GraphProt_CLIP_sequences/'):
    data = dict()
    tmp = []
    listfiles = os.listdir(path)
    
    key = '.train.'
    if not train:
        key = '.ls.'
    mix_label = []
    mix_seq = []
    mix_structure = []    
    for tmpfile in listfiles:
        if protein not in tmpfile:
            continue
        if key in tmpfile:
            if 'positive' in tmpfile:
                label = 1
            else:
                label = 0
            seqs, labels = read_seq_graphprot(os.path.join(path, tmpfile), label = label)
            #pdb.set_trace()
            mix_label = mix_label + labels
            mix_seq = mix_seq + seqs
    
    data["seq"] = mix_seq
    data["Y"] = np.array(mix_label)
    
    return data

def loaddata_graphprot(protein, train = True, ushuffle = True):
    #pdb.set_trace()
    data = load_graphprot_data(protein, train = train)
    label = data["Y"]
    rna_array = []
    #trids = get_6_trids()
    #nn_dict = read_rna_dict()
    for rna_seq in data["seq"]:
        #rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        
        seq_array = get_RNA_seq_concolutional_array(seq)
        #tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
        rna_array.append(seq_array)
    
    return np.array(rna_array), label

def get_bag_data(data):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    longlen = 0
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = 501)
        tri_fea = get_RNA_concolutional_array(bag_seq)
        bags.append(tri_fea)

    return bags, labels # bags,

def mil_squared_error(y_true, y_pred):
    return K.tile(K.square(K.max(y_pred) - K.max(y_true)), 5)

def custom_objective(y_true, y_pred):
    #prediction = Flatten(name='flatten')(dense_3)
    #prediction = ReRank(k=k, label=1, name='output')(prediction)
    #prediction = SoftReRank(softmink=softmink, softmaxk=softmaxk, label=1, name='output')(prediction)
    '''Just another crossentropy'''
    #y_true = K.clip(y_true, _EPSILON, 1.0-_EPSILON)
    y_true = K.max(y_true)
    #y_armax_index = numpy.argmax(y_pred)
    y_new = K.max(y_pred)
    #y_new = max(y_pred)
    '''
    if y_new >= 0.5:
        y_new_label = 1
    else:
        y_new_label = 0
    cce = abs(y_true - y_new_label)
    '''
    logEps=1e-8
    cce = - (y_true * K.log(y_new+logEps) + (1 - y_true)* K.log(1-y_new + logEps))
    return cce

def set_cnn_model_mil(ninstance=4, input_dim = 4, input_length = 107):
    nbfilter = 16
    model = Sequential() # #seqs * seqlen * 4
    #model.add(brnn)
    model.add(Conv2D(input_shape=(ninstance, input_length, input_dim),
                            filters=nbfilter,
                            kernel_size=(1,10),
                            padding="valid",
                            #activation="relu",
                            strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,3))) # 32 16
    # model.add(Dropout(0.25)) # will be better
    model.add(Conv2D(filters=nbfilter*2, kernel_size=(1,32), padding='valid', activation='relu', strides=1))

    model.add(Dropout(0.25))
    #model.add(Conv2D(filters=1, kernel_size=(1,1), padding='valid', activation='sigmoid', strides=1))
    return model



def get_rnarecommend(rnas, rna_seq_dict, shuffle = True):
    data = {}
    label = []
    rna_seqs = []
    if not shuffle:
        all_rnas = set(rna_seq_dict.keys()) - set(rnas)
        all_rnas = list(all_rnas)
        random.shuffle(all_rnas)
    ind = 0    
    for rna in rnas:
        rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        label.append(1)
        rna_seqs.append(rna_seq)
        label.append(0)
        if shuffle:
            shuff_seq = doublet_shuffle(rna_seq)
        else:
            sele_rna = all_rnas[ind]
            shuff_seq = rna_seq_dict[sele_rna]
            ind = ind + 1
            
        rna_seqs.append(shuff_seq)
    data["seq"] = rna_seqs
    data["Y"] = np.array(label)
    
    return data

def get_bag_data_1_channel(seqs, labels, max_len = 501):
    bags = []
    #seqs = data["seq"]
    #labels = data["Y"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        #bag_subt.append(tri_fea.T)
        bags.append(np.array(tri_fea))    
    return bags, labels

def load_rnacomend_data(datadir = '../data/'):
    pair_file = datadir + 'interactions_HT.txt'
    #rbp_seq_file = datadir + 'rbps_HT.fa'
    rna_seq_file = datadir + 'utrs.fa'
     
    rna_seq_dict = read_fasta_file(rna_seq_file)

    inter_pair = {}
    with open(pair_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split()
            protein = values[0]
            rna = values[1]
            inter_pair.setdefault(protein, []).append(rna)
    
    return inter_pair, rna_seq_dict

def remove_redundant_seq(inputfa1, inputfa2, outputfile):
    
    clistr = 'cd-hit-est-2d -i '+ inputfa1 + ' -i2 ' + inputfa2 + ' -o ' + outputfile + ' -c 0.8 -n 6 -T 8'
    fcli = os.popen(clistr, 'r')
    fcli.close()

def read_seq_again(seq_file):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line.rstrip().split()
                label = int(name[-1])
                labels.append(label)
            else:
                seq = line[:-1].upper()
                seq_list.append(seq)

    return seq_list, labels  

def get_all_rna_mildata(seqs, labels, training_val_indice, train_val_label, test_indice, test_label, channel = 1, max_len = 2695, outputdir = 'datasave', redund = False):
    train_file = outputdir + '/train.fa'
    train_seqs = []
    if not os.path.exists(train_file):
    	#train_seqs = []
    	#train_file = outputdir + '/train.fa'
    	fw  = open(train_file, 'w')
    	index = 0
    	for val in training_val_indice:
        	train_seqs.append(seqs[val])
        	fw.write('>seq' + str(index)  +'\t' + str(train_val_label[index]) + '\n')
        	fw.write(seqs[val] + '\n')
        	index = index + 1
    	fw.close()
    else:
    	train_seqs, train_val_label = read_seq_again(train_file)

    if channel == 1:
        train_bags, label = get_bag_data_1_channel(train_seqs, train_val_label, max_len = max_len)
    else:
        train_bags, label = get_bag_data(train_seqs, train_val_label, channel = channel)

    test_seqs = []
		
    testfile = outputdir + '/test.fa'
    test_seqs = []
    if not os.path.exists(testfile):
    	fw = open(testfile, 'w')
    	index = 0
    	for val in test_indice:
		test_seqs.append(seqs[val]) 
		fw.write('>seq' + str(index) + '\t' + str(test_label[index]) + '\n')
		fw.write(seqs[val] + '\n')
		index = index + 1      
    	fw.close()
    else:
	test_seqs, test_label = read_seq_again(testfile)

    if redund:
	outputfile = outputdir + '/nonredundant_test.fa'
	remove_redundant_seq(train_file, testfile, outputfile)

	os.remove(train_file)
	os.remove(testfile)
	test_seqs, test_label = read_seq_again(outputfile)
	os.remove(outputfile)

    if channel == 1:
        test_bags, true_y = get_bag_data_1_channel(test_seqs, test_label, max_len = max_len) 
    else:
        test_bags, true_y = get_bag_data(test_seqs, test_label, channel = channel) 
    #pdb.set_trace()
    return train_bags, label, test_bags, true_y
        
def get_all_embedding(protein):
    
    data = load_graphprot_data(protein)
    #pdb.set_trace()
    train_bags, label = get_bag_data(data)
    #pdb.set_trace()
    test_data = load_graphprot_data(protein, train = False)
    test_bags, true_y = get_bag_data(test_data) 
    
    return train_bags, label, test_bags, true_y

def set_cnn_model(input_dim = 4, input_length = 107):
    nbfilter = 16
    model = Sequential()
    #model.add(brnn)
    
    model.add(Conv1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=10,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(nbfilter*2, activation='relu'))
    model.add(Dropout(0.5))

    return model

def set_cnn_model_attention(input_dim = 4, input_length = 2701):
	attention_reg_x = 0.25
	attention_reg_xr = 1
	attentionhidden_x = 16
	attentionhidden_xr = 8
	nbfilter = 16
	input = Input(shape=(input_length, input_dim))
	x = conv.Convolution1D(nbfilter, 10 ,border_mode="valid")(input) 
	x = Dropout(0.5)(x)
	x = Activation('relu')(x)
	x = conv.MaxPooling1D(pool_length=3)(x)
	x_reshape=core.Reshape((x._keras_shape[2],x._keras_shape[1]))(x)

	x = Dropout(0.5)(x)
	x_reshape=Dropout(0.5)(x_reshape)

	decoder_x = Attention(hidden=attentionhidden_x,activation='linear') # success  
	decoded_x=decoder_x(x)
	output_x = myFlatten(x._keras_shape[2])(decoded_x)

	decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear')
	decoded_xr=decoder_xr(x_reshape)
	output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)

	output=merge([output_x,output_xr, Flatten()(x)],mode='concat')
        #output = BatchNormalization()(output)
	output=Dropout(0.5)(output)
        print output.shape
	output=Dense(nbfilter*10,activation="relu")(output)
	output=Dropout(0.5)(output)
	out=Dense(2,activation='softmax')(output)
        #output = BatchNormalization()(output)
	model=Model(input,out)
	model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

	return model

def run_network(model, total_hid, train_bags, test_bags, y_bags):
    model.add(Dense(2)) # binary classification
    model.add(Activation('softmax')) # #instance * 2
    #model.add(GlobalMaxPooling2D()) # max pooling multi instance 

    model.summary()
    savemodelpng = 'net.png'
    #plot_model(model, to_file=savemodelpng, show_shapes=True)
    print(len(train_bags), len(test_bags), len(y_bags), train_bags[0].shape, y_bags[0].shape, len(train_bags[0]))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #model.compile(loss=custom_objective, optimizer='rmsprop')
    print 'model training'
    nb_epochs= 5
    y_bags = to_categorical(y_bags, 2)
    model.fit(np.array(train_bags), np.array(y_bags), batch_size = 100, epochs=nb_epochs, verbose = 1)
    print 'predicting'         
    predictions = model.predict_proba(np.array(test_bags))[:, 1]

    return predictions

def run_network_att(model, total_hid, train_bags, test_bags, y_bags):
    #model.add(Dense(2)) # binary classification
    #model.add(Activation('softmax')) # #instance * 2
    #model.add(GlobalMaxPooling2D()) # max pooling multi instance 

    #model.summary()
    print(len(train_bags), len(test_bags), len(y_bags), train_bags[0].shape, len(train_bags[0]))

    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #model.compile(loss=custom_objective, optimizer='rmsprop')
    print 'model training'
    nb_epochs= 30
    y_bags = to_categorical(y_bags, 2)
    #pdb.set_trace()
    model.fit(np.array(train_bags), np.array(y_bags), batch_size = 100, nb_epoch=nb_epochs, verbose = 0)
    print 'predicting' 
    #pdb.set_trace()        
    predictions = model.predict(np.array(test_bags))[:, 1]

    return predictions


def run_rnacomend_ideepa( shuffle = False):
    inter_pair_dict, rna_seq_dict = load_rnacomend_data()
    #fw = open('result_rnarecommend', 'w')

    fw = open('attention_out', 'w')    
    start_time = timeit.default_timer()
    protien_list =['MOV10', 'ELAVL1', 'PABPC1', 'SRRM4', 'C22ORF28', 'YTHDF2', 'PUM1', 
                   'HNRNPH1', 'TAF15', 'CAPRIN1', 'TIA1', 'FUS', 'U2AF2', 'HNRNPU', 'RBM10',
                   'PCBP2', 'HNRNPA2B1', 'YTHDF1', 'RC3H1', 'HNRNPC', 'HNRNPD', 'ZFP36',
                   'ADAR1', 'LARP4B', 'MSI1', 'LIN28B', 'LIN28A', 'FXR1', 'FXR2', 'IGF2BP1',
                   'IGF2BP3', 'IGF2BP2', 'STAU1', 'PUM2', 'DDX21', 'RBM47', 'AGO1', 'RBPMS',
                   'FMR1_iso1', 'FMR1_iso7', 'HNRNPF', 'TIAL1', 'ZC3H7B', 'ATXN2', 'EWSR1',
                   'EIF4A3', 'AGO2']
    protein_dict = {'ELAVL1':'CLIPSEQ_ELAVL1', 'MOV10':'PARCLIP_MOV10_Sievers', 
                    'C22ORF28':'C22ORF28_Baltz2012', 'TIA1':'ICLIP_TIA1', 'TIAL1':'ICLIP_TIAL1',
                    'TAF15':'PARCLIP_TAF15', 'FUS':'PARCLIP_FUS', 'ZC3H7B':'ZC3H7B_Baltz2012',
                    'AGO2':'CLIPSEQ_AGO2', 'EWSR1':'PARCLIP_EWSR1', 'PUM2':'PARCLIP_PUM2',
                    'HNRNPC':'ICLIP_HNRNPC'}
    #new_protein_dict = {v: k for k, v in protein_dict.iteritems()}
    dir1 = 'data/'
    for protein, rnas in inter_pair_dict.iteritems():
        if len(rnas) < 2000:
            continue
        #if protein not in protein_dict:
        #    continue
    	out_dir = dir1 + protein
    	if not os.path.isdir(out_dir):
        	os.mkdir(out_dir)
        print protein
        fw.write(protein + '\t')
        data = get_rnarecommend(rnas, rna_seq_dict, shuffle = shuffle)
        labels = data["Y"]
        seqs = data["seq"]
        training_val_indice, train_val_label, test_indice, test_label = split_training_validation(labels)
        train_bags, train_labels, test_bags, test_labels = get_all_rna_mildata(seqs, labels, training_val_indice, train_val_label, test_indice, test_label, channel = 1, max_len = 2695, outputdir = out_dir)
        net =  set_cnn_model_attention()#set_cnn_model(input_length = 2701)
        
        #seq_auc, seq_predict = calculate_auc(seq_net)
        hid = 16
        predict = run_network_att(net, hid, train_bags, test_bags, train_labels)
        
        auc = roc_auc_score(test_labels, predict)
        print 'AUC:', auc
        fw.write(str(auc) + '\n')
        mylabel = "\t".join(map(str, test_labels))
        myprob = "\t".join(map(str, predict))  
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')

run_rnacomend_ideepa()
#run_milcnn()
#seq= 'TTATCTCCTAGAAGGGGAGGTTACCTCTTCAAATGAGGAGGCCCCCCAGTCCTGTTCCTCCACCAGCCCCACTACGGAATGGGAGCGCATTTTAGGGTGGTTACTCTGAAACAAGGAGGGCCTAGGAATCTAAGAGTGTGAAGAGTAGAGAGGAAGTACCTCTACCCACCAGCCCACCCGTGCGGGGGAAGATGTAGCAGCTTCTTCTCCGAACCAA'
#print len(seq)
#split_overlap_seq(seq)
