import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from itertools import product
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from keras.models import load_model
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv1D, AveragePooling1D
from keras.layers import Dense, Dropout, ZeroPadding1D, Dot, Reshape, Concatenate
from keras import optimizers
from sklearn.metrics import roc_auc_score

def Generator(molsList, one_hot, pad_len=1200, xdtype=np.float32):
    # this function transform rdkit molecules in the needed format for the 
    # graph convolutional neural networks

    N = len(molsList)
    if isinstance(molsList[0], str):
        molsList = [Chem.MolFromSmiles(x) for x in molsList]
    
    x = []
    idx_map = []

    for i in range(0,N):
        try:
            smiEnc, per = get_mol_and_pos(molsList[i], one_hot, pad_len=pad_len)
        except:
            smiEnc = np.zeros((int(pad_len/8),len(one_hot)+4))
            per = np.zeros((pad_len,int(pad_len/8)+1))
#            print('problematic smiles: '+Chem.MolToSmiles(molsList[i])) if molsList[i] != None else print('None')
        x.append(smiEnc)
        idx_map.append(per)

    return np.asarray(x,dtype=xdtype)/(len(one_hot)+4), np.asarray(idx_map,dtype=np.int8)

def get_mol_and_pos(mol, one_hot, pad_len):
    # this function converts each molecule in two arrays
    # the first array indicates the atom types of the molecule with an one hot encoding 
    # plus a bond encoding
    # the second array indicates the atom neighborhood
    atoms = mol.GetAtoms()
    mol_grid = np.zeros((int(pad_len/8),len(one_hot)+4))
    pos_grid = np.zeros((pad_len,int(pad_len/8)+1))
    k = 1 
    i = 0
    for a in range(min(len(atoms), int(pad_len/8))):
        atom = atoms[a]
        c_sym = atom.GetSymbol()
        center = get_one_hot(atom, atom, one_hot)
        mol_grid[a]=center
        center_idx = atom.GetIdx()+1
        for neighbor in atom.GetNeighbors()[:4]: # at max 4 neighbors are considered
            idx = neighbor.GetIdx()+1 
            n_sym = atom.GetSymbol()
            pos_grid[i, idx] = 1
            pos_grid[i+1, center_idx] = 1
            i+=2     
        i = (a+1)*8*k
    return(mol_grid, pos_grid)

def get_one_hot(a, bond, one_hot):
    # this function transforms a rdkit-atom into a one hot vector plus a bond encoding
    sym = a.GetSymbol()
    arom = a.GetIsAromatic()
    if arom:
        sym = sym.lower()
    else:
        add = str(a.GetNumImplicitHs()) #add number of implicit Hs for the one hot encoding
        sym = "".join((sym, add))
    if sym not in one_hot:
        sym = sym[:-1]
    if sym not in one_hot:
        sym = 'X'
    vec = np.zeros(len(one_hot))
    bonds = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    bond_vec = [str(b.GetBondType()) for b in a.GetBonds()]
    bond_vec = [int(bond_vec.count(bond)>1) for bond in bonds]
    vec[one_hot.index(sym)] = 1
    vec = np.concatenate([vec, bond_vec])
    return(vec)

def create_model(input_shape, output_shape, filters, dense_layers, opt):
    # this function creates a keras network for graph convolutions
    inputx = Input(shape=input_shape)
    x = inputx
    idx_input = Input(shape=(int(input_shape[0]*8), input_shape[0]+1))
    nr_atoms = int(input_shape[0])
    f=0
    p=0
    for f in range(len(filters)):
        x = ZeroPadding1D(padding=(1,0))(x)
        x = Dot(axes=[2,1])([idx_input, x])
        x = Conv1D(kernel_size=2, filters=filters[f], strides=2, kernel_initializer='he_normal', 
                   activation='relu', padding='valid', use_bias=False)(x)
        x = AveragePooling1D(pool_size=4, padding='valid')(x)

    x = AveragePooling1D(pool_size=nr_atoms)(x)
    x = Reshape((filters[f],))(x)

    for i in range(len(dense_layers)):
        x = Dense(dense_layers[i], activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = Dropout(0.5)(x)

    outputs = Dense(output_shape, activation='sigmoid',kernel_initializer='lecun_normal',bias_initializer='zeros')(x)
    model = Model(inputs=[inputx, idx_input], outputs=outputs)
    model.compile(loss=K.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    return model

def run_model(model, inputTrain, yTrain, inputVal, yVal, inputTest, yTest, n=100, batch_size=32, model_file=None):
    # this function trains the network and tracks the process
    tr = np.zeros((n, yTrain.shape[1]))
    val = np.zeros((n, yTrain.shape[1])) 
    tst = np.zeros((n, yTrain.shape[1])) 
    
    best_val = 0
    not_improv = 0
    
    for i in range(n):
        model.fit(x=inputTrain, y=yTrain, batch_size=batch_size, epochs=1)
        P_train = model.predict(inputTrain) 
        P_val = model.predict(inputVal)
        P_tst = model.predict(inputTest)
        
        tr[i] = roc_auc_score(yTrain, P_train)
        val[i] = roc_auc_score(yVal, P_val)
        tst[i] = roc_auc_score(yTest, P_tst)
            
        if model_file!=None:
            if val[i]>best_val:
                best_val = val[i]
                model.save(model_file+'.h5')
                not_improv=0
            else:
                not_improv+=1
        print(i, tr[i], val[i], tst[i])
        
        if not_improv == 50 or np.logical_and(i>50,best_val<0.63):
            return model, tr, val, tst
    return model, tr, val, tst

def get_activations(model, x, p, layers, filters, pad_len=800, bs = 56):
    # this function calculates the hidden representations in a batch mode
    atoms = int(pad_len/8)
    dim = x.shape[0]
    mol_shape = filters[-1]
    act_atom = np.zeros((dim*atoms, np.sum(filters)))
    act_mol = np.zeros((dim, mol_shape))
    
    act_last0 = K.function([model.input[0], model.input[1]],[model.layers[layers[0]].output])
    act_last1 = K.function([model.input[0], model.input[1]], [model.layers[layers[1]].output])
    act_last2 = K.function([model.input[0], model.input[1]],[model.layers[layers[2]].output])
    mol_act = K.function([model.input[0], model.input[1]],[model.layers[layers[3]].output])

    
    for b in range(int(dim/bs)+int(dim%bs>0)):
        train0 = act_last0([x[bs*b:bs*(b+1)], p[bs*b:bs*(b+1)],False])[0].reshape(-1,filters[0])
        train1 = act_last1([x[bs*b:bs*(b+1)], p[bs*b:bs*(b+1)],False])[0].reshape(-1,filters[1])
        train2 = act_last2([x[bs*b:bs*(b+1)], p[bs*b:bs*(b+1)],False])[0].reshape(-1,filters[2])
        mol_layer = mol_act([x[bs*b:bs*(b+1)], p[bs*b:bs*(b+1)],False])[0].reshape(-1,mol_shape)
        act_atom[bs*b*atoms:bs*(b+1)*atoms] = np.concatenate([train0, train1, train2], axis=1)
        act_mol[bs*b:bs*(b+1)] = mol_layer
    return(act_atom, act_mol)

def reduce_act(act, mols, nr_atoms=100):
    # this function reduces the atom activations to unique representations
    bool_idx = np.zeros(len(mols)*nr_atoms).astype(bool)
    atoms_per_mol = []
    atoms = []
    for i in range(len(mols)):
        mol = mols[i]
        if mol != None:
            nr = np.min([len(mol.GetAtoms()),nr_atoms])
            atomsx = mol.GetAtoms()
            atoms.extend(np.array(atomsx)[:nr])
        else:
            nr = 1
        nr = np.min([nr_atoms,nr])
        atoms_per_mol.append(nr)
        bool_idx[i*nr_atoms:(i+1)*nr_atoms] = np.concatenate([np.repeat(True,nr), np.repeat(False, nr_atoms-nr)])
    reduced = act[bool_idx]
    reduced_df = pd.DataFrame(reduced)
    unique_mask = np.logical_not(reduced_df.duplicated())
    unique_reduced = reduced_df[unique_mask].values
    rep_idx = np.repeat(np.arange(x.shape[0]), atoms_per_mol)
    aa = []
    idx = []
    for i in range(np.sum(bool_idx)):
        atom = atoms[i].GetIdx() if atoms[i]!= 'x' else 'x'
        if unique_mask[i]: 
            aa.append(atom)
            idx.append(rep_idx[i])
    return unique_reduced, idx, aa

def get_substruct(mol, atom_idx, radius=1):
    # this function creates submolecules
    for r in range(radius)[::-1]:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
        amap={}
        submol=Chem.PathToSubmol(mol,env,atomMap=amap)
        smi = Chem.MolToSmiles(submol)
        if smi!="":
            break
    return submol


