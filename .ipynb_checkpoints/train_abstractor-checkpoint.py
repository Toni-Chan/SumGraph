""" train the abstractor"""
# public models
import argparse
import json
import os, re
from os.path import join, exists
import pickle as pkl

from cytoolz import compose, concat

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# training pipeline. no need to change
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer, MultiTaskPipeline, MultiTaskTrainer
from model.multibart import multiBartGAT
from model.util import sequence_loss


# dataset processing. This is same as GraphAugmentedSum
from data.data import CnnDmDataset
from data.batcher import coll_fn, prepro_fn
from data.batcher import prepro_fn_copy_bart, convert_batch_copy_bart, batchify_fn_copy_bart
from data.batcher import convert_batch_copy, batchify_fn_copy
from data.batcher import BucketedGenerater
from data.abs_batcher import convert_batch_gat, batchify_fn_gat, prepro_fn_gat, coll_fn_gat
from data.abs_batcher import convert_batch_gat_bart, batchify_fn_gat_bart, prepro_fn_gat_bart
from training import multitask_validate

from utils import PAD, UNK, START, END
from utils import make_vocab, make_embedding
from transformers import BartTokenizer, BartConfig
import pickle

# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 6400

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class MatchDataset(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted'])
        extracts = sorted(extracts)
        matched_arts = [art_sents[i] for i in extracts]
        return matched_arts, abs_sents[:len(extracts)]

class SumDataset(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents = (
            js_data['article'], js_data['abstract'])
        art_sents = [' '.join(art_sents)]
        abs_sents = [' '.join(abs_sents)]
        return art_sents, abs_sents

class MatchDataset_all2all(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents = (
            js_data['article'], js_data['abstract'])
        matched_arts = [' '.join(art_sents)]
        abs_sents = [' '.join(abs_sents)]
        return matched_arts, abs_sents

class MatchDataset_graph(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, key='nodes_pruned2', subgraph=False):
        super().__init__(split, DATA_DIR)
        self.node_key = key
        self.edge_key = key.replace('nodes', 'edges')
        self.subgraph = subgraph

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, nodes, edges, subgraphs, paras = (
            js_data['article'], js_data['abstract'], js_data[self.node_key], js_data[self.edge_key], js_data['subgraphs'], js_data['paragraph_merged'])
        #art_sents = [' '.join(art_sents)]
        abs_sents = [' '.join(abs_sents)]
        return art_sents, abs_sents, nodes, edges, subgraphs, paras

def get_bart_align_dict(filename='preprocessing/bartalign-base.pkl'):
    with open(filename, 'rb') as f:
        bart_dict = pickle.load(f)
    return bart_dict

def configure_bart_gat(vocab_size, emb_dim, n_encoder, n_decoder, drop_encoder, drop_decoder,
                  load_from=None, gat_args={}, max_art=2048,
                  static_pos_emb=False):
    
    net_args = BartConfig(
        vocab_size=vocab_size,
        d_model=emb_dim,
        encoder_layers=n_encoder,
        decoder_layers=n_decoder,
        encoder_layerdrop=drop_encoder,
        decoder_layerdrop=drop_decoder,
        static_position_embedding=static_pos_emb,
        max_position_embeddings=max_art,
    )
    
    net = multiBartGAT(net_args, gat_args)
    
    if load_from is not None:
        abs_ckpt = load_best_ckpt(load_from)
        net.load_state_dict(abs_ckpt)

    return net, net_args, gat_args



def load_best_ckpt(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward/score"""
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    print('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(
        join(model_dir, 'ckpt/{}'.format(ckpts[0])), map_location=lambda storage, loc: storage
    )['state_dict']
    return ckpt

def configure_training(opt, lr, clip_grad, lr_decay, batch_size, bart):
    """ supports Adam optimizer only"""
    assert opt in ['adam', 'adagrad']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    if opt == 'adagrad':
        opt_kwargs['initial_accumulator_value'] = 0.1
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay
    if bart:
        PAD = 1
    else:
        PAD = 0
    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)
    def criterion(logits, targets):
        return sequence_loss(logits, targets, nll, pad_idx=PAD)

    print('pad id:', PAD)
    return criterion, train_params

def configure_training_multitask(opt, lr, clip_grad, lr_decay, batch_size, mask_type, bart):
    """ supports Adam optimizer only"""
    assert opt in ['adam', 'adagrad']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    if opt == 'adagrad':
        opt_kwargs['initial_accumulator_value'] = 0.1
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    if bart:
        PAD = 1
    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)

    bce = lambda logit, target: F.binary_cross_entropy(logit, target, reduce=False)
    def criterion(logits1, logits2, targets1, targets2):
        aux_loss = None
        for logit in logits2:
            if aux_loss is None:
                aux_loss = sequence_loss(logit, targets2, bce, pad_idx=-1, if_aux=True, fp16=False).mean()
            else:
                aux_loss += sequence_loss(logit, targets2, bce, pad_idx=-1, if_aux=True, fp16=False).mean()
        return (sequence_loss(logits1, targets1, nll, pad_idx=PAD).mean(), aux_loss)
    print('pad id:', PAD)
    return criterion, train_params


def build_batchers_bart(cuda, debug, bart_model):
    tokenizer = BartTokenizer.from_pretrained(bart_model)
    #tokenizer = BertTokenizer.from_pretrained(bart_model)
    prepro = prepro_fn_copy_bart(tokenizer, args.max_art, args.max_abs)
    def sort_key(sample):
        src, target = sample[0], sample[1]
        return (len(target), len(src))
    batchify = compose(
        batchify_fn_copy_bart(tokenizer, cuda=cuda),
        convert_batch_copy_bart(tokenizer, args.max_art)
    )

    train_loader = DataLoader(
        SumDataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)
    val_loader = DataLoader(
        SumDataset('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                        single_run=True, fork=not debug)

    return train_batcher, val_batcher, tokenizer.encoder

def build_batchers_gat_bart(cuda, debug, gold_key, adj_type,
                       mask_type, num_worker=4, bart_model='bart-base'):
    print('adj_type:', adj_type)
    print('mask_type:', mask_type)
    tokenizer = BartTokenizer.from_pretrained(bart_model)

    with open(os.path.join(DATA_DIR, 'bart-base-align.pkl'), 'rb') as f:
        align = pickle.load(f)

    prepro = prepro_fn_gat_bart(tokenizer, align, args.max_art, args.max_abs, key=gold_key, adj_type=adj_type, docgraph=False)
    key = 'nodes'
    _coll_fn = coll_fn_gat(max_node_num=400)
    def sort_key(sample):
        src, target = sample[0], sample[1]
        return (len(target), len(src))

    batchify = compose(
            batchify_fn_gat_bart(tokenizer, cuda=cuda,
                         adj_type=adj_type, mask_type=mask_type, docgraph=docgraph),
            convert_batch_gat_bart(tokenizer, args.max_art)
        )

    train_loader = DataLoader(
        MatchDataset_graph('train', key=key, subgraph=subgraph), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=num_worker if cuda and not debug else 0,
        collate_fn=_coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)
    val_loader = DataLoader(
        MatchDataset_graph('val', key=key, subgraph=subgraph), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=num_worker if cuda and not debug else 0,
        collate_fn=_coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)

    return train_batcher, val_batcher, tokenizer.encoder

def main(args):
    import logging
    logging.basicConfig(level=logging.ERROR)
    
    # create data batcher, vocabulary

    # batcher
    train_batcher, val_batcher, word2id = build_batchers_gat_bart(
                                                            args.cuda, args.debug, args.gold_key, args.adj_type,
                                                            args.mask_type, args.topic_flow_model,
                                                            num_worker=args.num_worker, bart_model=args.bartmodel)

    # make net
    _args = {}
    _args['rtoks'] = 1
    _args['graph_hsz'] = args.n_hidden
    _args['blockdrop'] = 0.1
    _args['sparse'] = False
    _args['graph_model'] = 'transformer'
    _args['adj_type'] = args.adj_type
    _args['mask_type'] = args.mask_type
    _args['node_freq'] = args.node_freq

    net, net_args = configure_bart_gat(args.vsize, args.emb_dim, args.n_encoder, args.n_decoder, 
                                      args.drop_encoder, args.drop_decoder, args.load_from, _args, 
                                      args.max_art, args.static_pos_emb)

    # configure training setting
    if 'soft' in args.mask_type and args.gat:
        criterion, train_params = configure_training_multitask(
            'adam', args.lr, args.clip, args.decay, args.batch, args.mask_type,
            args.bart
        )
    else:
        criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch, args.bart
        )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['net']           = 'base_abstractor'
    meta['net_args']      = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    if args.cuda:
        net = net.cuda()


    if 'soft' in args.mask_type and args.gat:
        val_fn = multitask_validate(net, criterion)
    else:
        val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    print(net._embedding.weight.requires_grad)

    optimizer = optim.AdamW(net.parameters(), **train_params['optimizer'][1])
    #optimizer = optim.Adagrad(net.parameters(), **train_params['optimizer'][1])

    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    # pipeline = BasicPipeline(meta['net'], net,
    #                          train_batcher, val_batcher, args.batch, val_fn,
    #                          criterion, optimizer, grad_fn)
    # trainer = BasicTrainer(pipeline, args.path,
    #                        args.ckpt_freq, args.patience, scheduler)

    if 'soft' in args.mask_type and args.gat:
        pipeline = MultiTaskPipeline(meta['net'], net,
                                 train_batcher, val_batcher, args.batch, val_fn,
                                 criterion, optimizer, grad_fn)
        trainer = MultiTaskTrainer(pipeline, args.path,
                               args.ckpt_freq, args.patience, scheduler)
    else:
        pipeline = BasicPipeline(meta['net'], net,
                                 train_batcher, val_batcher, args.batch, val_fn,
                                 criterion, optimizer, grad_fn)
        trainer = BasicTrainer(pipeline, args.path,
                               args.ckpt_freq, args.patience, scheduler)


    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    # Basics
    parser.add_argument('--path', required=True, help='root of the model')
    
    # parser.add_argument('--key', type=str, default='extracted_combine', help='constructed sentences')
    # Settings that align with BART
    parser.add_argument('--vsize', type=int, action='store', default=50000,
                        help='vocabulary size') # BartConfig.vocab_size
    parser.add_argument('--emb_dim', type=int, action='store', default=1024,
                        help='the dimension of word embedding') # BartConfig.d_model
    parser.add_argument('--n_encoder', type=int, action='store', default=12,
                        help='number of encoder layer') # BartConfig.encoder_layers
    parser.add_argument('--n_decoder', type=int, action='store', default=12,
                        help='number of decoder layer') # BartConfig.decoder_layers
    parser.add_argument('--drop_encoder', type=int, action='store', default=0.0,
                        help='dropout rate of encoder between layers') # BartConfig.decoder_layerdrop
    parser.add_argument('--drop_decoder', type=int, action='store', default=0.0,
                        help='dropout rate of decoder between layers') # BartConfig.decoder_layerdrop
    parser.add_argument('--max_art', type=int, action='store', default=2048,
                        help='maximun words in a single article sentence') # BartConfig.max_position_embeddings
    parser.add_argument('--max_abs', type=int, action='store', default=256,
                        help='maximun words in a single abstract sentence') # BartConfig.max_position_embeddings
    parser.add_argument('--static_pos_emb', type=int, action='store', default=False,
                        help='use of sinosuidal position embeddings or learned ones') # BartConfig.static_position_embeddings
    
    ## Can add other settings based on BartConfig class if needed

    # GAT Configs
    parser.add_argument('--adj_type', action='store', default='edge_as_node', type=str,
                        help='concat_triple, edge_up, edge_down, no_edge, edge_as_node')
    parser.add_argument('--mask_type', action='store', default='soft', type=str,
                        help='none, encoder, soft')
    parser.add_argument('--node_freq', action='store_true', default=False)

    # data preprocessing
    parser.add_argument('--adj_type', action='store', default='edge_as_node', type=str,
                        help='concat_triple, edge_up, edge_down, no_edge, edge_as_node')
    parser.add_argument('--gold_key', action='store', default='summary_worthy', type=str,
                        help='attention type')


    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument('--num_worker', type=int, action='store', default=4,
                        help='cpu num using for dataloader')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=9000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--load_from', type=str, default=None,
                        help='loading from file')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id if only 1 gpu is used')
    args = parser.parse_args()
    if args.debug:
        BUCKET_SIZE = 64

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    args.n_gpu = 1

    main(args)
