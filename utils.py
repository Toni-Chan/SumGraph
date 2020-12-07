""" utility functions"""
import re
import os
from os.path import basename

import gensim
import torch
from torch import nn
"""Encoding Data Parallel"""
import socket

# import threading
# import functools
# from torch.autograd import Variable, Function
# import torch.cuda.comm as comm
# from torch.nn.parallel.data_parallel import DataParallel
# from torch.nn.parallel.parallel_apply import get_a_var
# from torch.nn.parallel.scatter_gather import gather
# from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
# from torch.nn.parallel import DistributedDataParallel

import logging
logger = logging.getLogger(__name__)

def count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


PAD = 0
UNK = 1
START = 2
END = 3
SPLIT = 4
def make_vocab(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2id[w] = i
    return word2id

def make_vocab_entity(wc, vocab_size):
    word2id, id2word = {}, {}
    word2id['<pad>'] = PAD
    word2id['<unk>'] = UNK
    word2id['<start>'] = START
    word2id['<end>'] = END
    word2id['<split>'] = SPLIT
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 5):
        word2id[w] = i
    return word2id

def make_embedding(id2word, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(id2word)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in range(len(id2word)):
            # NOTE: id2word can be list or dict
            if i == START:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == END:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif id2word[i] in w2v:
                embedding[i, :] = torch.Tensor(w2v[id2word[i]])
            else:
                oovs.append(i)
    return embedding, oovs


def init_gpu_params(params):
    """
    Handle single and multi-GPU / multi-node.
    """
    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()

    logger.info("Initializing GPUs")
    if params.n_gpu > 1:
        assert params.local_rank != -1

        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        params.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        assert params.n_nodes == int(os.environ["N_NODES"])
        assert params.node_id == int(os.environ["NODE_RANK"])

    # local job (single GPU)
    else:
        assert params.local_rank == -1

        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
        params.multi_gpu = False

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in multi-node distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1

    # summary
    PREFIX = f"--- Global rank: {params.global_rank} - "
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:
        logger.info("Initializing PyTorch distributed")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang, Rutgers University, Email: zhang.hang@rutgers.edu
## Modified by Thomas Wolf, HuggingFace Inc., Email: thomas@huggingface.co
## Copyright (c) 2017-2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# torch_ver = torch.__version__[:3]

# __all__ = ['allreduce', 'DataParallelModel', 'DataParallelCriterion',
#            'patch_replication_callback']

# def allreduce(*inputs):
#     """Cross GPU all reduce autograd operation for calculate mean and
#     variance in SyncBN.
#     """
#     return AllReduce.apply(*inputs)

# class AllReduce(Function):
#     @staticmethod
#     def forward(ctx, num_inputs, *inputs):
#         ctx.num_inputs = num_inputs
#         ctx.target_gpus = [inputs[i].get_device() for i in range(0, len(inputs), num_inputs)]
#         inputs = [inputs[i:i + num_inputs]
#                  for i in range(0, len(inputs), num_inputs)]
#         # sort before reduce sum
#         inputs = sorted(inputs, key=lambda i: i[0].get_device())
#         results = comm.reduce_add_coalesced(inputs, ctx.target_gpus[0])
#         outputs = comm.broadcast_coalesced(results, ctx.target_gpus)
#         return tuple([t for tensors in outputs for t in tensors])

#     @staticmethod
#     def backward(ctx, *inputs):
#         inputs = [i.data for i in inputs]
#         inputs = [inputs[i:i + ctx.num_inputs]
#                  for i in range(0, len(inputs), ctx.num_inputs)]
#         results = comm.reduce_add_coalesced(inputs, ctx.target_gpus[0])
#         outputs = comm.broadcast_coalesced(results, ctx.target_gpus)
#         return (None,) + tuple([Variable(t) for tensors in outputs for t in tensors])


# class Reduce(Function):
#     @staticmethod
#     def forward(ctx, *inputs):
#         ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
#         inputs = sorted(inputs, key=lambda i: i.get_device())
#         return comm.reduce_add(inputs)

#     @staticmethod
#     def backward(ctx, gradOutput):
#         return Broadcast.apply(ctx.target_gpus, gradOutput)

# class DistributedDataParallelModel(DistributedDataParallel):
#     """Implements data parallelism at the module level for the DistributedDataParallel module.
#     This container parallelizes the application of the given module by
#     splitting the input across the specified devices by chunking in the
#     batch dimension.
#     In the forward pass, the module is replicated on each device,
#     and each replica handles a portion of the input. During the backwards pass,
#     gradients from each replica are summed into the original module.
#     Note that the outputs are not gathered, please use compatible
#     :class:`encoding.parallel.DataParallelCriterion`.
#     The batch size should be larger than the number of GPUs used. It should
#     also be an integer multiple of the number of GPUs so that each chunk is
#     the same size (so that each GPU processes the same number of samples).
#     Args:
#         module: module to be parallelized
#         device_ids: CUDA devices (default: all devices)
#     Reference:
#         Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
#         Amit Agrawal. “Context Encoding for Semantic Segmentation.
#         *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
#     Example::
#         >>> net = encoding.nn.DistributedDataParallelModel(model, device_ids=[0, 1, 2])
#         >>> y = net(x)
#     """
#     def gather(self, outputs, output_device):
#         return outputs

# class DataParallelModel(DataParallel):
#     """Implements data parallelism at the module level.
#     This container parallelizes the application of the given module by
#     splitting the input across the specified devices by chunking in the
#     batch dimension.
#     In the forward pass, the module is replicated on each device,
#     and each replica handles a portion of the input. During the backwards pass,
#     gradients from each replica are summed into the original module.
#     Note that the outputs are not gathered, please use compatible
#     :class:`encoding.parallel.DataParallelCriterion`.
#     The batch size should be larger than the number of GPUs used. It should
#     also be an integer multiple of the number of GPUs so that each chunk is
#     the same size (so that each GPU processes the same number of samples).
#     Args:
#         module: module to be parallelized
#         device_ids: CUDA devices (default: all devices)
#     Reference:
#         Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
#         Amit Agrawal. “Context Encoding for Semantic Segmentation.
#         *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
#     Example::
#         >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
#         >>> y = net(x)
#     """
#     def gather(self, outputs, output_device):
#         return outputs

#     def replicate(self, module, device_ids):
#         modules = super(DataParallelModel, self).replicate(module, device_ids)
#         execute_replication_callbacks(modules)
#         return modules


# class DataParallelCriterion(DataParallel):
#     """
#     Calculate loss in multiple-GPUs, which balance the memory usage.
#     The targets are splitted across the specified devices by chunking in
#     the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.
#     Reference:
#         Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
#         Amit Agrawal. “Context Encoding for Semantic Segmentation.
#         *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*
#     Example::
#         >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
#         >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
#         >>> y = net(x)
#         >>> loss = criterion(y, target)
#     """
#     def forward(self, inputs, *targets, **kwargs):
#         # input should be already scatterd
#         # scattering the targets instead
#         if not self.device_ids:
#             return self.module(inputs, *targets, **kwargs)
#         targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
#         if len(self.device_ids) == 1:
#             return self.module(inputs, *targets[0], **kwargs[0])
#         replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
#         outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
#         #return Reduce.apply(*outputs) / len(outputs)
#         #return self.gather(outputs, self.output_device).mean()
#         return self.gather(outputs, self.output_device)


# def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
#     assert len(modules) == len(inputs)
#     assert len(targets) == len(inputs)
#     if kwargs_tup:
#         assert len(modules) == len(kwargs_tup)
#     else:
#         kwargs_tup = ({},) * len(modules)
#     if devices is not None:
#         assert len(modules) == len(devices)
#     else:
#         devices = [None] * len(modules)

#     lock = threading.Lock()
#     results = {}
#     if torch_ver != "0.3":
#         grad_enabled = torch.is_grad_enabled()

#     def _worker(i, module, input, target, kwargs, device=None):
#         if torch_ver != "0.3":
#             torch.set_grad_enabled(grad_enabled)
#         if device is None:
#             device = get_a_var(input).get_device()
#         try:
#             with torch.cuda.device(device):
#                 # this also avoids accidental slicing of `input` if it is a Tensor
#                 if not isinstance(input, (list, tuple)):
#                     input = (input,)
#                 if not isinstance(target, (list, tuple)):
#                     target = (target,)
#                 output = module(*(input + target), **kwargs)
#             with lock:
#                 results[i] = output
#         except Exception as e:
#             with lock:
#                 results[i] = e

#     if len(modules) > 1:
#         threads = [threading.Thread(target=_worker,
#                                     args=(i, module, input, target,
#                                           kwargs, device),)
#                    for i, (module, input, target, kwargs, device) in
#                    enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]

#         for thread in threads:
#             thread.start()
#         for thread in threads:
#             thread.join()
#     else:
#         _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

#     outputs = []
#     for i in range(len(inputs)):
#         output = results[i]
#         if isinstance(output, Exception):
#             raise output
#         outputs.append(output)
#     return outputs


# ###########################################################################
# # Adapted from Synchronized-BatchNorm-PyTorch.
# # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# #
# class CallbackContext(object):
#     pass


# def execute_replication_callbacks(modules):
#     """
#     Execute an replication callback `__data_parallel_replicate__` on each module created
#     by original replication.
#     The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`
#     Note that, as all modules are isomorphism, we assign each sub-module with a context
#     (shared among multiple copies of this module on different devices).
#     Through this context, different copies can share some information.
#     We guarantee that the callback on the master copy (the first copy) will be called ahead
#     of calling the callback of any slave copies.
#     """
#     master_copy = modules[0]
#     nr_modules = len(list(master_copy.modules()))
#     ctxs = [CallbackContext() for _ in range(nr_modules)]

#     for i, module in enumerate(modules):
#         for j, m in enumerate(module.modules()):
#             if hasattr(m, '__data_parallel_replicate__'):
#                 m.__data_parallel_replicate__(ctxs[j], i)


# def patch_replication_callback(data_parallel):
#     """
#     Monkey-patch an existing `DataParallel` object. Add the replication callback.
#     Useful when you have customized `DataParallel` implementation.
#     Examples:
#         > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
#         > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
#         > patch_replication_callback(sync_bn)
#         # this is equivalent to
#         > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
#         > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
#     """

#     assert isinstance(data_parallel, DataParallel)

#     old_replicate = data_parallel.replicate

#     @functools.wraps(old_replicate)
#     def new_replicate(module, device_ids):
#         modules = old_replicate(module, device_ids)
#         execute_replication_callbacks(modules)
#         return modules

#     data_parallel.replicate = new_replicate