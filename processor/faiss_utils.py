import os
import numpy as np
import faiss
import torch

#数据类型为 float32 且连续存储的 tensor转换为一个 SWIG 指针
def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous() #检查x是否是连续的 因为只有连续的内存块才能被有效地访问和操作
    assert x.dtype == torch.float32 #检查数据类型
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)#它使用了 x.storage().data_ptr()获取 tensor的存储数据指针，然后使用 x.storage_offset() * 4 计算出tensor在存储中的起始偏移量（考虑到每个元素的大小为4字节，因为这通常是一个32位浮点数的大小）。

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype

    return faiss.cast_integer_to_idx_t_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I

def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()    # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    #将 FloatTensor 类型的对象 xq 的指针赋值给变量 xq_ptr。这个指针可以用于在高级语言中引用和操作 xq 对象
    xq_ptr = swig_ptr_from_FloatTensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_FloatTensor(xb)#创建xb的指针

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)#创建一个新张量D【12936，30】
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)#创建一个新张量I【12936，30】
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)

    faiss.bruteForceKnn(res, metric,  #bruteForceKnn（）用于执行基于暴力匹配的 k-近邻搜索。这种方法将对所有数据点执行比较，以找到与查询点最近的 k 个邻居
                xb_ptr, xb_row_major, nb, #metric距离度量 xb_ptr,xb_row_major,nb:这些参数指定了第一个数据集的内存位置、是否按行存储以及数据集的大小
                xq_ptr, xq_row_major, nq,#这些参数指定了查询点的内存位置、是否按行存储以及查询点的大小
                d, k, D_ptr, I_ptr) #d: 数据点的特征维度 k: 要返回的最近邻居的数量 D_ptr,I_ptr:这两个指针用于存储搜索结果。D_ptr 指向一个存储距离的数组，而 I_ptr 指向一个存储邻居索引的数组。

    return D, I #D返回的是30个最近邻的距离 I返回的是30个最近邻的索引

def index_init_gpu(ngpus, feat_dim):
    flat_config = []
    for i in range(ngpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    indexes = [faiss.GpuIndexFlatL2(res[i], feat_dim, flat_config[i]) for i in range(ngpus)]
    index = faiss.IndexShards(feat_dim)
    for sub_index in indexes:
        index.add_shard(sub_index)
    index.reset()
    return index

def index_init_cpu(feat_dim):
    return faiss.IndexFlatL2(feat_dim)
