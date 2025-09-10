import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from datasets.build import build_test_dataloader, build_cluster_dataloader, build_train_dataloader
from processor.extract import extract_features
from processor.faiss_rerank import compute_jaccard_distance,compute_euclidean_dist
from sklearn.cluster import DBSCAN
import collections
import random
import torch.nn.functional as F
from model.cm import ClusterMemory
from itertools import cycle
from torch.nn.functional import cosine_similarity
from .filter import data_filter
import os
import pickle

def do_train(start_epoch, args, model, evaluator, optimizer,
             scheduler, checkpointer,  train_data, caption_data):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("DFLSG.train")
    #logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "mem_loss": AverageMeter(),
        "tri_inter_loss": AverageMeter(),
        "tri_intra_loss": AverageMeter()
    }
    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        logger.info('==> Create pseudo labels for unlabeled data')
        cl_img_loader, cl_txt_loader = build_cluster_dataloader(args, train_data)

        image_features, text_features = extract_features(model, cl_img_loader, cl_txt_loader,  print_freq=10)
        image_features = torch.stack(image_features, dim=0)
        text_features = torch.stack(text_features,dim=0)
        image_rerank_dist = compute_jaccard_distance(image_features, k1=30, k2=6)

        if epoch == 1:
            # DBSCAN cluster
            img_eps = 0.5
            img_cluster = DBSCAN(eps=img_eps, min_samples=4, metric='precomputed', n_jobs=-1)

        # select & cluster images as training set of this epochs
        pseudo_labels = img_cluster.fit_predict(image_rerank_dist)
        # save pseudo labels
        # save_dict = {
        #     'epoch': epoch,
        #     'pseudo_labels': pseudo_labels,
        #     'image_paths': cl_img_loader.dataset.img_paths,
        # }
        # save_path = os.path.join(args.output_dir, f"pseudo_labels_epoch_{epoch}.pkl")
        # with open(save_path, 'wb') as f:
        #     pickle.dump(save_dict, f)
        #
        # logger.info(f"Saved pseudo labels to {save_path}")

        image_num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        @torch.no_grad()
        def generate_cluster_features(labels, image_features, text_features):
            centers = collections.defaultdict(list)
            text_centers = collections.defaultdict(list)
            fake_label = []
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(image_features[i])
                fake_label.append(label)
                text_centers[labels[i]].append(text_features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            similarities = []
            for idx in sorted(text_centers.keys()):
                text_feats = torch.stack(text_centers[idx], dim=0)
                image_center = centers[idx]
                similarities.append(cosine_similarity(text_feats, image_center.unsqueeze(0), dim=-1))

            return centers, similarities

        image_cluster_features, similarities = generate_cluster_features(pseudo_labels, image_features, text_features)
        del cl_img_loader, image_features

        # Create memory
        image_memory = ClusterMemory(model.embed_dim, image_num_cluster, temp=args.temperature,
                               momentum=args.memory_update, use_hard=True).cuda()
        image_memory.features = F.normalize(image_cluster_features, dim=1).cuda()

        caption_dict_clean  = data_filter(caption_data, pseudo_labels, similarities, args.beta)
        image_path = []
        captions = []
        pids = []
        for label, image_text_pairs in caption_dict_clean.items():
            for caption_dict in image_text_pairs:
                fname, caption = next(iter(caption_dict.items()))

                image_path.append(args.root_dir + args.target_dataset_name + '/imgs/' + fname)
                captions.append(caption)
                pids.append(label.item())

        target_data = [(pid, img_path, caption) for pid, img_path, caption in zip(pids, image_path, captions)]
        logger.info('==> Statistics for epoch {}: images with {} clusters '.format(epoch, image_num_cluster))
        logger.info('==> Constructing Dataloader')
        target_data_loader = build_train_dataloader(args, target_data)

        logger.info('==> Start training')
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, target_batch in enumerate(target_data_loader):

            target_batch = {k: v.to(device) for k, v in target_batch.items()}
            ret = model(target_batch, image_memory)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = target_batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['mem_loss'].update(ret.get('mem_loss', 0), batch_size)
            meters['tri_inter_loss'].update(ret.get('tri_inter_loss', 0), batch_size)
            meters['tri_intra_loss'].update(ret.get('tri_intra_loss', 0), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(target_data_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

            n_iter += 1

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        args.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("DFLSG.test")
    logger.info("Enter inferencing")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
