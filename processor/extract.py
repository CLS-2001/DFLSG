import time
from utils.meter import AverageMeter
from collections import OrderedDict
import torch
from utils import to_torch
import logging

def extract_img_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    with torch.no_grad():
        outputs = model.encode_image(inputs)
    return outputs


def extract_txt_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    with torch.no_grad():
        outputs = model.encode_text(inputs)
    return outputs

@torch.no_grad()
def extract_features(model, cluster_img_loader,cluster_txt_loader, print_freq=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("DFLSG.train")
    img_batch_time = AverageMeter()
    img_data_time = AverageMeter()

    image_features = []
    text_features = []

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames) in enumerate(cluster_img_loader):
            img_data_time.update(time.time() - end)
            imgs = imgs.to(device)
            outputs = extract_img_feature(model, imgs)
            for fname, output in zip(fnames, outputs):
                image_features.append(output)

            img_batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logger.info('Extract Imgs Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(cluster_img_loader),
                              img_batch_time.val, img_batch_time.avg,
                              img_data_time.val, img_data_time.avg))

        for j, captions in enumerate(cluster_txt_loader):
            captions = captions.to(device)
            outputs = extract_txt_feature(model, captions)
            for output in outputs:
                text_features.append(output)


    return image_features,text_features