# -*- coding: utf-8 -*-
import argparse
import os
import time
import logging
import sys
import itertools
from utils import LoggerConfig
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite

from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

parser = argparse.ArgumentParser(description="Single Shot MultiBox Detector Training With Pytorch")


parser.add_argument("--dataset_path", help="Dataset directory path")
parser.add_argument("--validation_dataset", help="Dataset directory path")

parser.add_argument(
    "--net",
    default="vgg16-ssd",
    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.",
)

# Params for SGD
parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="initial learning rate")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum value for optim")
parser.add_argument("--weight_decay", default=5e-5, type=float, help="Weight decay for SGD")
parser.add_argument("--gamma", default=0.1, type=float, help="Gamma update for SGD")
parser.add_argument("--base_net_lr", default=None, type=float, help="initial learning rate for base net.")
parser.add_argument(
    "--extra_layers_lr", default=None, type=float, help="initial learning rate for the layers not in base net and prediction heads."
)

# Params for loading pretrained basenet or checkpoints.
parser.add_argument("--base_net", help="Pretrained base model")
parser.add_argument("--pretrained_ssd", help="Pre-trained base model")
parser.add_argument("--resume", default=None, type=str, help="Checkpoint state_dict file to resume training from")

# Train params
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
parser.add_argument("--num_epochs", default=120, type=int, help="the number epochs")
parser.add_argument("--num_workers", default=4, type=int, help="Number of workers used in dataloading")
parser.add_argument("--validation_epochs", default=5, type=int, help="the number epochs")
parser.add_argument("--debug_steps", default=5, type=int, help="Set the debug log output frequency.")
parser.add_argument("--use_cuda", default=True, type=str2bool, help="Use CUDA to train model")

parser.add_argument("--checkpoint_folder", default="models/", help="Directory for saving checkpoint models")

LoggerConfig()

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Usando Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=10, epoch=-1):

    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    # start_time = time.time()
    for i, data in enumerate(loader):
        # logging.warning(f"timer {time.time() - start_time}" )
        # start_time = time.time()
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        confidence, locations = net(images)

        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, "
                + f"Average Loss: {avg_loss:.4f}, "
                + f"Average Regression Loss {avg_reg_loss:.4f}, "
                + f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == "__main__":
    os.makedirs(args.checkpoint_folder, exist_ok=True)
    timer = Timer()

    logging.info(args)
    if args.net == "mb1-ssd":
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == "mb1-ssd-lite":
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")

    dataset = VOCDataset(args.dataset_path, transform=train_transform, target_transform=target_transform)
    label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
    store_labels(label_file, dataset.class_names)
    num_classes = len(dataset.class_names)

    logging.info(f"Stored labels into file {label_file}.")
    start = time.time()
    train_loader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    logging.info(f"DataLoader Time: {time.time() - start}")
    logging.info("Prepare Validation datasets.")
    val_dataset = VOCDataset(args.validation_dataset, transform=test_transform, target_transform=target_transform, is_test=True)

    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)
    logging.info("Build network.")
    net = create_net(num_classes)
    logging.info(f"num classe {num_classes}")
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr

    params = [
        {"params": net.base_net.parameters(), "lr": base_net_lr},
        {"params": itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters()), "lr": extra_layers_lr},
        {"params": itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())},
    ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)
    logging.info(f"################# {DEVICE}")

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, " + f"Extra Layers learning rate: {extra_layers_lr}.")

    scheduler = CosineAnnealingLR(optimizer, args.num_epochs * 2, last_epoch=last_epoch)
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    is_fine_process = False
    for epoch in range(last_epoch + 1, args.num_epochs):

        logging.info(f"start train lr: {scheduler.get_last_lr()}")

        train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        scheduler.step()

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, "
                + f"Validation Loss: {val_loss:.4f}, "
                + f"Validation Regression Loss {val_regression_loss:.4f}, "
                + f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")
