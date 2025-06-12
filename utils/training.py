import random
import torch
import numpy as np
import copy
from utils.models import save_model_ckpt
from utils.testing import test
import psutil
import gc
import os
from utils.attack import PGD


def set_training_seed(args):

    # set the seed for training
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)

    # this is critical for reproducibility for ResNet50 models
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_zeroshot(args, test_dataloader, model, logger, classifier):

    zs_test_acc = test(dataloader=test_dataloader, model=model, classifier=classifier,
                       test_label_map=[i for i in range(1000)], device=args.device)
    logger.info(f"+++++ Zero-shot Test Acc: {round(zs_test_acc, 3)}")

    return zs_test_acc


def train_ce(args, logger, loss_logger, model, classifier, train_dataloader, val_dataloader):
    """ Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier"""

    logger.info(f"Start standard finetuning ......")

    model.train()
    classifier.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    acc_v2, acc_s, acc_a, acc_r, ood_avg = -1, -1, -1, -1, -1

    for epoch in range(1, args.epochs+1):

        train_loss_sum = 0
        for inputs, labels in train_dataloader:
            num_iter += 1

            if num_iter % 1000 == 0:
                logger.info(f"Processing epoch {epoch}, iter {num_iter} ...")

            images = inputs.to(args.device)
            labels = labels.to(args.device)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier(image_feature)
            logits = logits * logit_scale.exp()

            total_loss = loss(logits, labels)

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step() # update learning rate for each iteration

        logger.info(f'done epoch: {epoch}, testing...')

        # watch the GPU, CPU memory usage
        logger.info(f'GPU Memory Usage after epoch {epoch}: {torch.cuda.memory_allocated(args.device) / (1024.0 ** 3):.2f} GB') # in GB
        logger.info(f'CPU Memory Usage after epoch {epoch}: {psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 3):.2f} GB') # in GB

        torch.cuda.empty_cache()  # Clears unused GPU memory
        gc.collect()              # Clears unused CPU memory

        logger.info(f'Garbage collected: GPU Memory Usage after epoch {epoch}: {torch.cuda.memory_allocated(args.device) / (1024.0 ** 3):.2f} GB') # in GB
        logger.info(f'Garbage collected: CPU Memory Usage after epoch {epoch}: {psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 3):.2f} GB') # in GB


        # validate
        if args.early_stop or epoch == args.epochs:
            val_acc = test(dataloader=val_dataloader, model=model, classifier=classifier, 
                           test_label_map=[i for i in range(1000)], device=args.device)

            # check if val acc has improved, here i use the val_acc rather than val_loss
            if  val_acc >= best_val_acc:
                best_val_acc = val_acc
                # best_logit_scale = copy.deepcopy(logit_scale)
                best_logit_scale = torch.zeros_like(logit_scale).to(args.device) # set the logit_scale to 0
                best_epoch = epoch
                best_iter = num_iter
                best_head = copy.deepcopy(classifier)
                best_model = copy.deepcopy(model)

                # save into the best_records
                best_records['best_val_acc'] = best_val_acc
                best_records['best_logit_scale'] = best_logit_scale
                best_records['best_epoch'] = best_epoch
                best_records['best_iter'] = best_iter

        # test
        if args.early_stop or epoch == args.epochs:
            test_acc = val_acc

        train_loss_avg = train_loss_sum / len(train_dataloader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},'
                          f'{round(test_acc, 6)}, {round(acc_v2, 6)}, {round(acc_s, 6)}, {round(acc_a, 6)}, {round(acc_r, 6)}, {round(ood_avg, 6)}\n')
        loss_logger.flush()
        logger.info(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, "
                    f"logit_scale: {round(logit_scale.item(), 6)} Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}, OOD Avg: {round(ood_avg, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records, model, classifier, optimizer, 
                                         scheduler, logit_scale, val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')


    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_ce_ap(args, logger, loss_logger, model, classifier, train_dataloader, val_dataloader, eps):
    """ Train the model with Cross-Entropy Loss and Adversarial Perturbation, finetuning visual encoder and classifier"""

    logger.info(f"Start standard finetuning with adversarial perturbation ......")
    attack = PGD(model, classifier, eps=eps, alpha=eps / 10)

    model.train()
    classifier.train()

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    acc_v2, acc_s, acc_a, acc_r, ood_avg = -1, -1, -1, -1, -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        classifier.train()

        train_loss_sum = 0
        for inputs, labels in train_dataloader:
            num_iter += 1

            if num_iter % 1000 == 0:
                logger.info(f"Processing epoch {epoch}, iter {num_iter} ...")

            images = inputs.to(args.device)
            labels = labels.to(args.device)

            if args.ft_topk_blks == -1:
                clean_features = model.encode_image(images)

                adv_token = attack.forward_token(images, model.visual.class_embedding, labels)
                adv_features = model.encode_clstoken(images, adv_token)
            else:
                # use forzen blocks to extract midfeatures
                image_midfeatures = model.get_midfeatures(images, k=args.ft_topk_blks)
                clean_features = model.get_features(image_midfeatures, k=args.ft_topk_blks)

                adv_midfeatures = attack(image_midfeatures, labels, k=args.ft_topk_blks)
                adv_features = model.get_features(adv_midfeatures, k=args.ft_topk_blks)

            # normalized features
            clean_features = clean_features / clean_features.norm(dim=-1, keepdim=True)
            adv_features = adv_features / adv_features.norm(dim=-1, keepdim=True)

            clean_logits = classifier(clean_features)
            clean_logits = clean_logits * logit_scale.exp()
            adv_logits = classifier(adv_features)
            adv_logits = adv_logits * logit_scale.exp()

            clean_loss = loss(clean_logits, labels)
            adv_loss = loss(adv_logits, labels)
            total_loss = clean_loss + adv_loss

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()  # update learning rate for each iteration


        logger.info(f'done epoch: {epoch}, testing...')

        # watch the GPU, CPU memory usage
        logger.info(
            f'GPU Memory Usage after epoch {epoch}: {torch.cuda.memory_allocated(args.device) / (1024.0 ** 3):.2f} GB')  # in GB
        logger.info(
            f'CPU Memory Usage after epoch {epoch}: {psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 3):.2f} GB')  # in GB

        torch.cuda.empty_cache()  # Clears unused GPU memory
        gc.collect()  # Clears unused CPU memory

        logger.info(
            f'Garbage collected: GPU Memory Usage after epoch {epoch}: {torch.cuda.memory_allocated(args.device) / (1024.0 ** 3):.2f} GB')  # in GB
        logger.info(
            f'Garbage collected: CPU Memory Usage after epoch {epoch}: {psutil.Process(os.getpid()).memory_info().rss / (1024.0 ** 3):.2f} GB')  # in GB

        # validate
        if args.early_stop or epoch == args.epochs:
            val_acc = test(dataloader=val_dataloader, model=model, classifier=classifier, 
                           test_label_map=[i for i in range(1000)], device=args.device)

            # check if val acc has improved, here i use the val_acc rather than val_loss
            if  val_acc >= best_val_acc:
                best_val_acc = val_acc
                # best_logit_scale = copy.deepcopy(logit_scale)
                best_logit_scale = torch.zeros_like(logit_scale).to(args.device) # set the logit_scale to 0
                best_epoch = epoch
                best_iter = num_iter
                best_head = copy.deepcopy(classifier)
                best_model = copy.deepcopy(model)

                # save into the best_records
                best_records['best_val_acc'] = best_val_acc
                best_records['best_logit_scale'] = best_logit_scale
                best_records['best_epoch'] = best_epoch
                best_records['best_iter'] = best_iter

        # test
        if args.early_stop or epoch == args.epochs:
            test_acc = val_acc

        train_loss_avg = train_loss_sum / len(train_dataloader)
        loss_logger.write(f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)},'
                          f'{round(test_acc, 6)}, {round(acc_v2, 6)}, {round(acc_s, 6)}, {round(acc_a, 6)}, {round(acc_r, 6)}, {round(ood_avg, 6)}\n')
        loss_logger.flush()
        logger.info(
            f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, "
            f"logit_scale: {round(logit_scale.item(), 6)} Val Acc: {round(val_acc, 3)}, Test Acc: {round(test_acc, 3)}, OOD Avg: {round(ood_avg, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records, model, classifier, optimizer, 
                                         scheduler, logit_scale, val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')


    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale