import torch
from utils.models import MyLinear, load_model, save_best_model, get_zeroshot_weights
import time
from utils.parser import parse_args
from utils.logger import set_logger
from utils.testing import test, ood_test
import copy
import torch.nn as nn
from utils.training import set_training_seed, run_zeroshot, train_ce, train_ce_ap
from utils.optimizer import set_optimizer, set_params
from utils.scheduler import build_lr_scheduler
import clip.clip as clip
import datasets


def run_stage1_finetuning(args, logger, model, classifier, train_preprocess, test_preprocess):

    # dataloaders
    imagenet_train, _ = datasets.build_imagenet_few_shot_dataset('imagenet', args, args.data_seed, train_preprocess, 
                                                                 root=args.root, num_shots=args.num_shots, 
                                                                 w_retrival=True if args.data_source == 'fewshot+retrieved' else False)


    train_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_train,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=8,
        pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_test,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_test,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=False)

    test_loader_copy = copy.deepcopy(test_dataloader)

    loss = nn.CrossEntropyLoss()
    params, logit_scale = set_params(args, model, classifier, logger) # depend on method
    optimizer, total_iter = set_optimizer(args, params, train_dataloader)
    scheduler = build_lr_scheduler(optimizer,
                                   lr_scheduler="cosine",
                                   warmup_iter=args.warmup_iter,
                                   max_iter=total_iter,
                                   warmup_type=args.warmup_type,
                                   warmup_lr=args.warmup_lr,
                                   verbose=False)

    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler
    stage1_method = args.method

    if args.model_path:
        load_model(args, logger, model, classifier, test_dataloader)

    # check zeroshot acc
    if args.check_zeroshot or args.method == 'zeroshot':
        logger.info(f"Check Zero-shot Acc ......")
        zs_test_acc = run_zeroshot(args, val_dataloader, model, logger, classifier)

    if args.method == 'zeroshot':
        result_summary = f'{args.dataset},{stage1_method},{args.data_source},{args.cls_init},{args.num_shots},{args.data_seed},{round(zs_test_acc,3)}'
        logger.info(f'{result_summary}')
        exit()

    if args.skip_stage1:
        logger.info(f"Skip stage 1 finetuning.")
        return -1, None, test_loader_copy

    #---------- Training
    if args.method == 'finetune':
        if args.add_ap_stage1:
            best_model, best_head, \
                best_records, best_logit_scale = train_ce_ap(args, logger, loss_logger, model, classifier,
                                                             train_dataloader, val_dataloader, eps=args.eps_stage1)
        else:
            best_model, best_head, \
                best_records, best_logit_scale = train_ce(args, logger, loss_logger, model, classifier, 
                                                          train_dataloader, val_dataloader)
    
    else:
        raise NotImplementedError(f"Method {args.method} not implemented.")


    # print the logit_scale
    logger.info(f"logit_scale: {round(logit_scale.item(), 8)}")
    logger.info(f"best_logit_scale: {round(best_logit_scale.item(), 8)}")

    # test the best model after finetuning
    test_acc = test(dataloader=test_dataloader, model=best_model, classifier=best_head, 
                    test_label_map=[i for i in range(1000)], device=args.device)
    logger.info(f"+++++ Stage 1 Finetuning Test Acc: {round(test_acc, 3)}")

    #----------- save stage 1 best model
    best_model_path = save_best_model(args, best_records, best_model, best_head, best_logit_scale, test_acc, stage=1)
    logger.info(f'Stage 1 Best Model saved to: {best_model_path}')

    #----------- Test ImageNet OOD performance
    logger.info(f"+++++ Test Stage 1 ImageNet OOD ......")
    acc_list = ood_test(args, best_model, best_head, test_preprocess, logger)

    return test_acc, best_model_path



def run_stage2_FSFT(model, classifier, stage1_best_model_path, train_preprocess):

    # reset the flag
    args.epochs = 10
    args.data_source = 'fewshot'
    args.model_path = stage1_best_model_path

    logger.info(f"Run stage 2 few-shot finetuning ......")

    # set the dataloaders
    imagenet_train, _ = datasets.build_imagenet_few_shot_dataset('imagenet', args, args.data_seed, train_preprocess, 
                                                                 root=args.root, num_shots=args.num_shots, 
                                                                 w_retrival=True if args.data_source == 'fewshot+retrieved' else False)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_train,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=8,
        pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_test,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=imagenet_test,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=False)

    load_model(args, logger, model, classifier, test_dataloader)

    # Imporatnt! Need to reset the params, optimizer, scheduler, loss, logit_scale
    loss = nn.CrossEntropyLoss()
    params, logit_scale = set_params(args, model, classifier, logger)
    optimizer, total_iter = set_optimizer(args, params, train_dataloader)
    scheduler = build_lr_scheduler(optimizer,
                                   lr_scheduler="cosine",
                                   warmup_iter=args.warmup_iter,
                                   max_iter=total_iter,
                                   warmup_type=args.warmup_type,
                                   warmup_lr=args.warmup_lr,
                                   verbose=False)

    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler

    #---------- Training
    if args.add_ap_stage2:
        best_model, best_head, \
            best_records, best_logit_scale = train_ce_ap(args, logger, loss_logger, model, classifier,
                                                         train_dataloader, val_dataloader, eps=args.eps_stage2)
    else:
        best_model, best_head, \
            best_records, best_logit_scale = train_ce(args, logger, loss_logger, model, classifier,
                                                      train_dataloader, val_dataloader)

    # test the best model after FSFT
    test_acc = test(dataloader=test_dataloader, model=best_model, classifier=best_head, 
                    test_label_map=[i for i in range(1000)], device=args.device)
    logger.info(f"+++++ Stage 2 FSFT Test Acc: {round(test_acc, 3)}")

    #----------- save stage 2 best model
    best_model_path = save_best_model(args, best_records, best_model, best_head, 
                                      logit_scale, test_acc, stage=2)
    logger.info(f'Stage 2 FSFT Best Model saved to: {best_model_path}')


    #----------- Test ImageNet OOD performance
    logger.info(f"+++++ Test Stage 1 ImageNet OOD ......")
    acc_list = ood_test(args, best_model, best_head, test_preprocess, logger)


    return test_acc, best_model_path



if __name__ == '__main__':
    # import argparse
    program_start = time.time()
    args = parse_args()
    # args = argparse.Namespace(add_adverb_stage1=False, add_adverb_stage2=False, alpha=0.5, attentive_name='c-name', attentive_threshold=0.85, bsz=64, check_zeroshot=False, ckpt_path='output/FFT_w_RA/output_imagenet/imagenet_finetune_fewshot+retrieved_text_16shots_seed1_10eps/model_ckpts', cls_init='text', cmo_alpha=1.0, cutmix_beta=1.0, data_source='fewshot+retrieved', database='LAION400M', dataset='imagenet', dataset_path='/home/whx/Dataset/ImageNet/ImageNet-1k/', dataset_root='data/imagenet', dataset_wd=1.0, device='cuda', early_stop=True, epochs=10, eps_stage1=0, eps_stage2=0, fewshot_data=[['fewshot16_seed1.txt'], ['/home/whx/Dataset/ImageNet/ImageNet-1k/']], fewshot_ratio=0.5, fewshot_weight=1.0, focal_alpha=1.0, focal_gamma=2.0, folder='output/FFT_w_RA/output_imagenet', freeze_visual=False, ft_topk_blks=-1, lambda_u=1.0, locked_text=False, log_mode='both', loss_name='CE', lr_backbone=1e-06, lr_classifier=0.001, lr_projector=None, method='finetune', mix_prob=0.5, mixup_alpha=1.0, model_cfg='vitb16_clip', model_path=None, mu=1, no_tau=True, no_wsft=False, num_workers=8, optim='AdamW', output_dir='output/FFT_w_RA/output_imagenet/imagenet_finetune_fewshot+retrieved_text_16shots_seed1_10eps', pre_extracted=False, prefix=None, prompt_name='name', recal_fea=False, recal_prompt=False, retrieval_data=[['T2T500.txt'], ['/home/whx/Dataset/retrieved/']], retrieval_split='T2T500.txt', retrieved_path='/home/whx/Dataset/retrieved/', save_ckpt=False, save_freq=10, seed=1, shots=16, skip_stage1=False, skip_stage2=False, stage1_model_path=None, start_validation=0, stop_epochs=200, tau_norm=True, temperature=0.07, test_imagenet_ood=True, test_split=[['test.txt'], ['/home/whx/Dataset/ImageNet/ImageNet-1k/']], threshold=0.95, train_split=[['fewshot16_seed1.txt', 'T2T500.txt'], ['/home/whx/Dataset/ImageNet/ImageNet-1k/', '/home/whx/Dataset/retrieved/']], training_seed=1, unlabeled_split='u_train_in_oracle.txt', use_attribute=False, utrain=None, val_split=[['test.txt'], ['/home/whx/Dataset/ImageNet/ImageNet-1k/']], warmup_iter=18, warmup_lr=1e-08, wd=0.01, zeroshot_only=False)
    logger, loss_logger = set_logger(args)
    args.logger = logger
    args.loss_logger = loss_logger
    set_training_seed(args)

    # load model
    model, train_preprocess, test_preprocess = clip.load('ViT-B/16', jit=False)
    tokenizer = clip.tokenize


    # Prepare test dataset
    # ID testset
    imagenet_test, text_name = datasets.build_imagenet_dataset('imagenet', 'test', test_preprocess, root=args.root)

    # set classifier head
    num_classes = 1000
    num_features = 512
    logit_scale = model.logit_scale

    if args.cls_init == 'openai':
        with torch.no_grad():
            template = 'openai_imagenet_template'
            logger.info(f"Getting zeroshot weights from {args.cls_init}.")
            zeroshot_weights = get_zeroshot_weights(text_name, template, model, tokenizer, logit_scale)
        logger.info(f"Initialize classifier head with text embedding. weights.shape: {zeroshot_weights.shape}")
        classifier = MyLinear(input_dim=num_features, num_classes=num_classes, bias=False)
        classifier._init_weights(zeroshot_weights)

    elif args.cls_init == 'random':
        logger.info(f'Initialized classifier head with random weights.')
        classifier = MyLinear(input_dim=num_features, num_classes=num_classes, bias=False)

    model.to(args.device)
    classifier.to(args.device)

    #---------- run finetuning for stage 1
    stage1_acc, stage1_best_model_path = run_stage1_finetuning(args, logger, model, classifier, train_preprocess, test_preprocess)
    stage1_method = args.method # record method here, as in stage 2 method will be updated to probing

    # replace the stage1_best_model_path to run stage 2 for certrain checkpoints
    if args.skip_stage1:
        stage1_best_model_path = args.stage1_model_path


    #---------- run FSFT for stage 2
    if not args.skip_stage2:
        stage2_fsft_acc, stage2_best_model_path = run_stage2_FSFT(model, classifier, stage1_best_model_path, train_preprocess)

    else:
        logger.info(f"Skip stage 2 FSFT.")
        stage2_fsft_acc = -1
        stage2_best_model_path = 'None'

    loss_logger.close()
    program_end = time.time()
    logger.info(f"Total time: {round((program_end-program_start)/60, 1)} mins.")


    result_summary = f'{args.dataset},{stage1_method},{args.data_source},{args.cls_init},'\
                     f'{args.num_shots},{args.data_seed},{round(stage1_acc,3)},'\
                     f'{round(stage2_fsft_acc,3)}'
    logger.info(f'{result_summary}')
    print(f'{result_summary}')