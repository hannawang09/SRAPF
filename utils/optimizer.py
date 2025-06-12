from torch import optim
import torch

"""
partially borrowed from: https://github.com/linzhiqiu/cross_modal_adaptation/blob/main/engine/optimizer/scheduler.py
"""


def set_optimizer(args, params, train_loader):

    optimizer = get_optimizer(params, optim_type=args.optim, wd=args.wd)
    total_iter = len(train_loader) * args.epochs

    return optimizer, total_iter


def set_params(args, model, classifier, logger):

    if args.method == "zeroshot":
        logger.info('zeroshot only.')
        for param in model.parameters():
            param.requires_grad = False
        params_classifier = [{'params': classifier.parameters(), 'lr': args.lr_classifier}]
        params = params_classifier # place holder
        logit_scale = torch.tensor([4.60517]).to(device=args.device)

    elif args.method == "finetune":

        for param in model.parameters():
            param.requires_grad = False

        logger.info(f'Total visual transformer blocks: {len(model.visual.transformer.resblocks)}')
        assert args.ft_topk_blks <= len(model.visual.transformer.resblocks), \
            "Requested more blocks than available"

        if args.ft_topk_blks == -1:
            logger.info('Finetune all blocks of the visual transformer.')
            for param in model.visual.parameters():
                param.requires_grad = True
        else:
            logger.info(f'Finetune top-{args.ft_topk_blks} blocks of the visual transformer.')
            for blk in model.visual.transformer.resblocks[-args.ft_topk_blks:]:
                for param in blk.parameters():
                    param.requires_grad = True

            for param in model.visual.ln_post.parameters():
                param.requires_grad = True
            model.visual.proj.requires_grad = True


        # only add the params which require gradients
        params_visual = [{'params': [p for p in model.visual.parameters() if p.requires_grad], 'lr': args.lr_backbone}]
        params_classifier = [{'params': classifier.parameters(), 'lr': args.lr_classifier}]
        params = params_classifier + params_visual


        #---------- learn temp
        # logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.temperature)) # ln(1/0.07)=2.65926
        # logit_scale = nn.Parameter(torch.ones([]) * 4.6052) # use CLIP.logit as initial value, exp(4.6052) = 100 = 1/0.01, where 0.01 is the temperature
        # params.append({'params': [logit_scale], 'lr': args.lr_classifier})

        #---------- not learning temp
        logit_scale = torch.zeros([]).to(device=args.device)

    else:
        raise NotImplementedError(f'Method {args.method} not implemented.')

    args.logit_scale = logit_scale
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {n_trainable}")

    return params, logit_scale



def get_optimizer(params, optim_type, wd,):
    if optim_type == 'SGD':
        for param in params:
            param['momentum'] = 0.9
            param['weight_decay'] = wd
        return optim.SGD(params)

    elif optim_type == 'AdamW':
        for param in params:
            param['betas'] = (0.9,0.999)
            param['weight_decay'] = wd
        return optim.AdamW(params)


