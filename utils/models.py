import torch
from torch import nn
from utils.testing import test
import templates


class MyLinear(nn.Module):
    def __init__(self, input_dim=512, num_classes=1000, bias = False):
        super(MyLinear, self).__init__()

        self.linear = nn.Linear(input_dim, num_classes, bias=bias)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.linear(x)

        return x

    def _init_weights(self, weights):
        # Initialize the weights of the linear layer with the given weights
        self.linear.weight = nn.Parameter(weights.clone())


def get_zeroshot_weights(text_name, template, model, tokenizer, logit_scale):
    zeroshot_weights = []
    template = getattr(templates, template)
    for classname in text_name:
        texts = []
        for t in template:
            texts.append(t(classname))
        texts = tokenizer(texts).cuda()
        embeddings = model.encode_text(texts) #(80, dim)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        
        embeddings = embeddings.mean(dim=0) #(dim)
        embeddings /= embeddings.norm()
    
        zeroshot_weights.append(embeddings)
    
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0) #(1000, dim)
    zeroshot_weights *= logit_scale.exp()
    
    return zeroshot_weights


def save_model_ckpt(args, best_records, model, classifier_head, optimizer, scheduler, logit_scale,
                    val_acc=-1, epoch=-1, num_iter=-1):

    model_path = f'{args.ckpt_path}/model_bs{args.bsz}_lr-cls{args.lr_classifier}_lr-bkb{args.lr_backbone}_wd{args.wd}_epoch_{epoch}_iter_{num_iter}.pth'

    state = {}
    state['best_val_acc'] = best_records['best_val_acc']
    state['best_epoch'] = best_records['best_epoch']
    state['best_iter'] = best_records['best_iter']
    state['val_acc'] = val_acc
    state['epoch'] = epoch
    state['num_iter'] = num_iter
    state['clip'] = model.state_dict()
    state['head'] = classifier_head.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['scheduler'] = scheduler.state_dict()
    state['logit_scale'] = logit_scale

    torch.save(state, model_path)

    return model_path


def save_best_model(args, best_records, best_model, best_head, best_logit_scale, test_acc, stage=1):

    best_epoch = best_records['best_epoch']
    best_iter = best_records['best_iter']
    model_path = f'{args.output_dir}/stage{stage}_model_best-epoch_{best_epoch}_best.pth'

    state = {}
    state['best_val_acc'] = best_records['best_val_acc']
    state['best_epoch'] = best_records['best_epoch']
    state['best_iter'] = best_records['best_iter']
    state['clip'] = best_model.state_dict()
    state['head'] = best_head.state_dict()
    state['logit_scale'] = best_logit_scale
    state['test_acc'] = round(test_acc, 3)

    torch.save(state, model_path)

    return model_path


def load_model(args, logger, model, classifier, test_dataloader=None):

    logger.info(f'Loading model from: {args.model_path}')
    ckpt = torch.load(args.model_path)

    #----- load normal model
    model.load_state_dict(ckpt['clip'])
    classifier.load_state_dict(ckpt['head'])

    logger.info(f'ckpt[test_acc]: {ckpt["test_acc"]}')

    if test_dataloader is not None:
        model_test_acc = test(dataloader=test_dataloader, model=model, classifier=classifier,
                               test_label_map=[i for i in range(1000)], device=args.device)
        logger.info(f"Loaded Model Test Acc: {round(model_test_acc, 3)}")


