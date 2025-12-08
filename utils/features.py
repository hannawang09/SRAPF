import os
import torch
import datasets


def extract_features(model, dataloader, is_encoder):

    img_feats_lst, labels_lst = [], []

    for data in dataloader:
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.long()

        model.eval()
        with torch.no_grad():
            if is_encoder:
                img_feats = model.encode_image(imgs)
                img_feats /= img_feats.norm(dim=-1, keepdim=True)
            else:
                img_feats = model(imgs)

        img_feats_lst.append(img_feats.cpu())
        labels_lst.append(labels.cpu())

    img_feats_store = torch.cat(img_feats_lst, dim=0)
    labels_store = torch.cat(labels_lst, dim=0)

    result = {'image_features': img_feats_store,
                'labels': labels_store}

    return result



def pre_extract_feature(args, logger, model, dataloader_dict, is_encoder=True):

    pre_extract_train_fea_path = f'pre_extracted/{args.dataset}_{args.method}_{args.model_cfg}_{args.data_seed}_train_features.pth'
    if args.val_type == 'fewshot+retrieved':
        pre_extract_val_fea_path_ID = f'pre_extracted/{args.dataset}_{args.method}_{args.model_cfg}_{args.data_seed}_val_features_ID.pth'
        pre_extract_val_fea_path_RT = f'pre_extracted/{args.dataset}_{args.method}_{args.model_cfg}_{args.data_seed}_val_features_RT.pth'
        pre_extract_val_fea_path = (pre_extract_val_fea_path_ID, pre_extract_val_fea_path_RT)
    else:
        pre_extract_val_fea_path = f'pre_extracted/{args.dataset}_{args.method}_{args.model_cfg}_{args.data_seed}_val_features.pth'
    pre_extract_test_fea_path = f'pre_extracted/{args.dataset}_{args.method}_{args.model_cfg}_test_features.pth'
    BATCH_SIZE = 512 # this may cause OOM, reduce it if necessary

    if args.recal_fea:
        logger.info(f'Extracting val features ...')
            
        if args.val_type == 'fewshot+retrieved':
            val_dataloader_ID = dataloader_dict['val_dataloader'][0]
            val_dataloader_RT = dataloader_dict['val_dataloader'][1]
            val_dataloader = (val_dataloader_ID, val_dataloader_RT)
        else:
            val_dataloader = dataloader_dict['val_dataloader']
        
        if args.val_type == 'fewshot+retrieved':
            val_features_ID = extract_features(model, dataloader=val_dataloader[0], is_encoder=is_encoder)
            torch.save(val_features_ID, pre_extract_val_fea_path[0])
            logger.info(f'Extracted val features to {pre_extract_val_fea_path[0]}')

            val_features_RT = extract_features(model, dataloader=val_dataloader[1], is_encoder=is_encoder)
            torch.save(val_features_RT, pre_extract_val_fea_path[1])
            logger.info(f'Extracted val features to {pre_extract_val_fea_path[1]}')
            
        else:
            val_features = extract_features(model, dataloader=val_dataloader, is_encoder=is_encoder)
            torch.save(val_features, pre_extract_val_fea_path)
            logger.info(f'Extracted val features to {pre_extract_val_fea_path}')


        logger.info(f'Extracting test features ...')
        test_dataloader = dataloader_dict['test_dataloader']
        test_features = extract_features(model, dataloader=test_dataloader, is_encoder=is_encoder)
        torch.save(test_features, pre_extract_test_fea_path)
        logger.info(f'Extracted test features to {pre_extract_test_fea_path}')


        logger.info(f'Extracting train features ...')
        train_dataloader = dataloader_dict['train_dataloader']
        train_features = extract_features(model, dataloader=train_dataloader, is_encoder=is_encoder)
        torch.save(train_features, pre_extract_train_fea_path)
        logger.info(f'Extracted train features to {pre_extract_train_fea_path}')


    return pre_extract_train_fea_path, pre_extract_val_fea_path, pre_extract_test_fea_path


def get_dataloader_preextracted(args, logger, pre_extract_train_fea_path, pre_extract_val_fea_path,
                                pre_extract_test_fea_path, device):

    train_dataset = datasets.TensorDataset(pre_extracted_path=pre_extract_train_fea_path, device=device)
    logger.info(f'Loaded pre-extracted train features from: {pre_extract_train_fea_path}')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, drop_last=False, num_workers=0)

    if args.val_type == 'fewshot+retrieved':
        val_dataset_ID = datasets.TensorDataset(pre_extracted_path=pre_extract_val_fea_path[0], device=device)
        logger.info(f'Loaded pre-extracted val ID features from: {pre_extract_val_fea_path[0]}')
        val_dataloader_ID = torch.utils.data.DataLoader(val_dataset_ID, batch_size=args.bsz, shuffle=False, drop_last=False, num_workers=0)

        val_dataset_RT = datasets.TensorDataset(pre_extracted_path=pre_extract_val_fea_path[1], device=device)
        logger.info(f'Loaded pre-extracted val RT features from: {pre_extract_val_fea_path[1]}')
        val_dataloader_RT = torch.utils.data.DataLoader(val_dataset_RT, batch_size=args.bsz, shuffle=False, drop_last=False, num_workers=0)

        val_dataloader = (val_dataloader_ID, val_dataloader_RT)
    else:
        val_dataset = datasets.TensorDataset(pre_extracted_path=pre_extract_val_fea_path, device=device)
        logger.info(f'Loaded pre-extracted val features from: {pre_extract_val_fea_path}')
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bsz, shuffle=False, drop_last=False, num_workers=0)

    test_dataset = datasets.TensorDataset(pre_extracted_path=pre_extract_test_fea_path, device=device)
    logger.info(f'Loaded pre-extracted test features from: {pre_extract_test_fea_path}')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bsz, shuffle=False, drop_last=False, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader