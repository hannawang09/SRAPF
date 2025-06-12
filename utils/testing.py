import torch
import numpy as np
import datasets

def logit_adaption(pred_logit, label_map):
    """
    Map logit to imageNet class indexs(1000)

    Returns:
        pred_logit in imageNet class index(1000).
    """
    assert pred_logit.shape[1]==1000, "Error! The shape of logits doesn't match."

    pred_logit=pred_logit[:,label_map]
    return pred_logit


def test(dataloader, model, classifier, test_label_map=None, device='cuda'):
        model.eval()
        classifier.eval()
        with torch.no_grad():
            targets_list = []
            preds_list = []
            for idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                image_features = model.encode_image(inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                outputs = classifier(image_features)
                outputs = logit_adaption(outputs, test_label_map)

                targets_list.append(targets.detach().cpu().numpy())
                preds_list.append(outputs.detach().cpu().numpy())

        targets_list = np.hstack(targets_list)
        preds_list = np.vstack(preds_list)
        preds_list = torch.tensor(preds_list)

        test_acc = (torch.softmax(preds_list, dim=1).argmax(1).numpy() == targets_list).mean()

        model.train()
        classifier.train()
        
        return test_acc * 100


def ood_test(args, model, classifier, test_preprocess, logger):

        # OOD testsets
        imagenet_a_test, _ = datasets.build_imagenet_dataset('imagenet_a', 'test', test_preprocess, root=args.root)
        imagenet_r_test, _ = datasets.build_imagenet_dataset('imagenet_r', 'test', test_preprocess, root=args.root)
        imagenet_sketch_test, _ = datasets.build_imagenet_dataset('imagenet_sketch', 'test', test_preprocess, root=args.root)
        imagenetv2_test, _ = datasets.build_imagenet_dataset('imagenetv2', 'test', test_preprocess, root=args.root)

        acc_list = []
        for test_dataset in [imagenetv2_test, imagenet_sketch_test, imagenet_a_test, imagenet_r_test]:

            dataset_name = test_dataset.dataset_name
            test_dataloader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=4)
            test_label_map = test_dataset.label_map

            test_acc = test(test_dataloader, model, classifier, test_label_map)

            acc_list.append(test_acc)
            logger.info(f'{dataset_name}, Test Acc: {round(test_acc, 3)}')

        avg_ood = np.mean(acc_list) # Average OOD test accuracy excluding the ID dataset
        logger.info(f'Average OOD Test Acc: {round(avg_ood, 3)}')
        acc_list.append(avg_ood)

        return acc_list
