import torch

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, pre_extracted_path=None, device='cuda:0'):

        if pre_extracted_path is None:
            raise NotImplementedError
        else:
            pre_extracted_path = pre_extracted_path

        self.dataset = torch.load(pre_extracted_path, map_location=device)
        self.input_tensor = self.dataset['image_features']
        self.label_tensor = self.dataset['labels']

    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.input_tensor.size(0)