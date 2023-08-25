from torch.utils.data import Dataset
import torch

class mnist_dataset_with_z(Dataset):

    def __init__(self, mnist_dset):
        
        self.image = mnist_dset.data / 255.
        self.label = mnist_dset.train_labels
        self.z_indices = torch.arange(len(self.image))
    
    def __getitem__(self, index):
        
        image   = self.image[index, :, :].unsqueeze(dim=0)
        label   = self.label[index, ]
        z_idx   = self.z_indices[index, ]

        return image, label, z_idx
    
    def __len__(self):
        
        return self.image.shape[0]

    def collate_fn(self, batch):
        images  = list()
        labels  = list()
        z_indices = list()

        for b in batch:
            images.append(b[0])
            labels.append(b[1])
            z_indices.append(b[2])

        images  = torch.stack(images, dim=0)
        labels  = torch.stack(labels, dim=0).squeeze()
        z_indices = torch.stack(z_indices, dim=0).squeeze()

        return images, labels, z_indices

