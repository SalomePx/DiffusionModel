from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Cifar10:
    def __init__(self, args):

        transform = self.transform_tensor()
        train_set = datasets.CIFAR10(root='./process_data', train=True, transform=transform, download=True)
        test_set = datasets.CIFAR10(root='./process_data', train=False, transform=transform, download=True)

        self.batch_size = args.batch
        self.img_size = 224
        self.num_classes = 10

        self.name_dataset = 'cifar'
        self.make_dataloader(train_set, test_set)

    def transform_tensor(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def make_dataloader(self, train_set, test_set):
        self.train = DataLoader(dataset=train_set, batch_size=self.batch_size, num_workers=0, shuffle=True)
        self.test = DataLoader(dataset=test_set, batch_size=self.batch_size, num_workers=0, shuffle=False)
