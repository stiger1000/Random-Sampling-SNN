from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchtoolbox.transform import Cutout
import os


def build_dataset(args):
    transform = []
    num_classes = 0
    if args.dataset == 'CIFAR-10':
        transform.append(transforms.RandomCrop(32, padding=4))
        transform.append(Cutout())
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(transform)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'CIFAR-100':
        transform.append(transforms.RandomCrop(32, padding=4))
        transform.append(Cutout())
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        transform_train = transforms.Compose(transform)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
        num_classes = 100
    elif args.dataset == 'IMAGENET':
        transform.append(transforms.RandomResizedCrop(224))
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform_train = transforms.Compose(transform)
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = ImageFolder(root=os.path.join(args.data_path, 'train'), transform=transform_train)
        val_dataset = ImageFolder(root=os.path.join(args.data_path, 'val'), transform=transform_test)
        num_classes = 1000
    else:
        raise NotImplementedError

    return train_dataset, val_dataset, num_classes
