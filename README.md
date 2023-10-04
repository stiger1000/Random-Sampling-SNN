# Random-Sampling-SNN

### Requirements

pytorch==1.10.0+cu113

spikingjelly==0.0.0.0.12



### Training on ImageNet

```
python -m torch.distributed.launch --nproc_per_node=8 main.py --dataset IMAGENET --batch-size 16 --epochs 120 --T 4 --graph-model WS --skip-ratio 0.05 
```



### Training on CIFAR-10

````
python -m torch.distributed.launch --nproc_per_node=1 main.py --dataset CIFAR-10 --graph-model BA --skip-ratio 0.05
````



### Training on CIFAR-100

````
python -m torch.distributed.launch --nproc_per_node=1 main.py --dataset CIFAR-100 --graph-model BA --skip-ratio 0.05
````

