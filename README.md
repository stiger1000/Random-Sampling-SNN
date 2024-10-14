# Random-Sampling-SNN

### Requirements

pytorch==1.10.0+cu113

spikingjelly==0.0.0.0.12

networkx==3.1

timm==1.0.9

torchtoolbox==0.1.8.2


### Training on ImageNet

```
python -m torch.distributed.launch --nproc_per_node=8 main.py --dataset IMAGENET --batch-size 16 --epochs 120 --T 4 --graph-model WS --skip-ratio 0.05 --data-path YOUR_DATA_PATH
```



### Training on CIFAR-10

````
python -m torch.distributed.launch --nproc_per_node=1 main.py --dataset CIFAR-10 --graph-model BA --skip-ratio 0.05 --data-path YOUR_DATA_PATH
````



### Training on CIFAR-100

````
python -m torch.distributed.launch --nproc_per_node=1 main.py --dataset CIFAR-100 --graph-model BA --skip-ratio 0.05 --data-path YOUR_DATA_PATH
````

### Problems you may meet
If you get an output like follows:
AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.

You can fix it by replace `np.int` by `int`.