# DAFNet

deformable alignment fusion for real-time semantic segmentation



### Installation
This code has been tested with Python 3.6, pytorch 1.3, CUDA 9.0 and cuDNN 7.4.1 on Ubuntu 16.04. 
- install some packages
```
pip install tabulate
```
You can install Deformable convolution
···
cd DAFNet/DeformCN
sh make.sh
···

-One GPU with 11GB is needed


### Dataset

You need to download the [Cityscapes](https://www.cityscapes-dataset.com/downloads/) dataset.

```
├── cityscapes
|    ├── gtFine
|    ├── leftImg8bit

```

### Training and evaluation
Training with multi-gpu
```
CUDA_VISIBLE_DEVICES=0,1,... python -m torch.distributed.launch --nproc_per_node=n train.py 
```
Training with one gpu
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py 
```
Evaluation
```
python evaluate.py
```
Inference Speed
```
python eval_fps.py
```


### Result

- quantitative results:

| Dataset          | mIoU  | FPS  | model                                                        | Test                  |
| ---------------- | ----- | ---- | ------------------------------------------------------------ | --------------------- |
| Cityscapes(Fine) | 77.2% | 39   | [BaiduDrive](https://pan.baidu.com/s/1jsi1fiG474KA3DDbGJ3hzQ) (uzpj),  [GoogleDrive](https://drive.google.com/file/d/1PTI4nzjdx4iC7G_8q9BA5y9t-UXBYzxa/view?usp=sharing) | on validation dataset |
| Camvid           | 76.4% | --   | [Detailed result](https://www.cityscapes-dataset.com/anonymous-results/?id=ad42cc2ce867d17024596d52f2d93b6aa6215947058e01caaf9e8e8dfe148733)                                                         | on test dataset       |
| Camvid           | 71.0% | --   | --                                                           | on test dataset       |

- qualitative segmentation examples:

![1622728333455](imgs/1622728333455.gif)
