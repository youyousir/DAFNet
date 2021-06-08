# DAFNet

deformable alignment fusion for real-time semantic segmentation

### Dataset

You need to download the [Cityscapes](https://www.cityscapes-dataset.com/downloads/) dataset.

```
├── cityscapes
|    ├── gtFine
|    ├── leftImg8bit
|    ├── cityscapes_trainval_list.txt
|    ├── cityscapes_train_list.txt
|    ├── cityscapes_test_list.txt
|    └── cityscapes_val_list.txt 
```







### Result

- quantitative results:

| Dataset          | mIoU  | FPS  | model                                                        | Test                  |
| ---------------- | ----- | ---- | ------------------------------------------------------------ | --------------------- |
| Cityscapes(Fine) | 77.2% | 39   | [BaiduDrive](https://pan.baidu.com/s/1jsi1fiG474KA3DDbGJ3hzQ) (uzpj),  [GoogleDrive](https://drive.google.com/file/d/1PTI4nzjdx4iC7G_8q9BA5y9t-UXBYzxa/view?usp=sharing) | on validation dataset |
| Camvid           | 71.0% | --   | --                                                           | on test dataset       |

- qualitative segmentation examples:

![1622728333455](/Users/mac/Desktop/1622728333455.gif)
