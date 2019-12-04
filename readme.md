1、使用train_ssd.py训练模型，其中改的只有自己的检测类别，
     图像和对应的xml文件（训练集和验证集分开）存放在VOCdevkit/VOC2007和VOC2012下，
     再对epochs，batch_size等参数简单修改即可。

2、得到model后直接使用inference_ssd.py进行推理，只需改测试文件地址和一些简单阈值等参数的设置。

