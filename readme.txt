重要文件说明：

checkpoints文件夹：存储训练好的模型的权重参数
logs文件夹：存储tensotboard记录训练过程的文件，可用于观察训练过程中指标变化
model文件夹：模型代码
pic文件夹：保存模型训练推理过程中的一些感兴趣的结果图
vit_pytorch文件夹：好像暂时没用到
augmentations.py: 数据增强函数代码
dpandataset.py: 精神病数据集的代码
dpanmodel.py: 构建精神病识别分类模型的代码
dpanmodel_vis.py: 用于精神病识别分类模型可视化的代码，和dpanmodel.py应该几乎一样，训练过程中可能有时会改模型，重新建一个便于记录之前版本
main_detect.py: 主文件，训练验证代码

network_fea_visualize.ipynb: 用于结果统计和网络模型可视化的代码
network_fea_vis.py: 上述jupyter文件的python文件版本，可能改的过程中有小差异
vit_classifier.py: vit模型的构建代码，ViT会被用在精神病模型中
vit_classifier_vis.py: 类似dpanmodel_vis.py的作用
vis_mae.py: 好像暂时没用到


注意：由于个人习惯不好，代码里有很多地方用的绝对路径，自己在使用时要根据本地情况修改