import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from features.CNN import MyImageDataset,MyCNN
from features import GLCM,HIST,LBP,SIFT
from features.mix_feature import mix_feature
from function.tensor2numpy import denormalize,tensor_to_numpy
from function import model,decomposition
from function.model import write_prediction,write_ensemble_predictions
from function import eval

transforms_train_data = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transforms_test_data = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
#读取数据
def load_data(data_name,transforms_type):
    images, labels = [], []
    data_dir = '实验10 综合作业-数据集'
    data_root = os.path.join(data_dir, data_name)

    for i in range(20):  # 类别从 0 到 19
        class_dir = os.path.join(data_root, str(i))
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            try:
                image = Image.open(fpath).convert('RGB')  # 转为 RGB 图像
                image = transforms_type(image)
                images.append(image)
                labels.append(i)
            except Exception as e:
                print(f"无法读取图像 {fpath}，错误：{e}")
    return images, labels

idx2label = {
    0:'African people and villages',
    1:'Beach',
    2:'Historical buildings',
    3:'Buses',
    4:'Dinosaurs',
    5:'Elephants',
    6:'Flowers',
    7:'Horses',
    8:'Mountains and glacier',
    9:'Food',
    10:'Dogs',
    11:'Lizards',
    12:'Fashion',
    13:'Sunsets',
    14:'Cars',
    15:'WaterFalls',
    16:'Antiques',
    17:'Battle ships',
    18:'Skiing',
    19:'Desserts'
    }
print(idx2label)
train_images_tensor,train_labels = load_data('train',transforms_type=transforms_train_data)
test_images_tensor,test_labels = load_data('test',transforms_type = transforms_test_data)
#数据集可视化
cols ,rows= 3,3
figure = plt.figure(figsize=(8,8))
for i in range(1,cols*rows+1):
    sample_idx = np.random.randint(len(test_images_tensor))
    img,label  = test_images_tensor[sample_idx],test_labels[sample_idx]
    img = denormalize(img)  #(3,H,W)
    img = img.permute(1, 2, 0)  # [H, W, C]
    np_img = img.cpu().numpy()
    figure.add_subplot(rows,cols,i)
    plt.imshow(np_img)
    plt.axis('off')
    plt.title(idx2label[label])
plt.show()

#------------------------
#深度学习模型CNN
#------------------------
train_dataset = MyImageDataset(train_images_tensor,train_labels)
train_loader = DataLoader(train_dataset,batch_size =128,shuffle=True) #使用较小的batchsize
test_dataset = MyImageDataset(test_images_tensor,test_labels)
test_loader = DataLoader(test_dataset,batch_size=10,shuffle=True)
cnn_model = MyCNN(train_loader,test_loader)


#---------------------
#机器学习模型SVM
#---------------------
weights = [0.802]
train_images_np = [ tensor_to_numpy(img) for img in train_images_tensor ]
test_images_np  = [ tensor_to_numpy(img) for img in test_images_tensor ]

train_feature_GLCM = GLCM.GLCM_features(train_images_np)     
test_feature_GLCM = GLCM.GLCM_features(test_images_np)  #1000,20


train_feature_HIST = HIST.extract_rgb_histogram(train_images_np) #1000,768
test_feature_HIST = HIST.extract_rgb_histogram(test_images_np)

train_feature_LBP = LBP.extract_lbp_features(train_images_np)#1000,26
test_feature_LBP = LBP.extract_lbp_features(test_images_np)

train_feature_SIFT = SIFT.sift_feature(train_images_np)#1000,100
test_feature_SIFT = SIFT.sift_feature(test_images_np)
train_feature_SIFT,test_feature_SIFT = decomposition.pca(train_feature_SIFT,test_feature_SIFT)

model_GLCM = model.svm_train(train_feature_GLCM,train_labels,'model_GLCM')
score = model_GLCM.score(test_feature_GLCM,test_labels)
weights.append(score)
print(f'以GLCM为特征训练的SVM得分情况为:{score}')
print('\n')

model_HIST = model.svm_train(train_feature_HIST,train_labels,'model_HIST')
score = model_HIST.score(test_feature_HIST,test_labels)
weights.append(score)
print(f'以HIST为特征训练的SVM得分情况为:{score}')
print('\n')

model_LBP = model.svm_train(train_feature_LBP,train_labels,'model_LBP')
score = model_LBP.score(test_feature_LBP,test_labels)
weights.append(score)
print(f'以LBP为特征训练的SVM得分情况为:{score}')
print('\n')

model_SIFT = model.svm_train(train_feature_SIFT,train_labels,'model_SIFT')
score = model_SIFT.score(test_feature_SIFT,test_labels)
weights.append(score)
print(f'以SIFT为特征训练的SVM得分情况为:{score}')
print('\n')


#---------------------
#不同特征组合
#---------------------

#features1 : HIST and LBP
train_features1 = mix_feature(train_feature_HIST,train_feature_LBP)
test_features1 = mix_feature(test_feature_HIST,test_feature_LBP)
model_HISTandLBP = model.svm_train(train_features1,train_labels,'model_HISTandLBP')
score = model_HISTandLBP.score(test_features1,test_labels)
weights.append(score)
print(f'以　HIST 和 LBP　为混合特征训练的SVM得分情况为:{score}')
print('\n')

#features2: SIFT and GLCM
train_features2 = mix_feature(train_feature_SIFT,train_feature_GLCM)
test_features2  = mix_feature(test_feature_SIFT,test_feature_GLCM)
model_SIFTandGLCM = model.svm_train(train_features2,train_labels,'model_SIFTandGLCM')
score = model_SIFTandGLCM.score(test_features2,test_labels)
weights.append(score)
print(f'以　SIFT 和 GLCM　为混合特征训练的SVM得分情况为:{score}')
print('\n')

#features3 HIST LBP SIFT GLCM 共同组合
train_features3 = mix_feature(train_features1,train_features2)
test_features3  = mix_feature(test_features1,test_features2)
model_all = model.svm_train(train_features3,train_labels,'model_all')
score = model_all.score(test_features3,test_labels)
weights.append(score)
print(f'以　所有特征 为混合特征训练的SVM得分情况为:{score}')
print('\n')

#---------------
#集成学习
#---------------
svm_model_list = [model_GLCM,model_SIFT,model_LBP,model_SIFT,model_HISTandLBP,model_SIFTandGLCM,model_all]
svm_feature_list = [test_feature_GLCM,test_feature_SIFT,test_feature_LBP,test_feature_SIFT,test_features1,test_features2,test_features3]
acc = model.ensemble_classifier(
    cnn_model=cnn_model,
    svm_models=svm_model_list,
    cnn_test_data=train_images_tensor,
    svm_test_features_list=svm_feature_list,
    weights=weights,
    labels=test_labels,
    device='cpu')
print(f'集成学习的准确率为：{acc:.4f}')

# 预测结果写入txt文件
CNN_pred = write_prediction(model=cnn_model,feature=test_images_tensor,idx2label=idx2label,true_labels=test_labels,model_name='model_CNN')
GLCM_pred = write_prediction(model=model_GLCM,feature=test_feature_GLCM,idx2label=idx2label,true_labels=test_labels,model_name='model_GLCM')
HIST_pred = write_prediction(model=model_HIST,feature=test_feature_HIST,idx2label=idx2label,true_labels=test_labels,model_name='model_HIST')
LBP_pred = write_prediction(model=model_LBP,feature=test_feature_LBP,idx2label=idx2label,true_labels=test_labels,model_name='model_LBP')
SIFT_pred = write_prediction(model=model_SIFT,feature=test_feature_SIFT,idx2label=idx2label,true_labels=test_labels,model_name='model_SIFT')
HISTandLBP_pred = write_prediction(model=model_HISTandLBP,feature=test_features1,idx2label=idx2label,true_labels=test_labels,model_name='model_HISTandLBP')
SIFTandGLCM_pred = write_prediction(model=model_SIFTandGLCM,feature=test_features2,idx2label=idx2label,true_labels=test_labels,model_name='model_SIFTandGLCM')
all_fearue_pred = write_prediction(model=model_all,feature=test_features3,idx2label=idx2label,true_labels=test_labels,model_name='model_all')
ensemble_pred = write_ensemble_predictions(
        cnn_model=cnn_model,
        svm_models=svm_model_list,
        cnn_test_data=test_images_tensor,
        svm_test_features_list=svm_feature_list,
        test_labels=test_labels,
        idx2label=idx2label,
        weights=weights,
        model_name='ensemble'
    )
#绘制混淆矩阵
eval.plot_confusion_matrix(y_true=test_labels,y_pred=CNN_pred,idx2label=idx2label,feature_name='CNN')
eval.plot_confusion_matrix(y_true=test_labels,y_pred=GLCM_pred,idx2label=idx2label,feature_name='GLCM')
eval.plot_confusion_matrix(y_true=test_labels,y_pred=HIST_pred,idx2label=idx2label,feature_name='HIST')
eval.plot_confusion_matrix(y_true=test_labels,y_pred=LBP_pred,idx2label=idx2label,feature_name='LBP')
eval.plot_confusion_matrix(y_true=test_labels,y_pred=SIFT_pred,idx2label=idx2label,feature_name='SIFT')
eval.plot_confusion_matrix(y_true=test_labels,y_pred=HISTandLBP_pred,idx2label=idx2label,feature_name='HISTandLBP')
eval.plot_confusion_matrix(y_true=test_labels,y_pred=SIFTandGLCM_pred,idx2label=idx2label,feature_name='SIFTandGLCM')
eval.plot_confusion_matrix(y_true=test_labels,y_pred=all_fearue_pred,idx2label=idx2label,feature_name='All_feature')
eval.plot_confusion_matrix(y_true=test_labels,y_pred=ensemble_pred,idx2label=idx2label,feature_name='Ensemble')

#绘制混淆矩阵柱状图
eval.plot_confusion_matrix_3d(y_true=test_labels,y_pred=CNN_pred,idx2label=idx2label,feature_name='CNN')
eval.plot_confusion_matrix_3d(y_true=test_labels,y_pred=GLCM_pred,idx2label=idx2label,feature_name='GLCM')
eval.plot_confusion_matrix_3d(y_true=test_labels,y_pred=HIST_pred,idx2label=idx2label,feature_name='HIST')
eval.plot_confusion_matrix_3d(y_true=test_labels,y_pred=LBP_pred,idx2label=idx2label,feature_name='LBP')
eval.plot_confusion_matrix_3d(y_true=test_labels,y_pred=SIFT_pred,idx2label=idx2label,feature_name='SIFT')
eval.plot_confusion_matrix_3d(y_true=test_labels,y_pred=HISTandLBP_pred,idx2label=idx2label,feature_name='HISTandLBP')
eval.plot_confusion_matrix_3d(y_true=test_labels,y_pred=SIFTandGLCM_pred,idx2label=idx2label,feature_name='SIFTandGLCM')
eval.plot_confusion_matrix_3d(y_true=test_labels,y_pred=all_fearue_pred,idx2label=idx2label,feature_name='All_feature')
eval.plot_confusion_matrix_3d(y_true=test_labels,y_pred=ensemble_pred,idx2label=idx2label,feature_name='Ensemble')

