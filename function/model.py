import joblib
import os
from sklearn import svm
import numpy as np
import torch
from torch import device

def svm_train(X,y,save_path):
    check_path = os.path.join('save',save_path)
    if os.path.exists(check_path) :
        print('已保存对应模型，加载已有模型')
        model = read(check_path)
        return model
    else :
        print('未找到对应模型，重新训练\n')
        model = svm.SVC(C=1.0,probability=True,kernel='rbf')
        model.fit(X,y)
        save(model,check_path)
        return model
def save(model,path):
    joblib.dump(model, path)
    print ("Done\n")
    return 

def read(path):
    model = joblib.load(path)
    return model

def ensemble_classifier(cnn_model, svm_models, cnn_test_data, svm_test_features_list,weights,labels, device='cpu'):
    """
    参数说明：
    - cnn_model: PyTorch CNN 模型
    - svm_models: list of sklearn 模型
    - cnn_test_data: List/Tensor of images for CNN
    - svm_test_features_list: list of feature arrays，对应每个SVM模型的特征
    - labels: 真实标签（用于计算准确率等）
    - device: 模型所在设备（cuda 或 cpu）
    - weights 权重

    返回：
    - 预测标签列表
    """
    cnn_model.to(device)
    cnn_model.eval()
    predictions = []
    for i in range(len(cnn_test_data)):
        proba_list = []
        
        # CNN 推理
        with torch.no_grad():
            input_tensor = cnn_test_data[i].unsqueeze(0).to(device)
            output = cnn_model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        proba_list.append(probs)

        # 对每个 SVM，使用其特定的特征
        for svm_model, feature_set in zip(svm_models, svm_test_features_list):
            feature = feature_set[i].reshape(1, -1)  # 单个样本
            probs = svm_model.predict_proba(feature)[0]
            proba_list.append(probs)
        
        # 软投票平均
        avg_probs = np.average(proba_list, axis=0,weights=weights)
        pred_label = np.argmax(avg_probs)
        predictions.append(pred_label)
    
    # 如果需要评估准确率
    acc = np.mean(np.array(predictions) == np.array(labels))

    return acc



def write_prediction(model, feature, idx2label, true_labels, model_name='model_CNN', device='cpu'):
    """
    将单个模型对整个测试集的预测结果输出到 result/<model_name>.txt。
    
    参数：
      - model: sklearn 的 SVM 或 PyTorch 的 CNN 模型
      - feature:
          * 如果 model 是 sklearn 模型：feature 应该是一个 numpy 数组，shape = (n_samples, n_features)
          * 如果 model 是 PyTorch CNN：feature 应该是一个 list 或者 Tensor，长度 = n_samples
      - idx2label: dict，从整型标签到字符串标签的映射
      - true_labels: list 或 numpy array，长度 = n_samples，每个元素都是整型真实标签
      - model_name: str，用于生成文件名
      - device: 'cpu' 或 'cuda'，如果是 CNN 则把数据移动到该设备
    """

    # 1. 确保 result/ 目录存在
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    y_pred = []
    # 2. 拼出输出文件路径
    #    我们把后缀写成 .txt，也可以改成 .csv
    out_path = os.path.join(result_dir, f"{model_name}.txt")

    # 3. 打开文件并写入表头
    #    这里用 '\t' 作为分隔符，也可以用空格或 ','，看自己需要
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("序号\t\t\t预测标签\t\t\t真实标签\n")

        n_samples = len(true_labels)
        # 4. 逐样本预测并写入
        for i in range(n_samples):
            # 根据模型类型做不同预测
            if isinstance(model, torch.nn.Module):
                # --- CNN 的预测分支 ---
                model.to(device)
                model.eval()
                with torch.no_grad():
                    # 假设 feature[i] 是一个 [3,H,W] 的 Tensor
                    img_tensor = feature[i].unsqueeze(0).to(device)  # (1,3,H,W)
                    logits = model(img_tensor)                       # (1, n_classes)
                    probs = torch.softmax(logits, dim=1)             # (1, n_classes)
                    pred_idx = torch.argmax(probs, dim=1).item()     # 整数
                    y_pred.append(pred_idx)
            else:
                # --- sklearn SVM 的预测分支 ---
                # 此时假设 feature 是一个 numpy 数组，shape=(n_samples,n_features)
                sample_feat = feature[i].reshape(1, -1)  # (1, n_features)
                pred_idx = model.predict(sample_feat)[0]  # 整数标签
                y_pred.append(pred_idx)
            # 5. 将整数标签翻译成字符串标签
            pred_label_str = idx2label[pred_idx]
            true_label_str = idx2label[int(true_labels[i])]

            # 6. 写入一行：序号、预测标签、真实标签，用 '\t' 分隔
            f.write(f"{i}\t\t\t{pred_label_str}\t\t\t{true_label_str}\n")

    print(f"已将 {model_name} 的预测结果写入：{out_path}")
    return np.array(y_pred)

def write_ensemble_predictions(
    cnn_model,
    svm_models,
    cnn_test_data,
    svm_test_features_list,
    test_labels,
    idx2label,
    weights,
    model_name='ensemble',
    device='cpu'
):
    """
    对多个模型（1 个 CNN + N 个 SVM）做加权软投票预测，并将结果写到 result/<model_name>.txt。

    参数：
      - cnn_model: 已加载好、eval 模式下的 PyTorch CNN 模型
      - svm_models: List[sklearn.svm.SVC]，所有 SVM 模型（都必须是 probability=True）
      - cnn_test_data: List[Tensor] 或 Tensor，长度 = n_samples，Tensor 形状 [3, H, W]
      - svm_test_features_list: List[np.ndarray]，长度 = len(svm_models)，每个 shape = (n_samples, n_features_i)
      - test_labels: List[int] 或 np.ndarray，长度 = n_samples，整数真实标签
      - idx2label: dict，将整数标签映射到字符串标签
      - weights: None 或可迭代对象，长度 = 1 + len(svm_models)。如果为 None，则所有模型权重相同（均值投票）。
      - model_name: str，用于生成输出文件名 result/<model_name>.txt
      - device: 'cpu' 或 'cuda'，用于 CNN 推理

    返回值：无。函数会在 result/ 目录下生成一个名为 <model_name>.txt 的结果文件。
    """
    y_pred = []
    out_path = os.path.join('result', f"{model_name}.txt")

    # 确定样本总数
    n_samples = len(test_labels)

    #检查权重
    weights = np.array(weights)
    n_models = 1 + len(svm_models)  # 1 for CNN + N for each SVM
    if weights.shape[0] != n_models:
        raise ValueError(f"weights 长度应为 {n_models}，但收到 {weights.shape[0]}")

    #确保 CNN 在正确设备、eval 模式
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    #开始写文件
    with open(out_path, 'w', encoding='utf-8') as f:
        # 写表头
        f.write("序号\t\t\t预测标签\t\t\t真实标签\n")

        # 对每个样本做预测
        for i in range(n_samples):
            proba_list = []

            # 6.1) CNN 预测概率
            with torch.no_grad():
                # 取出第 i 个图像，形状 [3, H, W]
                img_tensor = cnn_test_data[i]  # 假设已经是 [3, H, W] 的 Tensor
                # 扩展维度到 [1, 3, H, W]
                img_batched = img_tensor.unsqueeze(0).to(device)
                logits = cnn_model(img_batched)                      # [1, n_classes]
                probs_cnn = torch.softmax(logits, dim=1).squeeze().cpu().numpy()  # [n_classes]
            proba_list.append(probs_cnn)

            # 6.2) 每个 SVM 预测概率
            # 这里假设 svm_test_features_list[j][i] 就是第 i 个样本给第 j 个 SVM 的特征
            for j, svm_model in enumerate(svm_models):
                svm_feat = svm_test_features_list[j][i].reshape(1, -1)  # 1 行
                probs_svm = svm_model.predict_proba(svm_feat)[0]       # [n_classes]
                proba_list.append(probs_svm)

            # 6.3) 加权平均所有模型的概率
            proba_array = np.stack(proba_list, axis=0)  # shape = (n_models, n_classes)
            avg_probs = np.average(proba_array, axis=0, weights=weights)

            # 6.4) 选出最大概率对应的标签索引
            pred_idx = int(np.argmax(avg_probs))
            y_pred.append(pred_idx)
            true_idx = int(test_labels[i])

            # 6.5) 转为字符串标签
            pred_label_str = idx2label[pred_idx]
            true_label_str = idx2label[true_idx]

            # 6.6) 写入一行：序号、预测标签、真实标签，用制表符隔开
            f.write(f"{i}\t\t\t{pred_label_str}\t\t\t{true_label_str}\n")

        print(f"已将集成模型的预测结果写入：{out_path}")
    return np.array(y_pred)
