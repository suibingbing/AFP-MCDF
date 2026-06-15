#import package
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import numpy as np
import csv
import os
from scipy.interpolate import interp1d
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
#Calculation of indicators and dataset definitions
device = torch.device("cuda:0")
random_seed = 42
loss_all=99999
metrics = (0, 0, 0,0,0,0,0)  
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

def calculate_metrics2(labels, scores, threshold):

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    binary_predictions = [1 if scores[i] >= threshold else 0 for i in sorted_indices]
    TP = sum([1 for i in range(len(labels)) if labels[i] == 1 and binary_predictions[i] == 1])
    FP = sum([1 for i in range(len(labels)) if labels[i] == 0 and binary_predictions[i] == 1])
    TN = sum([1 for i in range(len(labels)) if labels[i] == 0 and binary_predictions[i] == 0])
    FN = sum([1 for i in range(len(labels)) if labels[i] == 1 and binary_predictions[i] == 0])
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0.0
    sensitivity = recall
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    return recall, specificity, sensitivity, precision
import torch
import torch.nn.functional as F

def pad_tensor_to_shape(tensor, target_shape):
    """
    将给定 tensor 补零为目标形状。

    参数:
    tensor (torch.Tensor): 需要补零的 tensor。
    target_shape (tuple): 目标形状 (batch_size, target_dim, width)。

    返回:
    torch.Tensor: 补零后的 tensor。
    """
    current_shape = tensor.shape
    assert len(current_shape) == len(target_shape), "输入的 tensor 和目标形状必须具有相同的维度数量"
    
    # 计算第二维度需要补零的数量
    padding_needed = target_shape[1] - current_shape[1]
    
    assert padding_needed >= 0, "目标形状的第二维度必须大于等于当前形状的第二维度"
    
    # 使用 F.pad 函数补零
    padded_tensor = F.pad(tensor, (0, 0, 0, padding_needed))
    
    return padded_tensor



def update_best_metrics(a, b, c, recall, specificity, sensitivity, precision, metrics):
    d, e, f,recall3, specificity3, sensitivity3, precision3= metrics
    updated = False 
    if a >=d:
        d = a
        recall3=recall
        specificity3=specificity
        sensitivity3=sensitivity
        precision3=precision
        updated = True
        
    if specificity >=specificity3:
        specificity3=specificity
        updated = True
        
    if sensitivity >=sensitivity3:
        sensitivity3=sensitivity
        updated = True
        
    if b >= e:
        e = b
        updated = True
    if f >= c:
        c = f
        updated = True
    return (d, e, c,recall3, specificity3, sensitivity3, precision3), updated

def process_sequence(sequence):
    max_length = 400
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence + '0' * (max_length - len(sequence))

class MyDataset(Dataset):
    def __init__(self, file):
        self.sequence, self.label = self.read_file(file)
        self.sequence_protbert = self.add_space_between_characters(self.sequence)
        
        # 将数据打乱
        combined = list(zip(self.sequence, self.label, self.sequence_protbert))
        random.shuffle(combined)
        self.sequence, self.label, self.sequence_protbert = zip(*combined)

    def read_file(self, file_path):
        sequences = []
        labels = []
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None) 
            for row in csv_reader:
                sequences.append(row[0])
                labels.append(row[1])
        return sequences, labels
    
    def add_space_between_characters(self, input_list):
        new_list = []
        for element in input_list:
            new_element = ' '.join(element)
            new_list.append(new_element)
        return new_list

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        sample = process_sequence(self.sequence[index])
        sample_protbert = self.sequence_protbert[index]
        label = int(self.label[index])
        return sample, label, sample_protbert
    
    
#FusPB-ESM2 model Definition
class ESM_Model(nn.Module):
    def __init__(self,):
        super(ESM_Model, self).__init__()
        
        self.model = AutoModel.from_pretrained("ESM")
        self.tokenizer = AutoTokenizer.from_pretrained("ESM")
        self.bilstm1 = nn.LSTM(self.model.config.hidden_size, 512, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc_esm = nn.Linear(3564, 512)  
        
    def forward(self, inputs):
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt",max_length=400)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs_esm = self.model(input_ids=input_ids, attention_mask=attention_mask)

        outputs_esm_pool = outputs_esm.pooler_output   
        # outputs_esm_last = outputs_esm.last_hidden_state  
        # outputs_esm_last_2d=pad_tensor_to_shape(outputs_esm_last,(16, 400, 480))
        # outputs_esm_last_2d=outputs_esm_last_2d.unsqueeze(1)

        
        return outputs_esm_pool


        
class Prot_Model(nn.Module):
    def __init__(self,):
        super(Prot_Model, self).__init__()

        self.tokenizer_pro = BertTokenizer.from_pretrained("protbert", do_lower_case=False)
        self.model_pro = BertModel.from_pretrained("protbert")
        self.bilstm2 = nn.LSTM(self.model_pro.config.hidden_size, 512, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(0.2)  

    def forward(self, inputs2):

        encoded_input = self.tokenizer_pro(inputs2, padding=True, truncation=True,return_tensors='pt',max_length=400).to(device)
        outputs_pro = self.model_pro(**encoded_input)
        
        outputs_pro_pool = outputs_pro.pooler_output   
        # outputs_pro_last = outputs_pro.last_hidden_state  
        # outputs_pro_last=pad_tensor_to_shape(outputs_pro_last,(16, 400, 1024))
        # outputs_pro_last = outputs_pro_last.unsqueeze(1)
        # outputs_pro_2d = F.interpolate(outputs_pro_last, size=(400, 480), mode='bilinear', align_corners=False)
        
        return outputs_pro_pool 
    
    



class MyModel(nn.Module):
    def __init__(self,):
        super(MyModel, self).__init__()
        self.esm_based=ESM_Model()
        self.prot_based=Prot_Model()
        # self.sigmoid = nn.Sigmoid()
        self.fc_class1 = nn.Linear(1024, 2)
        # self.bilstm1 = nn.LSTM(480, 256, num_layers=1, bidirectional=True) 
        # self.conv4=nn.Conv2d(1,3,kernel_size=(4,480))
        # self.conv5=nn.Conv2d(1,3,kernel_size=(5,480))
        # self.conv6=nn.Conv2d(1,3,kernel_size=(6,480))
        # self.fc=nn.Linear(3564, 512)
        self.fc2=nn.Linear(1024, 480)
        # self.dropout = nn.Dropout(0.2)
        
    def forward(self, inputs,inputs2):
        
        lstm_pool_esm=self.esm_based(inputs)
        lstm_pool_pro=self.prot_based(inputs2)
        lstm_pool_pro=self.fc2(lstm_pool_pro)
#         one_d=lstm_pool_esm+lstm_pool_pro
#         two_d=output_2d_esm+output_2d_pro
        
#         x1,_=self.bilstm1(one_d.unsqueeze(0))
#         x1 = self.dropout(x1)
#         x1 = x1.squeeze(0)
        
#         batch_pro=two_d.size(0)
#         # two_d=two_d.unsqueeze(1)
#         output1=self.conv4(two_d).view(batch_pro,-1)
#         output2=self.conv5(two_d).view(batch_pro,-1)
#         output3=self.conv6(two_d).view(batch_pro,-1)
#         output_cnn=torch.cat((output1,output2,output3),dim=1)
#         x2=self.fc(output_cnn)
#         x3=self.fc_class1(torch.cat((x1,x2),dim=1))
        x=lstm_pool_esm+lstm_pool_pro


        return x
#Read the dataset

train_file = 'data/train_300.csv'  #Read training set
test_file = 'data/val_300.csv'  #Read independent test set
train_dataset = MyDataset(train_file)
test_dataset = MyDataset(test_file)
batch_size = 8  #Setting batchsize
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
#Model loading and setting

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
criterion = nn.CrossEntropyLoss()
model = MyModel()#Model loading
model.to(device)
# learning_rates=0.00006 #Setting learning rates
# optimizer = torch.optim.Adadelta(model.parameters())
# optimizer = optim.Adam([
#     {'params': model.esm_based.parameters(), 'lr': 0.000065},
#     {'params': model.prot_based.parameters(), 'lr': 0.00005},
#     {'params': model.fc2.parameters(), 'lr': 1e-3},
#     {'params': model.bilstm1.parameters(), 'lr': 1e-3},
#     {'params': model.conv4.parameters(), 'lr': 1e-3},
#     {'params': model.conv5.parameters(), 'lr': 1e-3},
#     {'params': model.conv6.parameters(), 'lr': 1e-3},
#     {'params': model.fc.parameters(), 'lr': 1e-3},
#     {'params': model.fc_class1.parameters(), 'lr': 1e-3}
# ])
#Model training and evaluation
kf = KFold(n_splits=5, shuffle=False)
best_auc=0
best_acc=0
best_epoch=0
best_epoch2=0
all_fpr = []
all_tpr = []
all_aucs = []
all_accs = []
# 模型训练
for fold, (train_indices, valid_indices) in enumerate(kf.split(train_dataset)):
    # 根据KFold的划分获取训练集和验证集
    best_auc=0
    best_acc=0
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    print(train_indices,len(train_indices))
    print(valid_indices,len(valid_indices))
    best_fpr=np.array([])
    best_tpr=np.array([])
    # 创建数据加载器实例，使用SubsetRandomSampler来划分数据
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
    model = MyModel()#Model loading
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0000658)# 模型训练
    item=0
    for epoch in range(2):
        item=item+1
        print(item)
        for batch_data, batch_labels,batch_data_pro in train_dataloader:
            model.train()
            batch_labels = batch_labels.to(device)
            # 前向传播
            outputs = model(batch_data,batch_data_pro)
            loss = criterion(outputs, batch_labels)
            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        all_labels = []
        all_scores = []
        model.eval()      
        for batch_data, batch_labels,batch_data_pro in valid_dataloader:
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data,batch_data_pro)
            probabilities = nn.functional.softmax(outputs, dim=1)
            scores = probabilities[:, 1]  # 正类的概率

            # 收集真实标签和预测得分
            all_labels.extend(batch_labels.tolist())
            all_scores.extend(scores.tolist())
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        # print(len(fpr))
        # print(len(tpr))
        auc = roc_auc_score(all_labels, all_scores)
        correct_predictions = (np.array(all_scores) >= 0.5).astype(int)
        acc = np.mean(correct_predictions == np.array(all_labels))
        # if acc>best_acc:
        #     best_acc=acc
        #     file_path = "acc/esm/model_10fold_t6_{}.pth".format(fold)
        #     torch.save(model.state_dict(), file_path)
        print(auc)
        if auc>best_auc:
            best_fpr=fpr
            best_tpr=tpr
            best_auc=auc
            file_path = "outputs/1d/5_fold_roc_1d_{}.pth".format(fold)
            if not os.path.exists("outputs/1d"):
                os.makedirs("outputs/1d")
            torch.save(model.state_dict(), file_path)
    all_fpr.append(best_fpr)
    all_tpr.append(best_tpr)
    all_aucs.append(best_auc)
    all_accs.append(best_acc)
    print(f"Fold {fold + 1}: AUC = {best_auc:.6f}, Accuracy = {best_acc:.6f}")
    plt.figure(figsize=(8, 6))
max_length = max(len(fpr) for fpr in all_fpr)
new_all_fpr = []
new_all_tpr = []
# 进行插值操作
for fpr, tpr in zip(all_fpr, all_tpr):
    f = interp1d(np.linspace(0, 1, len(fpr)), fpr)
    t = interp1d(np.linspace(0, 1, len(tpr)), tpr)
    new_fpr = f(np.linspace(0, 1, max_length))
    new_tpr = t(np.linspace(0, 1, max_length))
    new_all_fpr.append(new_fpr)
    new_all_tpr.append(new_tpr)

all_fpr=new_all_fpr
all_tpr=new_all_tpr

for i in range(len(all_fpr)):
    plt.plot(all_fpr[i], all_tpr[i], linestyle='--',lw=1, label=f'Fold {i + 1} (AUC = {all_aucs[i]:.3f})')

mean_fpr = np.mean(all_fpr, axis=0)
mean_tpr = np.mean(all_tpr, axis=0)

plt.plot(mean_fpr, mean_tpr, color='b', linestyle='-', lw=1.5, label='Mean ROC (AUC = {:.3f})'.format(np.mean(all_aucs)))

# 设置图形属性
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('outputs/1d/5_fold_roc_1d.png',dpi=400)

# 显示图形
plt.show()

# 输出每个折叠的AUC和准确率
print("AUC for each fold:", all_aucs)
print("Accuracy for each fold:", all_accs)
print("Mean AUC:", np.mean(all_aucs))
print("Mean Accuracy:", np.mean(all_accs))