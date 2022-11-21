import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

# 读取数据
data = pd.read_csv('./project-44-at-2022-08-12-00-39-368803cf.csv')
# data.head(3)

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
# print(tokenizer)

print("数据大小：",data.shape)
# data.iloc[56:58, :]
data = data.dropna()  # 去除缺失值
data = data.reset_index()
# data.shape

# 提取每个text的每个命名实体的位置以及其对应的label，
positions = []
labels = []
for instance in data.loc[:, "label"]:
    position = []
    label = []
    instance = eval(instance)   # label本身转为dataframe是个字符串，因此使用eval函数将其转为代码
    for txt in instance:
        start = txt["start"]  # 存储当前命名实体的开始位置
        end = txt["end"]    # 存储当前命名实体的结束位置
        lab = txt["labels"]  # 存储当前命名实体的标签
        label.append(lab)  # lab格式是列表，如['backgound']
        position.append((start, end))
    positions.append(position)
    labels.append(label)
    
# 将labels转为数值型
# names = ["background", "definition", "feature", "method", "experimental_result"]
name = [1,2,3,4,5]
labels_ = labels[:]
for idx in range(len(labels_)):
    for i in range(len(labels_[idx])):
        label = labels_[idx][i]
        if label == "backgound": 
            labels_[idx][i] = 1
        elif label == "definition": 
            labels_[idx][i] = 2
        elif label == "feature": 
            labels_[idx][i] = 3
        elif label == "method": 
            labels_[idx][i] = 4
        elif label == "experimental_result": 
            labels_[idx][i] = 5


class Dataset(torch.utils.data.Dataset):
    def __init__(self, labels, texts):
        dataset = {
            "tokens": texts,
            "ner_tags": labels
        }
        self.dataset = dataset
            
    def __len__(self):
        return len(self.dataset["tokens"])
    
    def __getitem__(self, i):
        tokens = self.dataset["tokens"][i]
        labels = self.dataset["ner_tags"][i]
        
        return tokens, labels

dataset = Dataset(lab_list, text_list)
token, label = dataset[0]
len(dataset)
print(token)
print(label)

# 数据整理函数
def collate_fn(data):
    """
    function: 对tokens进行ids编码，以及序列化，paddings等操作
    outputs: 输出经过padding的tokens,以及tokens的labels
    """
    tokens = [i[0] for i in data]
    labels = [i[1] for i in data]
    
    inputs = tokenizer.batch_encode_plus(tokens,
                                         padding=True,  # 补齐tensor长度，以最大长度为准
                                         truncation=True,
                                         return_tensors="pt",
                                         is_split_into_words=True)
    lens = inputs["input_ids"].shape[1]
    
    # 对label也进行补齐 用0表示padding的位置
    for i in range(len(labels)):  # 对每个label都补齐
        labels[i] = [0] + labels[i]   # 开头补一个cls
        labels[i] += [0] * lens       # 保证长度足够
        labels[i] = labels[i][:lens]  # 只截取最长长度
    
    return inputs, torch.LongTensor(labels) 

# 数据加载
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,  
                                     shuffle=True,
                                     drop_last=True)
# 查看数据的样例
for i, (inputs, labels) in enumerate(loader):
    if i==1:
    #    print(inputs)
    #    print(labels.shape)
    #    break
        print(len(loader)) # 打印batch数目
        print(tokenizer.decode(inputs["input_ids"][0]))
        print(labels[0])

        for k, v in inputs.items():
            print(k, v.shape)
            
            
# 训练模型
from transformers import AutoModel

# 加载预训练模型
pretrained = AutoModel.from_pretrained("hfl/chinese-bert-wwm-ext")
# 参数量
print("模型参数量为",sum(i.numel() for i in pretrained.parameters()) / 10000, "万.")  # 万为单位

# 模型试算
pretrained(**inputs).last_hidden_state.shape  # 最后一维度输出的形状

# 定义下游任务模型
class NER(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuning = False
        self.pretrained = None
        
        self.rnn = torch.nn.GRU(768, 768, batch_first=True)
        self.fc = torch.nn.Linear(768, 7)
        
    def forward(self, inputs):
        if self.tuning: # 是否微调
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state  # pretrained是个外部全局变量，表示一个预训练好的模型
        
        out, _ = self.rnn(out)
        out = self.fc(out).softmax(dim=2)
        
        return out

    def fine_tuning(self, tuning):
        self.tuning = tuning
        if tuning:
            for i in pretrained.parameters():
                i.requires_grad = True  # 需要微调，则将grad加上用于反向传播训练
            
            pretrained.train()  
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)
            
            pretrained.eval()  
            self.pretrained = None
            
model = NER()
model(inputs).shape

#对计算结果和label变形,并且移除pad
def reshape_and_remove_pad(outs, labels, attention_mask):
    #变形,便于计算loss
    #[b, lens, 8] -> [b*lens, 8]
    outs = outs.reshape(-1, 7)
    #[b, lens] -> [b*lens]
    labels = labels.reshape(-1)

    #忽略对pad的计算结果
    #[b, lens] -> [b*lens - pad]
    select = attention_mask.reshape(-1) == 1
    outs = outs[select]
    labels = labels[select]

    return outs, labels


reshape_and_remove_pad(torch.randn(2, 3, 7), torch.ones(2, 3),
                       torch.ones(2, 3))


#获取正确数量和总数
def get_correct_and_total_count(labels, outs):
    #[b*lens, 8] -> [b*lens]
    outs = outs.argmax(dim=1)
    correct = (outs == labels).sum().item()
    total = len(labels)

    #计算除了0以外元素的正确率,因为0太多了,包括的话,正确率很容易虚高
    select = labels != 0
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs == labels).sum().item()
    total_content = len(labels)

    return correct, total, correct_content, total_content


get_correct_and_total_count(torch.ones(16), torch.randn(16, 7))


# 训练
from transformers import AdamW
from tqdm import tqdm

def train(epochs):
    lr = 2e-5 if model.tuning else 5e-4
    
    # 训练
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        for step, (inputs, labels) in tqdm(enumerate(loader)):
            outs = model(inputs)
            outs, labels = reshape_and_remove_pad(outs, labels,
                                                  inputs["attention_mask"])
            
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 10 == 0:
                counts = get_correct_and_total_count(labels, outs)
                acc = counts[0] / counts[1]
                acc_content = counts[2] / counts[3]
                print("epoch=",epoch, " step=", step, " loss=", loss.item(),
                      " accuracy=", acc, " accuracy_content=", acc_content)
        
        # torch.save(model, "./model/ner.model")

model.fine_tuning(False)
# 参数量
print("模型参数量为",sum(i.numel() for i in model.parameters()) / 10000, "万.")  # 万为单位
train(1)


#测试
def test(model):
    model_load = torch.load('/content/命名实体识别_中文.model')
    model_load = model
    model_load.eval()

    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=128,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    correct = 0
    total = 0

    correct_content = 0
    total_content = 0

    for step, (inputs, labels) in enumerate(loader_test):
        if step == 5:
            break
        print(step)

        with torch.no_grad():
            #[b, lens] -> [b, lens, 8] -> [b, lens]
            outs = model_load(inputs)

        #对outs和label变形,并且移除pad
        #outs -> [b, lens, 8] -> [c, 8]
        #labels -> [b, lens] -> [c]
        outs, labels = reshape_and_remove_pad(outs, labels,
                                              inputs['attention_mask'])

        counts = get_correct_and_total_count(labels, outs)
        correct += counts[0]
        total += counts[1]
        correct_content += counts[2]
        total_content += counts[3]

    print(correct / total, correct_content / total_content)


test(model)

#测试
def predict(model):
    model_load = torch.load('/content/命名实体识别_中文.model')
    model_load = model
    model_load.eval()

    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (inputs, labels) in enumerate(loader_test):
        break

    with torch.no_grad():
        #[b, lens] -> [b, lens, 8] -> [b, lens]
        outs = model_load(inputs).argmax(dim=2)

    for i in range(32):
        #移除pad
        select = inputs['attention_mask'][i] == 1
        input_id = inputs['input_ids'][i, select]
        out = outs[i, select]
        label = labels[i, select]
        
        #输出原句子
        print(tokenizer.decode(input_id).replace(' ', ''))

        #输出tag
        for tag in [label, out]:
            s = ''
            for j in range(len(tag)):
                if tag[j] == 0:
                    s += '·'
                    continue
                s += tokenizer.decode(input_id[j])
                s += str(tag[j].item())

            print(s)
        print('==========================')


predict(model)