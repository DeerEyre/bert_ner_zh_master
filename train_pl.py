# from transformers import AdamW
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from dataloader import data_loader
from model import NER
from transform import reshape_and_remove_pad, get_correct_and_total_count
import mlflow.pytorch
import torch
import mlflow
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from preprocessing import DataPreprocess
from torch.nn import functional as F
import pandas as pd


def train(model, epochs, loader):
    lr = 2e-5 if model.tuning else 5e-4
    mlflow.autolog()
    # 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        for step, (inputs, labels) in tqdm(enumerate(loader)):
            outs = model(inputs)
            outs, labels = reshape_and_remove_pad(outs, labels,
                                                  inputs["attention_mask"])

            loss = criterion(outs, labels)
            print('loss=', loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                counts = get_correct_and_total_count(labels, outs)
                acc = counts[0] / counts[1]
                acc_content = counts[2] / counts[3]
                print("epoch=", epoch, " step=", step, " loss=", loss.item(),
                      " accuracy=", acc, " accuracy_content=", acc_content)

    torch.save(model, "./model/ner_tune.model")


class Dataset(Dataset):
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


class DataLightning(pl.LightningDataModule):
    def __init__(self, path, batch_size, tokenizer):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.data = self.read_data(path)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data, self.val_data = self.train_test_split()

            lab_list, text_list = self.label_text(self.train_data)
            self.train_data = Dataset(lab_list, text_list)

            lab_list, text_list = self.label_text(self.val_data)
            self.val_data = Dataset(lab_list, text_list)

        if stage == 'test' or stage is None:
            lab_list, text_list = self.label_text(self.data)
            self.test_data = Dataset(lab_list, text_list)

    def collate_fn(self, data):

        tokens = [i[0] for i in data]
        # print("=================", len(tokens[0]))
        labels = [i[1] for i in data]
        inputs = self.tokenizer.batch_encode_plus(tokens,
                                                  padding="max_length",  # 补齐tensor长度，以最大长度为准
                                                  truncation=True,
                                                  max_length=497,
                                                  return_tensors="pt",
                                                  is_split_into_words=True)
        lens = inputs["input_ids"].shape[1]
        # print("----------------", inputs["input_ids"].shape)
        # print("----------------", inputs["attention_mask"].shape)
        # 对label也进行补齐 用0表示padding的位置
        for i in range(len(labels)):  # 对每个label都补齐
            labels[i] = [0] + labels[i]  # 开头补一个cls
            labels[i] += [0] * lens  # 保证长度足够
            labels[i] = labels[i][:lens]  # 只截取最长长度

        return inputs, torch.LongTensor(labels)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          shuffle=True,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          shuffle=True,
                          drop_last=True)

    def read_data(self, path):
        # 读取数据
        data = pd.read_csv(path)
        data = data.dropna()  # 去除缺失值
        data = data.sample(frac=1)  # 随机打乱
        print("数据大小：", data.shape)
        self.data = data.reset_index()
        return self.data

    def label_text(self, data):
        # 提取每个text的每个命名实体的位置以及其对应的label，
        positions = []
        labels = []
        for instance in data.loc[:, "label"]:
            position = []
            label = []
            instance = eval(instance)  # label本身转为dataframe是个字符串，因此使用eval函数将其转为代码
            for txt in instance:
                start = txt["start"]  # 存储当前命名实体的开始位置
                end = txt["end"]  # 存储当前命名实体的结束位置
                lab = txt["labels"]  # 存储当前命名实体的标签
                label.append(lab)  # lab格式是列表，如['backgound']
                position.append((start, end))
            positions.append(position)
            labels.append(label)
            # 将命名实体的位置上的text提取出来,并将text中start-end位置赋予标签1,2,3,4,5

        # 将labels转为数值型
        # names = ["background", "definition", "feature", "method", "experimental_result"]
        # name = [1,2,3,4,5]
        for idx in range(len(labels)):
            # print(labels[idx])
            for i in range(len(labels[idx])):
                label = labels[idx][i][0]
                if label == "backgound":
                    labels[idx][i] = 1
                elif label == "definition":
                    labels[idx][i] = 2
                elif label == "feature":
                    labels[idx][i] = 3
                elif label == "method":
                    labels[idx][i] = 4
                elif label == "experimental_result":
                    labels[idx][i] = 5
        ner_list = []
        text_list, lab_list = [], []
        for idx in range(data.shape[0]):
            text = self.data.loc[idx, "text"]
            text_list.append(list(text))
            ners = []
            lab = [0] * len(text)
            for pos, label in zip(positions[idx], labels[idx]):
                start = pos[0]
                end = pos[1]  # 因为索引是[start, end) 左开右闭
                lab[start:end] = [label] * (end - start)
                ner = text[start:end]
                ners.append(ner)
            ner_list.append(ners)
            lab_list.append(lab)

        return lab_list, text_list

    def train_test_split(self):
        lens = self.data.shape[0]
        train_lens = int(lens * 0.8)
        self.train_data = self.data.iloc[:train_lens, :]
        self.val_data = self.data.iloc[train_lens:, :]
        # train_data.to_csv("train_data.csv")
        # test_data.to_csv("test_data.csv")
        return self.train_data, self.val_data


class NERLightning(pl.LightningModule):
    def __init__(self, pretrained):
        super().__init__()
        self.pretrained = pretrained
        self.NER = NER(self.pretrained)

    def forward(self, x):
        return self.NER(x)

    def training_step(self, batch, batch_size):
        """
        """
        inputs, labels = batch
        output = self.forward(inputs)
        output, labels = reshape_and_remove_pad(output, labels,
                                                inputs["attention_mask"])
        loss = torch.nn.CrossEntropyLoss()(output, labels)
        mlflow.log_metric("loss", loss)
        print("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    mlflow.set_tracking_uri('http://192.168.11.95:5002')
    mlflow.set_experiment('BERT_NER_zh')
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    pretrained = AutoModel.from_pretrained("hfl/chinese-bert-wwm-ext")
    path = "./data/train_data.csv"
    # path2 = './data/test_data.csv'
    batch_size = 16
    with mlflow.start_run():
        # val_loader = data_loader(path2, batch_size, tokenizer)
        # train_loader = data_loader(path, batch_size, tokenizer)
        data_module = DataLightning(path, batch_size, tokenizer)
        # epochs = 2
        model = NERLightning(pretrained)
        trainer = pl.Trainer(max_epochs=3)
        mlflow.pytorch.autolog()
        trainer.fit(model, data_module)
        mlflow.log_artifacts('gs-bert-ner')
        print('training has been ended')
        # 参数量
        # train(ner, epochs, loader)
        
