import pandas as pd


class DataPreprocess():
    def __init__(self, path):
        self.path = path 
        # self.data = None 
        self.lab_list = []
        self.text_list = []
        self.data = None 
    
    def read_data(self):
        # 读取数据
        data = pd.read_csv(self.path)

        data = data.dropna()  # 去除缺失值
        data = data.sample(frac=1)# 随机打乱
        print("数据大小：",data.shape)
        self.data = data.reset_index()
        return self.data


    def label_text(self):
        # 提取每个text的每个命名实体的位置以及其对应的label，
        data = self.read_data()
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

        for idx in range(data.shape[0]):
            text = self.data.loc[idx, "text"]
            self.text_list.append(list(text))
            ners = []
            lab = [0] * len(text)
            for pos, label in zip(positions[idx], labels[idx]):
                start = pos[0]
                end = pos[1]  # 因为索引是[start, end) 左开右闭
                lab[start:end] = [label] * (end-start)
                ner = text[start:end]
                ners.append(ner)
            ner_list.append(ners)
            self.lab_list.append(lab)
        
        return self.lab_list, self.text_list

    def train_test_split(self):
        lens = self.data.shape[0]
        train_lens = int(lens * 0.8)
        train_data = self.data.iloc[:train_lens, :]
        test_data = self.data.iloc[train_lens:, :]
        train_data.to_csv("D:/03_code/chinese_ner/NER1/data/train_data.csv")
        test_data.to_csv("D:/03_code/chinese_ner/NER1/data/test_data.csv")

if __name__ == '__main__':   
    path = "D:/03_code/chinese_ner/NER1/project-44-at-2022-08-12-00-39-368803cf.csv"
    dp = DataPreprocess(path)
    data = dp.read_data()
    dp.train_test_split()
    lab_list, text_list = dp.label_text()
    print("lab example：", lab_list[0])
    print("text example：", text_list[0])
    
    lens = []
    for i in text_list:
        l = len(i)
        lens.append(l)
        
    print("序列的最大长度", max(lens))