# from transformers import AdamW
from tqdm import tqdm 
from transformers import AutoModel 
from transformers import AutoTokenizer
from dataloader import data_loader
from model import NER 
from transform import reshape_and_remove_pad, get_correct_and_total_count
import mlflow.pytorch
import torch

def train(model, epochs, loader):
    lr = 2e-5 if model.tuning else 5e-4
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
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 10 == 0:
                counts = get_correct_and_total_count(labels, outs)
                acc = counts[0] / counts[1]
                acc_content = counts[2] / counts[3]
                print("epoch=",epoch, " step=", step, " loss=", loss.item(),
                    " accuracy=", acc, " accuracy_content=", acc_content)
        
    mlflow.pytorch.log_metric('loss', loss)
    torch.save(model, "./model/ner_tune.model")
    
if __name__ == "__main__":
    with mlflow.start_run():
        mlflow.auto_log()
        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        pretrained = AutoModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        path = "./data/train_data.csv"
        batch_size = 32
        loader = data_loader(path, batch_size, tokenizer)
        epochs = 1
        ner = NER(pretrained)
        # 参数量
        print("模型参数量为",sum(i.numel() for i in ner.parameters()) / 10000, "万.")  # 万为单位
        train(ner, epochs, loader)
        mlflow.pytorch.log_model(ner, "model")
