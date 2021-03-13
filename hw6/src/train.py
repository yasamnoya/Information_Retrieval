import torch
import argparse
import random
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AdamW
from glob import glob
from multiprocessing import cpu_count


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, dict_data_list):
        self.dict_data_list = dict_data_list

    def __getitem__(self, idx):
        item = {key: torch.tensor([val])
                for key, val in self.dict_data_list[idx].items()}
        return item

    def __len__(self):
        return len(self.dict_data_list)

class CLSLayer(torch.nn.Module):
    def __init__(self, p = 0.2):
        super(CLSLayer, self).__init__()
        self.fc = torch.nn.Linear(768, 768)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = p)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class IRModel(torch.nn.Module):
    def __init__(self, model_name):
        super(IRModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.cls1 = CLSLayer(p = 0.2)
        self.cls2 = CLSLayer(p = 0.2)
        self.top = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooler_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        x = self.cls1(pooler_output)
        x = self.cls2(pooler_output)
        x = self.top(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", type=str, default="roberta-base")
    parser.add_argument("-bert_data_dir", type=str, default="../bert_data/")
    parser.add_argument("-check_pt_dir", type=str, default="../check_pt/")
    parser.add_argument("-epoches", type=int, default=1)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-neg_num", type=int, default=3)
    parser.add_argument("-use_device", type=int, default=0)
    parser.add_argument("-report_every", type=int, default=50)
    parser.add_argument("-train_from", type=int, default=0)
    args = parser.parse_args()

    pos_file_list = glob(f"{args.bert_data_dir}train.pos.*.pt")
    neg_file_list = glob(f"{args.bert_data_dir}train.neg.*.pt")

    device = torch.device(
        f'cuda:{args.use_device}') if args.use_device >= 0 else torch.device('cpu')

    model = IRModel(args.model_name)
    if args.train_from != 0:
        path = f"{args.check_pt_dir}{args.train_from}.pt"
        model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # training
    for epoch in range(args.train_from, args.epoches):
        for num_file, (pos_file, neg_file) in enumerate(zip(pos_file_list, neg_file_list)):
            pos_data = torch.load(pos_file)
            neg_data = torch.load(neg_file)
            pos_dataset = IRDataset(pos_data)
            neg_dataset = IRDataset(neg_data)

            acc = 0
            for step in range(len(pos_dataset)):
                task_list = []
                task_list.append(pos_dataset[step])
                for i in range(args.neg_num):
                    task_list.append(
                        neg_dataset[args.neg_num * step + i])

                random.shuffle(task_list)

                label = [i for i, task in enumerate(
                    task_list) if task["labels"] == 1]
                label = torch.tensor(label).to(device)

                weight = []
                for i in range(args.neg_num + 1):
                    if i == label:
                        weight.append(args.neg_num + 1)
                    else:
                        weight.append(1)
                weight = torch.tensor(weight).float().to(device)
                criterion = torch.nn.CrossEntropyLoss(weight)

                outputs = []
                for task in task_list:
                    input_ids = task['input_ids'].to(device)
                    attention_mask = task['attention_mask'].to(device)
                    token_type_ids = task['token_type_ids'].to(device)
                    # token_type_ids = torch.tensor([0] * 512).to(device)

                    output = model(input_ids, attention_mask,
                                   token_type_ids).float()
                    outputs.append(output[0])

                outputs = torch.reshape(
                    torch.cat(outputs), (1, args.neg_num + 1))
                predict = torch.argmax(outputs[0])
                ans = label[0]
                if predict == ans:
                    acc += 1

                optimizer.zero_grad()
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                if (step + 1) % args.report_every == 0:
                    print('Epoch [{}/{}], File[{}/{}], Step [{}/{}], Loss: {:.8f}, Acc: {:.8f}'
                          .format(epoch+1, args.epoches, num_file+1, len(pos_file_list), step+1, len(pos_dataset), loss.item(), acc / args.report_every))
                    acc = 0
        torch.save(model.state_dict(), f"{args.check_pt_dir}{epoch}.pt")
