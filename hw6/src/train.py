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


class IRModel(torch.nn.Module):
    def __init__(self, model_name):
        super(IRModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc1 = torch.nn.Linear(768, 768)
        self.fc2 = torch.nn.Linear(768, 768)
        self.fc3 = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        x = F.relu(last_hidden_state[:, -1, :])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", type=str, default="xlnet-base-cased")
    parser.add_argument("-bert_data_dir", type=str, default="../bert_data/")
    parser.add_argument("-check_pt_dir", type=str, default="../check_pt/")
    parser.add_argument("-epoches", type=int, default=1)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-neg_num", type=int, default=3)
    parser.add_argument("-use_cuda", type=bool, default=True)
    args = parser.parse_args()

    pos_file_list = glob(f"{args.bert_data_dir}train.pos.*.pt")
    neg_file_list = glob(f"{args.bert_data_dir}train.neg.*.pt")

    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    # device = torch.device('cpu')

    model = IRModel(args.model_name)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # training
    for epoch in range(args.epoches):
        for pos_file, neg_file in zip(pos_file_list, neg_file_list):
            pos_data = torch.load(pos_file)
            neg_data = torch.load(neg_file)
            random.shuffle(torch.load(pos_file))
            random.shuffle(torch.load(neg_file))
            pos_dataset = IRDataset(pos_data)
            neg_dataset = IRDataset(neg_data)

            for batch_num in range(len(pos_dataset)):
                task_list = []
                task_list.append(pos_dataset[batch_num])
                for i in range(args.neg_num):
                    task_list.append(neg_dataset[args.neg_num * batch_num + i])

                optimizer.zero_grad()

                outputs = []
                for task in task_list:
                    input_ids = task['input_ids'].to(device)
                    attention_mask = task['attention_mask'].to(device)
                    token_type_ids = task['token_type_ids'].to(device)

                    output = model(input_ids, attention_mask, token_type_ids)
                    outputs.append(output)

                outputs = torch.vstack(outputs)
                labels = torch.tensor(
                    [task["labels"] for task in task_list]).type(torch.LongTensor)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, args.epoches, batch_num+1, len(pos_dataset), loss.item()))
