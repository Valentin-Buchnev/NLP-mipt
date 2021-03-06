from torch.utils.data import DataLoader
import torch
import pandas as pd


class TextDataLoader(DataLoader):
    def __init__(
            self, data_set, shuffle=False, device="cuda", batch_size=16, num_workers=2,
            pad_token="<pad>",
            pad_label=0
    ):
        super(TextDataLoader, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.pad_token = pad_token
        self.device = device
        self.pad_label = pad_label

    def collate_fn(self, data):
        max_len = max(list(map(lambda x: len(x["labels"]), data)))
        res = {"labels": [], "texts": [], "labels_mask": []}
        for line in data:
            labels = line["labels"] + [self.pad_label] * max_len
            labels = labels[:max_len]
            res["labels"].append(labels)
            labels_mask = [1] * len(line["text"]) + [0] * max_len
            labels_mask = labels_mask[:max_len]
            text = line["text"] + [self.pad_label] * max_len
            text = text[:max_len]
            res["texts"].append(text)
            res["labels_mask"].append(labels_mask)
        res["texts"] = torch.LongTensor(res["texts"]).to(self.device)
        res["labels"] = torch.LongTensor(res["labels"]).to(self.device)
        res["labels_mask"] = torch.LongTensor(res["labels_mask"]).to(self.device)
        return res


class TextDataSet(object):
    def __init__(self, df_path, label2idx=None, pad_label=0, text2idx={"<pad>": 0}):
        self.df_path = df_path
        df = pd.read_csv(df_path, sep="\t")
        labels = list(map(lambda x: x.split(), df.labels))
        if label2idx is None:
            label2idx = {"<pad>": pad_label}
            for line in labels:
                for lbl in line:
                    if lbl not in label2idx:
                        label2idx[lbl] = len(label2idx)
        self.label2idx = label2idx
        self.labels = []
        for line in labels:
            line_labels = []
            for lbl in line:
                line_labels.append(self.label2idx[lbl])
            self.labels.append(line_labels)

        text = list(map(lambda x: x.split(), df.text))
        self.text = []
        for line in text:
            line_ids = []
            for word in line:
                if word not in text2idx:
                    text2idx[word] = len(text2idx)
                line_ids.append(text2idx[word])
            self.text.append(line_ids)
        self.text2idx = text2idx
        self.max_len = max(list(map(len, self.text)))
        self.idx2label = [0] * len(self.label2idx)
        for lbl in self.label2idx:
            self.idx2label[self.label2idx[lbl]] = lbl

    def __getitem__(self, item):
        return {"labels": self.labels[item], "text": self.text[item]}

    def __len__(self):
        return len(self.labels)


class LearnData(object):
    def __init__(
            self, train_df_path, valid_df_path, test_df_path,
            embedder=None, shuffle=True, device="cuda", batch_size=16, num_workers=0,
            pad_token="<pad>",
            pad_label=0):
        self.train_ds = TextDataSet(train_df_path, pad_label=pad_label)
        self.valid_ds = TextDataSet(valid_df_path, self.train_ds.label2idx, pad_label=pad_label, text2idx=self.train_ds.text2idx)
        self.test_ds = TextDataSet(test_df_path, self.train_ds.label2idx, pad_label=pad_label, text2idx=self.train_ds.text2idx)

        self.train_dl = TextDataLoader(
            self.train_ds,
            shuffle=shuffle,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pad_token=pad_token,
            pad_label=pad_label
        )
        self.valid_dl = TextDataLoader(
            self.valid_ds,
            shuffle=False,
            device=device,
            batch_size=batch_size, 
            num_workers=num_workers,
            pad_token=pad_token,
            pad_label=pad_label
        )
        
        self.test_dl = TextDataLoader(
            self.test_ds,
            shuffle=False,
            device=device,
            batch_size=batch_size, 
            num_workers=num_workers,
            pad_token=pad_token,
            pad_label=pad_label
        )