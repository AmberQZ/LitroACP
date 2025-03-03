import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
from transformers import EvalPrediction
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import click
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import f1_score, accuracy_score, classification_report

class ACPDataset(Dataset):

    def __init__(self, df, tokenizer, max_len = 256):
        
        t = df['input'].values.tolist()
        self.text = [tokenizer(str(s), 
                        padding='max_length', 
                        max_length = max_len, 
                        truncation=True, 
                        return_token_type_ids = False, 
                        return_tensors="pt") for s in t]
        self.labels = torch.tensor(df['acp'].tolist())

    def __len__(self):
        assert len(self.text) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        output = {k: v.flatten() for k,v in self.text[idx].items()}
        output['labels'] = torch.tensor(int(self.labels[idx]))
        return output

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    f1_per_class = f1_score(labels, predictions, average=None)  
    accuracy = accuracy_score(labels, predictions)

    report = classification_report(labels, predictions, target_names=[f'Class {i}' for i in range(len(f1_per_class))])
    print(report)

    return {
        # "classification_report": report,
        "accuracy": accuracy,
        "f1_macro": f1_score(labels, predictions, average="macro"),  
        "f1_micro": f1_score(labels, predictions, average="micro"),  
        "f1_per_class": f1_per_class.tolist(), 
    }



class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=0.2, alpha=0.1, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

def adver_train_classifier(train_ds, val_ds, model, tokenizer, batch_size=16, epochs=10, learning_rate=2e-5, out_dir=""):
    
    pgd = PGD(model)
    K = 3
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy='epoch',
        weight_decay=0.01,
        save_strategy="epoch",
        logging_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to='none'
    )
    
    class AdversarialTrainer(Trainer):
        def training_step(self, model, inputs):
            loss = super().compute_loss(model, inputs)
            
            loss.backward()
            pgd.backup_grad()
            
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) 
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()

                loss_adv = super().compute_loss(model, inputs)
                loss_adv.backward()
            pgd.restore()  
            
            # for name, param in model.named_parameters():
            #     if "embeddings" in name:
            #         print(f"Parameter {name} after training step: {param.data}")

            
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss

    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    

def evaluate(loader, model, device):
    
    all_preds = None
    all_labels = None

    for i,batch in enumerate(loader):
        input = {k: v.to(device) for k,v in batch.items()}

        out = model(**input)
        
        logits = out.logits.cpu().detach().numpy()
        labels = input['labels'].cpu().detach().numpy()
        
        all_preds = logits if all_preds is None else np.concatenate((all_preds, logits))
        all_labels = labels if all_labels is None else np.concatenate((all_labels, labels))
        
    eval_pred = EvalPrediction(predictions = all_preds, label_ids=all_labels)
    return compute_metrics(eval_pred)


def train_classifier(train_ds, val_ds, model, tokenizer, batch_size = 32, epochs = 10, learning_rate = 2e-5, out_dir = ""):
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy = 'epoch',
        weight_decay=0.01,
        save_strategy="epoch",
        # logging_steps=10,
        logging_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to='none'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    

@click.command()
@click.option('--mode', help='mode of the dataset to train the model', required=True)
@click.option('--adver', default=False, help='AdversarialTraining', required=True)
@click.option('--max_len', default=256, help='Maximum length for the input sequence')
@click.option('--batch_size', default=16, help='Batch size', required=True)
@click.option('--epochs', default=20, help='Number of epochs', required=True)
@click.option('--learning_rate', default=2e-5, help='Learning rate', required=True)
def main(mode, adver=False, max_len=256, batch_size = 32, epochs = 10, learning_rate = 2e-5):
    
    """Trains the NLACP identification module"""

    
    MODEL = 'distilbert-base-uncased'
    # MODEL = 'bert-base-uncased'
    NUM_CLASSES = 3
    if adver:
        out_dir =f'checkpoints\AdverDes\{mode}'
    else:
        out_dir =f'checkpoints\Des\{mode}'
    
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(f'data\{mode}_des.csv')
    
    train_df, val_df = train_test_split(df, test_size = 0.2, random_state=42)

    
    model = DistilBertForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_CLASSES)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL)
    
    train_ds = ACPDataset(train_df, tokenizer, max_len=max_len)
    val_ds = ACPDataset(val_df, tokenizer, max_len=max_len)
    
    print('\n =========================== Training details =========================== \n')
    print(f'Dataset: {mode}\nNum. of classes: {NUM_CLASSES}\nNum. of epochs: {epochs}\nLearning rate: {learning_rate}\nBatch size: {batch_size}\nCheckpoint dir.: {out_dir}\n')
    print(' ======================================================================= \n')
    
    if adver:
        print('对抗训练')
        adver_train_classifier(train_ds, val_ds, model, tokenizer, batch_size = batch_size, epochs = epochs, learning_rate = learning_rate, out_dir = out_dir)
    else:
        train_classifier(train_ds, val_ds, model, tokenizer, batch_size = batch_size, epochs = epochs, learning_rate = learning_rate, out_dir = out_dir)

    test_dataloader = DataLoader(val_ds, num_workers=1)
    model.eval()
    model.to('cuda:0')
    
    test_results = evaluate(test_dataloader, model, 'cuda:0')

    print("Evaluation Results:")
    # print(test_results['classification_report'])
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Macro F1 Score: {test_results['f1_macro']:.4f}")
    print(f"Micro F1 Score: {test_results['f1_micro']:.4f}")
    print("F1 Scores per Class:", test_results['f1_per_class'])


if __name__ == '__main__':
    main()