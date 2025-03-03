import os
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
import json
import random
import click

@click.command()
@click.option('--mode', help='Mode of the dataset to train the model', required=True)
@click.option('--auxiliary', default=False, help='Path to the ACP dataset', required=True)
@click.option('--acr', default=False, help='Path to the ACP dataset', required=True)
@click.option('--ratio', default=0.8, help='dataset split ratio', required=True)
@click.option('--batch_size', default=8, help='Batch size', required=True)
@click.option('--epochs', default=20, help='Number of epochs', required=True)
@click.option('--learning_rate', default=5e-5, help='Learning rate', required=True)
@click.option('--batch_size', default=8, help='Batch size', required=True)
@click.option('--weight_decay', default=0.01, help='weight_decay', required=True)

def train_and_evaluate(mode,auxiliary,acr,ratio, batch_size, epochs, learning_rate, weight_decay):
    
    acp_path = f'data/{mode}/{mode}_{"FGFacp" if acr else "GFacp"}.json'
    # non_path = f'data/{mode}/{mode}_{"FGnon" if acr else "Gnon"}.json'    


    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    with open(acp_path, "r", encoding='utf-8') as f:
        acp_dataset = json.load(f)
        
    # with open(non_path, "r", encoding='utf-8') as f:
    #     non_dataset = json.load(f)
        
    random.shuffle(acp_dataset)
    
    
    # if auxiliary:
    #     # train_dataset = acp_dataset[:int(len(acp_dataset)*ratio)]+non_dataset
    # else:
    train_dataset = acp_dataset[:int(len(acp_dataset)*ratio)]
        
    test_dataset = acp_dataset[int(len(acp_dataset)*ratio):]

    print('Dataset is splitted...')

    print('Train Dataset size:', len(train_dataset))

    print('Test Dataset size:', len(test_dataset))

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = GLiNER.from_pretrained("urchade/gliner_small").to(device)

    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)


    num_steps = 500
    data_size = len(train_dataset)
    num_batches = data_size // batch_size

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        others_lr=1e-5,
        others_weight_decay=0.001,
        lr_scheduler_type="linear", #cosine
        warmup_ratio=0.5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=epochs,
        evaluation_strategy="steps",
        eval_steps=100,  # 每10步评估一次
        logging_steps=100,  # 每50步记录一次日志
        save_steps = 100,
        save_total_limit=10,
        dataloader_num_workers = 0,
        use_cpu = False,
        report_to="none",
        log_level="info",  # 设置日志级别为 info
        load_best_model_at_end=True,  # 加载最佳模型
        save_strategy="steps",  # 每几步保存一次模型
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    
    if acr: 
        evaluationfull_results =model.evaluate(
            test_dataset, flat_ner=False, entity_types=["Action", "Subject","Resource"]
        )
    else:
        evaluationfull_results =model.evaluate(
            test_dataset, flat_ner=False, entity_types=["Action", "Subject","Resource","Condition","Purpose"]
        )        

    print(evaluationfull_results)

if __name__ == "__main__":
    train_and_evaluate()