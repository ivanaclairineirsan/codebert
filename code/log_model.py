from transformers import AutoTokenizer, AutoModel, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
import mlflow
import transformers
from model import Model
import pandas as pd 
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           RobertaConfig,
 
mlflow.set_tracking_uri(uri="http://127.0.0.1:88")
 
architecture = "microsoft/codebert-base" 
model_path = "/app/Vulnerability-Detection/WP1A/CodeBERT-classification/code/output/300724_Git_imbalance_diff/checkpoint-best-f1"
# tokenizer = AutoTokenizer.from_pretrained(architecture)
# config = RobertaConfig.from_pretrained(architecture)
# model = RobertaForSequenceClassification.from_pretrained(model_path, config=config, from_pt=True)

config = RobertaConfig.from_pretrained(architecture)
config.num_labels=2
tokenizer = RobertaTokenizer.from_pretrained(architecture)
model = RobertaForSequenceClassification.from_pretrained(architecture,config=config)    
# model=Model(model,config,tokenizer,args)

dataset_source_url = "/app/Vulnerability-Detection/WP1A/CodeBERT-classification/dataset/Git/context0_commit/imbalance/diff/Git_Train.json"
df = pd.read_csv(dataset_source_url)
dataset: PandasDataset = mlflow.data.from_pandas(df, source=dataset_source_url)
 
with mlflow.start_run():
    components = {
        "model": model,
        "tokenizer": tokenizer,
    }
 
    params = {
        'programming_language': 'java',
        'data': 'Git',
        'setting': 'imbalance'
    }
    mlflow.log_params(params)
 
    mlflow.log_metrics({"f1": 27.16, "precision": 25.5800000, "recall": 28.95})

    mlflow.log_input(dataset, context="training")
 
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="300724_Git_imbalance_diff",
        registered_model_name="300724_Git_imbalance_diff",
    )

    mlflow.log_text(csv_text, "test_data.csv")
 