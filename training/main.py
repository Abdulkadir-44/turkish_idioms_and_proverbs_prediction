# --- 1. Gerekli Kütüphaneler ---
import pandas as pd
import re
from datasets import Dataset
from transformers import MT5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import evaluate
import matplotlib.pyplot as plt
import os

# --- 2. Cihaz Kontrolü ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. CSV Dosyasını Yükle ---
veri_yolu = "../dataset/atasozleri_deyimler_50_yeni.csv"  
df = pd.read_csv(veri_yolu)

# --- 4. Temizlik Fonksiyonu ---
def temizle_metin(metin):
    metin = str(metin).lower()
    metin = re.sub(r'[0-9]', '', metin)
    metin = re.sub(r'[^\w\sçğıöşü]', '', metin)
    metin = re.sub(r'\s+', ' ', metin).strip()
    return metin

# --- 5. Temizlik ve Formatlama ---
df = df.dropna().drop_duplicates()
df['metin'] = df['metin'].apply(temizle_metin)
df['turu'] = df['turu'].str.lower().str.strip()
df['kategori'] = df['kategori'].str.lower().str.strip()



df['input'] = "metin: " + df['metin']
df['target'] = "tür: " + df['turu'] + " | kategori: " + df['kategori']


# --- 6. Dataset Split ---
train_df, val_df = train_test_split(df[['input', 'target']], test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# --- 7. Tokenizer ve Model ---
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# --- 8. Tokenizasyon ---
def tokenize(batch):
    input_enc = tokenizer(batch['input'], padding="max_length", truncation=True, max_length=128)
    target_enc = tokenizer(batch['target'], padding="max_length", truncation=True, max_length=128)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# --- 9. Data Collator ---
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# --- 10. Metrikler ---
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    result = {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bleu": bleu_result["bleu"]
    }
    return result

# --- 11. Eğitim Ayarları ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_atasozleri_checkpoints",
    do_eval=True,
    eval_steps=300,
    save_steps=300,
    logging_steps=10,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=3,
    learning_rate=3e-4,
    weight_decay=0.01,
    predict_with_generate=True,
    save_total_limit=3,
    logging_dir="./logs",
    report_to="none"
)

# --- 12. Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# --- 13. Eğitimi Başlat ---
trainer.train()

# --- 14. Loss Grafiklerini Çiz ---
logs = trainer.state.log_history
train_loss = [log["loss"] for log in logs if "loss" in log and "step" in log]
eval_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
steps = [log["step"] for log in logs if "loss" in log and "step" in log]
eval_steps = [log["step"] for log in logs if "eval_loss" in log]

plt.figure(figsize=(10, 5))
plt.plot(steps, train_loss, label="Train Loss")
plt.plot(eval_steps, eval_loss, label="Eval Loss")
plt.xlabel("Adım (Step)")
plt.ylabel("Loss")
plt.title("Eğitim ve Doğrulama Loss Grafiği")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 15. ROUGE & BLEU Tablosu ---
metric_logs = [log for log in logs if 'eval_rougeL' in log]
if metric_logs:
    metric_df = pd.DataFrame(metric_logs)[['step', 'eval_rouge1', 'eval_rouge2', 'eval_rougeL', 'eval_bleu']]
    print("\n=== Değerlendirme Metrikleri ===\n")
    print(metric_df.to_string(index=False))
else:
    print("Henüz metrik verisi kaydedilmedi.")

# --- 16. Modeli Kaydet ---
# model_dir = "./mt5_atasozleri_final_model"
# os.makedirs(model_dir, exist_ok=True)
# model.save_pretrained(model_dir)
# tokenizer.save_pretrained(model_dir)