import os
import re
import json
import torch
import numpy as np
from evaluate import load
from datasets import Dataset, Audio, load_metric
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# Set the default device based on availability
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
CUDA_LAUNCH_BLOCKING=1


# Preprocess audio files
def preprocess_audio(file_path):
    file_name = os.path.basename(file_path)
    pattern = r'_(.*)\.wav'
    match = re.search(pattern, file_name)
    if match:
        first_transcription = match.group(1)
    
    else:
        print(f"No match found in {file_path}")
        first_transcription = None
    second_pattern = r'_(.*)'
    second_match = re.search(second_pattern, first_transcription)
    if second_match:
        transcription = second_match.group(1)
        if transcription == 'F2':
                transcription = 'F'
    else:
        transcription = None
    return {
        'audio': file_path,
        'transcription': transcription.lower() if transcription else transcription
    }

audio_directory = './CleanedAudioDataFiles'
audio_files = [os.path.join(audio_directory, f) for f in os.listdir(audio_directory) if f.endswith('.wav')]
audio_data = [preprocess_audio(file) for file in audio_files]

audio_data = [data for data in audio_data if data is not None]

if not audio_data:
    print("No valid audio data found.")
else:
    dataset = Dataset.from_dict({
        'audio': [item['audio'] for item in audio_data],
        'transcription': [item['transcription'] for item in audio_data]
    })
   
    split_dataset = dataset.train_test_split(test_size=0.4)
    split_dataset = split_dataset.cast_column("audio", Audio(sampling_rate=16000))

    def extract_all_chars(batch):
        all_text = " ".join(batch['transcription'])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}
   
    vocabs = split_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=split_dataset["train"].column_names)
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    # After creating audio_data and before creating dataset:
    print("Printing audio_data to inspect transcriptions:")
    for item in audio_data:
        print(item['transcription'])

    # After creating vocabs:
    print("Printing vocab_list to inspect characters:")
    print(vocab_list)

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch["audio"]
        audio_array = audio["array"]
        inputs = processor(audio_array, sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
        with processor.as_target_processor():
            labels = processor(batch["transcription"], return_tensors="pt", padding=True)
        batch["input_values"] = inputs.input_values[0].cpu()  # Ensure this is on CPU
        batch["labels"] = labels.input_ids[0].cpu()  # Ensure this is on CPU
        return batch

    split_dataset = split_dataset.map(prepare_dataset, remove_columns=["audio", "transcription"])

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch
   
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


from transformers import TrainingArguments, Trainer, Wav2Vec2ForCTC
import optuna

training_args = TrainingArguments(
    output_dir="./wav2vec2-base-ivesa",
    group_by_length=True,
    per_device_train_batch_size=32,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    # fp16=True if device == "cuda" else False,  # Enable fp16 only if using CUDA
    save_steps=500,
    eval_steps=200,
    logging_steps=100,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
)


wer_metric = load("wer", trust_remote_code=True)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    pred_str = [s.lower() for s in pred_str]

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    label_str = [s.lower() for s in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# Define objective function for Optuna hyperparameter optimization
def objective(trial):
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8, 16, 32])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 20)

    model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    )


    model.gradient_checkpointing_enable()
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir="./wav2vec2-base-ivesa",
        group_by_length=True,
        per_device_train_batch_size=per_device_train_batch_size,
        evaluation_strategy="steps",
        num_train_epochs=num_train_epochs,
        fp16=True,  # Enable fp16 (mixed precision training)
        # optim = 'adafactor'
        save_steps=500,
        eval_steps=200,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=1000,
        save_total_limit=2,
    
        
    )


    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,  # Define your own compute_metrics function
    )


    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Final Evaluation Results: {eval_results}")

    # getting test data
    for i in range(20):
        i2 = processor(split_dataset['test'][i]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
        print(f"The input test audio is: {split_dataset['test'][i]['transcription']}")

        # prediction for test data
        with torch.no_grad():
            logits = model(**i2.to("cpu")).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)


        print(f'The output prediction is : {transcription[0]}')

    return eval_results["eval_loss"]  # Optuna minimizes this value


# Perform hyperparameter optimization with Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # Adjust n_trials as needed

# Print best hyperparameters found
print("Best trial:")
best_trial = study.best_trial
print("  Value: {}".format(best_trial.value))
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))


