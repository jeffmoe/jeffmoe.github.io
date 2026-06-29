---
title: Fine-Tuning a Pre-Trained LLM
parent: Natural Language Processing
nav_order: 1
---

## Overview
his application demonstrates how to fine-tune a T5 (Text-to-Text Transfer Transformer) model for legal text classification. The system takes legal case data from a CSV file, preprocesses it, fine-tunes a T5 model to generate natural language descriptions of legal outcomes, and provides interactive testing capabilities.

---
## Features
- Processes legal case data: Reads case titles and text from CSV files
- Fine-tunes a T5 model: Adapts a pre-trained T5 model to legal domain text
- Generates natural language responses: Produces descriptive outcome classifications
- Compares models: Evaluates fine-tuned model against base model performance
- Interactive testing: Allows users to test models with custom questions

### Dataset
<p><img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/63a075eb-1df6-441d-b0d5-22bd50db2ffe" /></p>

---
## Installation
### Prereqs
- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- CUDA-capable GPU (optional, but recommended for faster training)

### Steps
  1. Clone the repository
  2. Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
  3. Run the Pipeline:
     ```bash
     # Use mini T5 model (faster)
     python main.py --csv legal_text_classification.csv --mini
    
     # Use T5-base model (slower but potentially better)
     python main.py --csv legal_text_classification.csv --base
     
     # Show sample questions without training
     python main.py --show_questions
      ```
  ---
## Further Testing
### Interactive Test Mode
```bash
python main.py --test
```
Ask Questions to the model in the following format:  
Question: What is the outcome for the case 'Alpine Hardwood'? 

---
## Command Line Options

|Option | Description|
|-------|------------|
|--csv PATH | Path to your CSV data file |
|--mini |	Use T5-small (faster training) |
|--base |	Use T5-base (slower, better quality) |
|--test | Run interactive testing mode |
|--show_questions | 	Display 5 sample questions |
|--skip_train |Skip training (use existing model) |

---
## App Structure
<p><img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/4e057ed5-c375-44cd-9b9d-b120b5fae163" /></p>
<p><img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/b8949076-1a16-4c0f-835c-4a19c3120bc6" /></p>
<p><img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/72d7145e-be9f-43f3-82b7-74f757358ca0" /></p>

## Code Files
### Data Preprocessing
```python
"""
Data preprocessing module for legal text classification.
Prepares data for text-to-text generation models.
"""

import pandas as pd
import json
import re
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path: str, output_json_path: str = None) -> Dict[str, Any]:
    """
    Load CSV data and prepare for text-to-text generation.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Clean the text
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Keep legal notation
        text = re.sub(r'[^\w\s\.\,\;\:\'\"\(\)\[\]\{\}\@\#\$\%\^\&\*\+\=\-\/\|\\]+', ' ', text)
        return text.strip()
    
    # Clean case titles and text
    df['clean_title'] = df['case_title'].apply(clean_text)
    df['clean_text'] = df['case_text'].apply(clean_text)
    
    # Truncate text to a reasonable length for T5
    df['clean_text'] = df['clean_text'].apply(lambda x: x[:1500] if len(x) > 1500 else x)
    
    # Create prompt-response pairs for text-to-text generation
    df['input_text'] = df.apply(
        lambda row: f"Question: What is the legal outcome for the case '{row['clean_title']}'? Context: {row['clean_text']}",
        axis=1
    )
    
    # The target is a descriptive response
    df['target_text'] = df.apply(
        lambda row: f"The case '{row['clean_title']}' was {row['case_outcome']}. This means the court {get_outcome_description(row['case_outcome'])}.",
        axis=1
    )
    
    # Create dataset dictionary
    data = {
        'inputs': df['input_text'].tolist(),
        'targets': df['target_text'].tolist(),
        'case_ids': df['case_id'].tolist(),
        'outcomes': df['case_outcome'].tolist(),
        'titles': df['clean_title'].tolist(),
        'text_bodies': df['clean_text'].tolist(),
        'original_titles': df['case_title'].tolist(),
    }
    
    # Save as JSON if path provided
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {output_json_path}")
    
    return data

def get_outcome_description(outcome: str) -> str:
    """Get a descriptive phrase for the outcome."""
    descriptions = {
        'cited': 'referred to this case as precedent',
        'applied': 'applied the legal principles from this case',
        'followed': 'followed the reasoning established in this case',
        'considered': 'considered the reasoning in this case',
        'referred_to': 'referred to this case in its reasoning',
        'discussed': 'discussed the legal principles from this case',
        'distinguished': 'distinguished this case from the present circumstances',
        'related': 'found this case to be related',
        'approved': 'approved the reasoning in this case',
        'affirmed': 'affirmed the decision in this case',
        'referred': 'referred to this case for guidance'
    }
    return descriptions.get(outcome, f'handled with outcome: {outcome}')

def split_data(data: Dict[str, Any], test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and test sets.
    """
    inputs = data['inputs']
    targets = data['targets']
    
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state
    )
    
    train_data = {
        'inputs': train_inputs,
        'targets': train_targets
    }
    
    test_data = {
        'inputs': test_inputs,
        'targets': test_targets
    }
    
    return train_data, test_data

def create_sample_questions(data: Dict[str, Any], num_questions: int = 5) -> List[Dict[str, Any]]:
    """
    Create 5 sample questions for testing.
    """
    import random
    
    indices = list(range(len(data['inputs'])))
    
    # Try to get variety
    selected_indices = []
    outcomes_used = set()
    
    for idx in indices:
        outcome = data['outcomes'][idx]
        if outcome not in outcomes_used and len(selected_indices) < num_questions:
            selected_indices.append(idx)
            outcomes_used.add(outcome)
    
    if len(selected_indices) < num_questions:
        remaining = [i for i in indices if i not in selected_indices]
        random.shuffle(remaining)
        selected_indices.extend(remaining[:num_questions - len(selected_indices)])
    
    questions = []
    for idx in selected_indices:
        questions.append({
            'case_id': data['case_ids'][idx],
            'title': data['original_titles'][idx],
            'actual_outcome': data['outcomes'][idx],
            'input_text': data['inputs'][idx],
            'expected_response': data['targets'][idx],
            'context': data['text_bodies'][idx][:300]
        })
    
    return questions

if __name__ == "__main__":
    # Test the preprocessing
    data = load_and_preprocess_data('legal_text_classification.csv', 'legal_data.json')
    print(f"Loaded {len(data['inputs'])} cases")
    print(f"\nSample input: {data['inputs'][0][:200]}...")
    print(f"\nSample target: {data['targets'][0]}")
    
    # Show sample questions
    questions = create_sample_questions(data, num_questions=5)
    print("\nSample Questions Generated:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q['title']}")
        print(f"   Expected: {q['expected_response']}")
        print()
```
### Model Training
```python
"""
Fine-tune a T5 model for legal text generation.
Compatible with transformers 5.12.1 and torch 2.12.1
"""

import json
import torch
import numpy as np
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import os
from tqdm import tqdm

def load_data(json_path: str):
    """Load preprocessed data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def train_model(
    json_data_path: str,
    model_name: str = 't5-small',
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    max_length: int = 512,
    output_dir: str = './fine_tuned_model'
):
    """
    Fine-tune a T5 model for legal text generation.
    Compatible with transformers 5.12.1
    """
    # Load data
    print("Loading data...")
    data = load_data(json_data_path)
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Prepare dataset
    def preprocess_function(examples):
        # Tokenize inputs
        inputs = tokenizer(
            examples['input'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # Tokenize targets
        targets = tokenizer(
            examples['target'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors=None
        )
        
        inputs['labels'] = targets['input_ids']
        return inputs
    
    # Split data
    from data_to_json import split_data
    train_data, test_data = split_data(data)
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'input': train_data['inputs'],
        'target': train_data['targets']
    })
    
    test_dataset = Dataset.from_dict({
        'input': test_data['inputs'],
        'target': test_data['targets']
    })
    
    # Preprocess datasets
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=max_length,
        label_pad_token_id=tokenizer.pad_token_id
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        logging_steps=50,
        report_to='none',
        save_total_limit=2,
        push_to_hub=False,
    )
    
    # Initialize trainer - REMOVED tokenizer parameter
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate
    print("Evaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"Test loss: {eval_results['eval_loss']:.4f}")
    
    return trainer, model, tokenizer

def train_mini_model(
    json_data_path: str,
    output_dir: str = './fine_tuned_mini_model'
):
    """
    Train a mini T5 model for faster experimentation.
    """
    model_name = 't5-small'
    
    return train_model(
        json_data_path=json_data_path,
        model_name=model_name,
        num_epochs=3,
        batch_size=4,
        learning_rate=3e-4,
        output_dir=output_dir
    )

if __name__ == "__main__":
    print("Training mini T5 model...")
    trainer, model, tokenizer = train_mini_model(
        json_data_path='legal_data.json',
        output_dir='./fine_tuned_mini_model'
    )
```
### Model Testing
```python
"""
Test and compare base T5 model vs fine-tuned model on 5 legal questions.
Compatible with transformers 5.12.1
"""

import json
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_to_json import create_sample_questions

def load_model_and_tokenizer(model_path: str):
    """Load a trained T5 model and tokenizer."""
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def generate_response(input_text: str, tokenizer, model, max_length: int = 256):
    """
    Generate a response for a given input text using T5.
    """
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_test_questions(data_path: str, num_questions: int = 5):
    """
    Generate exactly 5 test questions from the dataset.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    import random
    indices = list(range(len(data['inputs'])))
    
    selected_indices = []
    outcomes_used = set()
    
    for idx in indices:
        outcome = data['outcomes'][idx]
        if outcome not in outcomes_used and len(selected_indices) < num_questions:
            selected_indices.append(idx)
            outcomes_used.add(outcome)
    
    if len(selected_indices) < num_questions:
        remaining = [i for i in indices if i not in selected_indices]
        random.shuffle(remaining)
        selected_indices.extend(remaining[:num_questions - len(selected_indices)])
    
    questions = []
    for idx in selected_indices:
        questions.append({
            'case_id': data['case_ids'][idx],
            'title': data['titles'][idx],
            'actual_outcome': data['outcomes'][idx],
            'input_text': data['inputs'][idx],
            'expected_response': data['targets'][idx],
            'context': data['text_bodies'][idx][:300]
        })
    
    return questions

def compare_models(questions: list, base_model_path: str, fine_tuned_model_path: str):
    """
    Compare base T5 model and fine-tuned T5 model responses.
    """
    # Load models
    print("Loading base T5 model...")
    base_tokenizer, base_model = load_model_and_tokenizer(base_model_path)
    
    print("Loading fine-tuned T5 model...")
    ft_tokenizer, ft_model = load_model_and_tokenizer(fine_tuned_model_path)
    
    base_name = base_model_path.split('/')[-1] if '/' in base_model_path else base_model_path
    ft_name = fine_tuned_model_path.split('/')[-1] if '/' in fine_tuned_model_path else fine_tuned_model_path
    
    results = []
    
    
    print(f"\nMODEL COMPARISON: BASE ({base_name}) vs FINE-TUNED ({ft_name})")
    print(f"Testing on {len(questions)} legal cases:\n")
    
    for i, q in enumerate(questions, 1):
        print(f"{'─'*80}")
        print(f"Question {i}:")
        print(f"  Case: {q['title']}")
        print(f"  Expected outcome: {q['actual_outcome']}")
        print(f"  Expected response: {q['expected_response']}")
        print()
        
        try:
            # Generate responses
            base_response = generate_response(q['input_text'], base_tokenizer, base_model)
            ft_response = generate_response(q['input_text'], ft_tokenizer, ft_model)
            
            # Check if responses contain the expected outcome
            base_contains_outcome = q['actual_outcome'].lower() in base_response.lower()
            ft_contains_outcome = q['actual_outcome'].lower() in ft_response.lower()
            
            results.append({
                'question_id': i,
                'case_title': q['title'],
                'expected_outcome': q['actual_outcome'],
                'expected_response': q['expected_response'],
                'base_model_response': base_response,
                'ft_model_response': ft_response,
                'base_contains_outcome': base_contains_outcome,
                'ft_contains_outcome': ft_contains_outcome,
                'base_correct': base_contains_outcome,
                'ft_correct': ft_contains_outcome
            })
            
            # Print results
            print("BASE MODEL RESPONSE:")
            print(f"  {base_response}")
            print(f"  Contains expected outcome: {'YES' if base_contains_outcome else 'NO'}")
            print()
            
            print("FINE-TUNED MODEL RESPONSE:")
            print(f"  {ft_response}")
            print(f"  Contains expected outcome: {'YES' if ft_contains_outcome else 'NO'}")
            print()
            
            # Comparison summary
            if base_contains_outcome and ft_contains_outcome:
                print("➜ Both models correctly identified the outcome ✓")
            elif ft_contains_outcome and not base_contains_outcome:
                print("➜ Fine-tuned model improved - correctly identified the outcome ✓")
            elif base_contains_outcome and not ft_contains_outcome:
                print("➜ Fine-tuned model worsened - missed the outcome ✗")
            else:
                print("➜ Both models failed to identify the outcome ✗")
            
        except Exception as e:
            print(f"Error generating response for question {i}: {e}")
            results.append({
                'question_id': i,
                'case_title': q['title'],
                'expected_outcome': q['actual_outcome'],
                'expected_response': q['expected_response'],
                'base_model_response': f"ERROR: {str(e)}",
                'ft_model_response': f"ERROR: {str(e)}",
                'base_contains_outcome': False,
                'ft_contains_outcome': False,
                'base_correct': False,
                'ft_correct': False
            })
        
        print()
    
    # Summary statistics
    if results:
        df_results = pd.DataFrame(results)
        
        print("\nSUMMARY")
        
        base_accuracy = df_results['base_correct'].mean()
        ft_accuracy = df_results['ft_correct'].mean()
        
        print(f"Base Model Accuracy (contains outcome): {base_accuracy:.2%} ({df_results['base_correct'].sum()}/{len(df_results)})")
        print(f"Fine-tuned Model Accuracy (contains outcome): {ft_accuracy:.2%} ({df_results['ft_correct'].sum()}/{len(df_results)})")
        print(f"Improvement: {(ft_accuracy - base_accuracy):.2%}")
        
        # Detailed comparison
        improved = ((~df_results['base_correct']) & df_results['ft_correct']).sum()
        worsened = (df_results['base_correct'] & (~df_results['ft_correct'])).sum()
        same_correct = (df_results['base_correct'] & df_results['ft_correct']).sum()
        same_incorrect = ((~df_results['base_correct']) & (~df_results['ft_correct'])).sum()
        
        print(f"\nDetailed Comparison:")
        print(f"  Both correct: {same_correct}")
        print(f"  Both incorrect: {same_incorrect}")
        print(f"  Improved by fine-tuning: {improved}")
        print(f"  Worsened by fine-tuning: {worsened}")
        
        # Per-outcome breakdown
        print("\nPER-OUTCOME BREAKDOWN")
        
        outcome_counts = df_results['expected_outcome'].value_counts()
        for outcome in outcome_counts.index:
            n = outcome_counts[outcome]
            base_correct = df_results[df_results['expected_outcome'] == outcome]['base_correct'].sum()
            ft_correct = df_results[df_results['expected_outcome'] == outcome]['ft_correct'].sum()
            print(f"\n{outcome} ({n} case{'s' if n > 1 else ''}):")
            print(f"  Base model: {base_correct}/{n} correct ({base_correct/n:.1%})")
            print(f"  Fine-tuned: {ft_correct}/{n} correct ({ft_correct/n:.1%})")
        
        # Save detailed results
        df_results.to_csv('model_comparison_results.csv', index=False)
        print(f"\nDetailed results saved to 'model_comparison_results.csv'")
        
        return df_results
    else:
        print("No results to display.")
        return None

def test_models_main(json_data_path: str, base_model_path: str = 't5-small'):
    """
    Main function to test and compare models on 5 questions.
    """
    print("Generating 5 test questions from data...")
    questions = generate_test_questions(json_data_path, num_questions=5)
    
    print(f"\nGenerated {len(questions)} questions from cases:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q['title']}")
        print(f"   Expected: {q['actual_outcome']}")
        print()
    
    results = compare_models(
        questions=questions,
        base_model_path=base_model_path,
        fine_tuned_model_path='./fine_tuned_mini_model'
    )
    
    return results

if __name__ == "__main__":
    from data_to_json import load_and_preprocess_data
    load_and_preprocess_data('legal_text_classification.csv', 'legal_data.json')
    results = test_models_main('legal_data.json')
```
### Main Script
```python
"""
Main script to run the complete pipeline with T5 text generation.
"""

import os
import argparse
import json
from data_to_json import load_and_preprocess_data, create_sample_questions
from model_train import train_mini_model, train_model
from model_test import test_models_main, generate_test_questions, compare_models
from model_test import load_model_and_tokenizer, generate_response

def run_pipeline(csv_path: str, use_mini_model: bool = True):
    """
    Run the complete pipeline.
    """
    # Step 1: Preprocess data
    print("STEP 1: Preprocessing Data for Text Generation")
    data = load_and_preprocess_data(csv_path, 'legal_data.json')
    print(f"Preprocessed {len(data['inputs'])} cases")
    print(f"Unique outcome types: {set(data['outcomes'])}")
    
    # Show sample
    print("\nSample training pair:")
    print(f"Input: {data['inputs'][0][:150]}...")
    print(f"Target: {data['targets'][0]}")
    
    # Step 2: Train model
    print("\nSTEP 2: Training T5 Model")
    
    if use_mini_model:
        print("Using mini T5 model (t5-small) for faster training...")
        trainer, model, tokenizer = train_mini_model(
            json_data_path='legal_data.json',
            output_dir='./fine_tuned_mini_model'
        )
        base_model = 't5-small'
        ft_model_path = './fine_tuned_mini_model'
    else:
        print("Using T5-base model (slower but potentially better)...")
        trainer, model, tokenizer = train_model(
            json_data_path='legal_data.json',
            model_name='t5-base',
            num_epochs=3,
            output_dir='./fine_tuned_model'
        )
        base_model = 't5-base'
        ft_model_path = './fine_tuned_model'
    
    # Step 3: Test and compare models
    print("\nSTEP 3: Testing and Comparing Models on 5 Legal Cases")
    
    questions = generate_test_questions('legal_data.json', num_questions=5)
    
    print("\nTesting on the following cases:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q['title']} (Expected: {q['actual_outcome']})")
    
    results = compare_models(
        questions=questions,
        base_model_path=base_model,
        fine_tuned_model_path=ft_model_path
    )

    print("\nPIPELINE COMPLETE")
    return results

def interactive_test():
    """
    Run interactive testing with custom legal input.
    """
    print("\nINTERACTIVE LEGAL TEXT GENERATION TEST")
    
    ft_model_path = './fine_tuned_mini_model'
    if not os.path.exists(ft_model_path):
        print("No fine-tuned model found. Please run training first.")
        return
    
    print("Loading models...")
    print("Using t5-small as base model")
    base_tokenizer, base_model = load_model_and_tokenizer('t5-small')
    ft_tokenizer, ft_model = load_model_and_tokenizer(ft_model_path)
    
    print("\nEnter a legal case description or question (or 'quit' to exit):")
    print("Example: 'What is the outcome for the case Alpine Hardwood? Context: [case text]'")
    
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue
        
        # Format as T5 input if not already formatted
        if not user_input.startswith("Question:"):
            formatted_input = f"Question: {user_input}"
        else:
            formatted_input = user_input
        
        print("\nGenerating responses...")
        
        try:
            base_response = generate_response(formatted_input, base_tokenizer, base_model)
            ft_response = generate_response(formatted_input, ft_tokenizer, ft_model)
            
            print("\nRESPONSES:")
            print(f"Base Model (t5-small):")
            print(f"  {base_response}")
            print()
            print(f"Fine-tuned Model:")
            print(f"  {ft_response}")
        except Exception as e:
            print(f"Error generating response: {e}")

def show_sample_questions(json_path: str = 'legal_data.json'):
    """Display 5 sample questions without running the full pipeline."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions = create_sample_questions(data, num_questions=5)
        print("\n5 SAMPLE LEGAL QUESTIONS GENERATED FROM THE DATA")

        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}:")
            print(f"  Case: {q['title']}")
            print(f"  Expected outcome: {q['actual_outcome']}")
            print(f"  Expected response: {q['expected_response']}")
            print(f"  Input: {q['input_text'][:150]}...")
        return questions
    except FileNotFoundError:
        print("Data file not found. Please run preprocessing first.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Legal Text Generation Pipeline')
    parser.add_argument('--csv', default='legal_text_classification.csv', help='Path to CSV file')
    parser.add_argument('--mini', action='store_true', default=True, help='Use mini T5 model')
    parser.add_argument('--base', action='store_true', help='Use T5-base model (slower)')
    parser.add_argument('--test', action='store_true', help='Run interactive test mode')
    parser.add_argument('--show_questions', action='store_true', help='Show 5 sample questions')
    parser.add_argument('--skip_train', action='store_true', help='Skip training (assumes model already trained)')
    
    args = parser.parse_args()
    
    if args.show_questions:
        load_and_preprocess_data(args.csv, 'legal_data.json')
        show_sample_questions()
    elif args.test:
        if not os.path.exists('./fine_tuned_mini_model') and not args.skip_train:
            print("No fine-tuned model found. Running training first...")
            run_pipeline(args.csv, use_mini_model=True)
        interactive_test()
    else:
        use_mini = not args.base
        run_pipeline(args.csv, use_mini_model=use_mini)
```

---
## Results
<p><img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/27f92648-93db-4fa6-baa2-696c744de3c1" /></p>
<p><img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/799997d3-9ecd-468d-bfe4-9b369433da8e" /></p>
<p><img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/c957c4c7-eb5d-4132-9a79-57b8d9b7241c" /></p>
