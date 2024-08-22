from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import torch

from data import get_wikitext
from models import load_from_path

from tqdm import tqdm
from transformers import HfArgumentParser

from lm_eval import evaluator

def get_metric_values(results, task):
    task_results = results['results'][task]
    metric_key = None 
    for key in task_results:
        if ',' not in key:
            continue 

        metric_name, filter = key.split(',')
        if metric_name.endswith('stderr'):
            metric_name = metric_name.split('_')[0]
            metric_key = f'{metric_name},{filter}'
            break

    assert metric_key is not None, 'Could not find metric key'
    metric_name, filter = metric_key.split(',')
    print(f'Using metric {metric_name} with filter {filter} for task {task}')
    mean = task_results[metric_key] 
    stderr = task_results[f'{metric_name}_stderr,{filter}']

    return mean, stderr

def format_number(mean, stderr):
    return f"{mean:.2f}±{stderr:.2f}"

def evaluate_checkpoint(model_path: str, task: str, batch_size=1): 
    model = load_from_path(model_path)
    task_name = task
    num_fewshot = 0
    if '@' in task:
        task_name, num_fewshot = task.split('@')
        num_fewshot = int(num_fewshot)

    results = evaluator.simple_evaluate(
        model=model.as_lmeval_obj(batch_size=batch_size),
        tasks=[task_name],
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device="cuda",
    )
    return results

def eval_and_format(model_path: str, tasks: List[str], batch_size=1):
    all_results = {task: [] for task in tasks}
    all_results.update({f'{task}_mean': [] for task in tasks})
    all_results.update({f'{task}_stderr': [] for task in tasks})
    all_results['model_path'] = [model_path]
    for task in tasks:
        results = evaluate_checkpoint(model_path, task)
        mean, stderr = get_metric_values(results, task)
        all_results[f'{task}_mean'].append(mean)
        all_results[f'{task}_stderr'].append(stderr)
        all_results[task].append(format_number(mean, stderr))
    
    df = pd.DataFrame(all_results)
    return df 

def benchmark_forward_pass(model_path: str, use_cache=True, n_examples=500, n_burnin=25):
    # TODO: Proper device management 
    model = load_from_path(model_path)
    model.for_inference()

    def count_tokens(tokenizer):
        def f(example):
            return {'num_tokens': len(tokenizer.tokenize(example['text']))}
        
        return f

    data = get_wikitext(filter_nonempty=False)
    data = (
        data.train_data
        .map(count_tokens(model.tokenizer))
        .filter(lambda example: 240 <= example['num_tokens'] <= 260)
        .select(range(n_examples))
    )

    input_speeds = []
    output_speeds = []
    output_buffer = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def timer_hook(start=False):
        def register_forward_pass(layer, *args):
            if start:
                # torch.cuda.synchronize()
                start_event.record()
            else:
                end_event.record()
                torch.cuda.synchronize()

                hidden_states = args[0][0]
                n_tokens = hidden_states.size(1) 
                forward_pass_time = start_event.elapsed_time(end_event) / 1000
                speed = n_tokens / forward_pass_time
                if n_tokens > 1:
                    input_speeds.append(speed)
                else:
                    output_buffer.append(speed)
        
        return register_forward_pass

    layers = model.get_decoder_layers() 
    layers[0].register_forward_pre_hook(timer_hook(True))
    layers[-1].register_forward_hook(timer_hook(False))

    def compute_speed_metrics(speeds, n_burnin):
        speeds_tensor = torch.tensor(speeds)[n_burnin:]
        return speeds_tensor.mean(), speeds_tensor.std()

    with torch.no_grad():
        for x in tqdm(data):
            question: str = x["text"]

            tensors = model.tokenizer(question, return_tensors='pt', return_attention_mask=True)
            input_ids = tensors.input_ids.cuda()
            n_inputs = input_ids.size(1) 
            attention_mask = tensors.attention_mask.cuda()

            output = model.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=model.tokenizer.eos_token_id,
                use_cache=use_cache,
            )
            # Does this help give more stable results?
            torch.cuda.empty_cache()
            
            num_tokens_to_generate = 256
            output = model.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens_to_generate,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=model.tokenizer.eos_token_id,
                use_cache=use_cache,
            )
            torch.cuda.empty_cache()
            tokens_generated = output.size(1) - n_inputs
            if tokens_generated == num_tokens_to_generate:
                output_speeds.append(sum(output_buffer) / len(output_buffer))
                output_buffer = []
    
    return (
        *compute_speed_metrics(input_speeds, n_burnin),
        *compute_speed_metrics(output_speeds, n_burnin)
    ) 

def benchmark_and_format(model_path: str, use_cache=True, n_examples=500, n_burnin=25):
    imean, istd, omean, ostd = map(
        lambda x: x.item(),
        benchmark_forward_pass(model_path, use_cache, n_examples, n_burnin)
    )
    all_results = {'model_path': [model_path]}
    all_results['input_speed_mean'] = imean
    all_results['input_speed_stderr'] = istd 
    all_results['output_speed_mean'] = omean
    all_results['ouput_speed_stderr'] = ostd
    all_results['input_speed'] = format_number(imean, istd)
    all_results['output_speed'] = format_number(omean, ostd)

    return pd.DataFrame(all_results)

@dataclass 
class EvalConfig:
    model_path: str
    csv_out: str
    csv_append: bool = False
    tasks: Optional[str] = None
    benchmark: bool = False
    benchmark_n_examples: int = 500
    benchmark_n_burnin: int = 25
    batch_size: str = 'auto'

def main(config: EvalConfig):
    csv_out = Path(config.csv_out)
    if csv_out.exists():
        assert config.csv_append, 'Results file already exists, not overwriting'

    eval_df = None  
    benchmark_df = None 
    if config.tasks is not None:
        tasks = config.tasks.split(',')
        batch_size = 'auto' if config.batch_size == 'auto' else int(config.batch_size)
        eval_df = eval_and_format(config.model_path, tasks, batch_size=batch_size)
    if config.benchmark:
        benchmark_df = benchmark_and_format(config.model_path, n_examples=config.benchmark_n_examples, n_burnin=config.benchmark_n_burnin)

    result_df = None
    if eval_df is not None and benchmark_df is not None:
        result_df = eval_df.merge(benchmark_df, on='model_path')
    elif eval_df is not None:
        result_df = eval_df
    elif benchmark_df is not None:
        result_df = benchmark_df
    else:
        assert False, 'Either eval or benchmark should be true'
    
    if config.csv_append and csv_out.exists():
        prev_df = pd.read_csv(csv_out, index_col='model_path').reset_index()
        result_df = pd.concat((prev_df, result_df))

    result_df.to_csv(csv_out, index=False)

if __name__ == '__main__':
    config = HfArgumentParser(EvalConfig).parse_args_into_dataclasses()[0]
    main(config)
