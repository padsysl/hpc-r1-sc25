# Copyright 2025 The HuggingFace Team. All rights reserved.
# Copyright 2025 PADSYS Laboratory. All rights reserved. (Modifications)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Custom imports
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from dataclasses import dataclass
import os
import logging
import signal
from typing import Any, Dict, List
import time
import datetime
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import torch
import torch.distributed as dist

import inspect
print(f"SANITY CHECK 1: inspect={inspect.getmodule(torch.profiler.profile)} ;;; {torch.profiler.profile.__mro__=}")
assert torch.profiler.profile.__mro__[1] is torch.profiler._KinetoProfile # If this is not the case, then the OLD type of profiler is probably in use, which is not going to work.

# Globals
trainer = None
job_name = "UnknownName"
job_id = "UnknownJobId"
world_rank = None
world_size = None
trace_has_been_saved = False

chrometrace_storage_dir = False
simpletrace_storage_dir = False


# Signal handler
def signal_handler(sig, frame):
    print(f"[DEBUG] Signal detected ({sig}), saving stats...")
    save_stats()
    print("[DEBUG] Stats saved successfully!")
    print("[DEBUG] Exiting training script...")
    exit()


# Function to handle pre-exit saving
def save_stats():
    """CURRENTLY BROKEN"""

    print("[WARNING] save_stats is currently broken and is a no-op")
    return
    
    csv_output_path = f"./logs/stats_rank[{world_rank}].csv"

    try:
        import pandas as pd
        df = pd.DataFrame(trainer.state.log_history)
        print("[DEBUG] Trainer State Log History:")
        print(df)
        
        print(f"[INFO] Writing CSV with log history to {csv_output_path}")
        df.to_csv(csv_output_path)
    except Exception as e:
        print(f"[WARN] Failed to run custom `trainer.state.log_history` or output to CSV: {e}")

    print(f"Stats saved to {csv_output_path}, exiting.")


def live_format_promt_fb_natural_reasoning(example: Dict[str, List[Dict[str, str]]]) -> str:
    """Live-format the dataset into a list of strings, each representing a prompt.

    Note: This is a custom formatting function for the Facebook Natural Reasoning dataset. It is designed
          to be used when the dataset has not been pre-formatted and is instead loaded from the Hugging Face 
          Hub or something similar.

    There are three columns in the dataset:
    - question: The question to answer
    - reference_answer: The /final/ answer to the question (does not include reasoning)
    - responses: A list of dictionaries [{'response_model': str, 'response': str}, ...], each representing a response to the 
                 question. A response is formatted as follows:
                 {
                     "response_model": str,
                     "response": str (but formatted as Markdown),
                 }

    The goal of this function is to format the dataset into a list of strings, each representing a prompt.

    ---

    Each prompt should be formatted as follows:

    # Question: 
    <question>
    # Response:
    <response_1>

    ---

    Note that we ignore the reference answer and only use the first response from the `responses` list (for now).
    """

    # Get the question
    question = example['question']

    # Get the first response
    response = example['responses'][0]['response']

    # Format the prompt
    prompt = f"# Question: {question}\n# Response:\n{response}"

    return prompt


def r1_ds_formatting_prompts_func(example):
    """Format the dataset into a list of strings, each representing a prompt.
    """

    output_texts = []
    for i in range(len(example['problem'])):
        text = f"### Question: {example['problem'][i]}\n{example['reannotated_assistant_content']}\nAnswer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts


def open_r1_ds_formatting_prompts_func(example):
    """Format the dataset into a list of strings, each representing a prompt.
    """

    return f"{{\"prompt\": \"{example['problem']}\", \"completion\": \"{example['solution']}\"}}"


def dataset_format_router(example: Dict[str, Any]) -> str:
    """Route to the correct dataset formatting function.
    """

    # If using the standard SFT format, immediately return (no need to format or route)
    if 'system' in example and 'conversations' in example:
        return example

    # Verify that the example has the correct format (keys)
    warn = False
    if 'data_source' not in example:
        warn = True
        print("[WARN] Example does not have the correct format (key: 'data_source')")

    if 'text' not in example:
        warn = True
        print("[WARN] Example does not have the correct format (key: 'text')")

    # Just return the example if it doesn't have the correct format
    if warn:
        return example

    # Otherwise, route to the correct formatting function
    if example['data_source'] == 'facebook/natural_reasoning':
        return example['text']  # We already formatted the text in the data prep script! We don't need to do it again here.
    elif example['data_source'] == 'open-r1/OpenR1-Math-220k':
        return example['text']  # Math220K is formatted the same way as the above if the preprocessing script was properly used
    else:
        if "text" not in example.keys():
            raise ValueError(f"No formatting function found for dataset: {example['data_source']}")
        else:
            return example['text']  # Just give it a try. If it's not in the keys, then we can error then instead of preemptively 


def test_filelock(dataset_path: str) -> None:
    """Test the filelock library.
    """
    from filelock import FileLock

    # Get the parent directory of the dataset path
    parent_dir = os.path.dirname(dataset_path)

    # Create a file if it doesn't exist
    lock = FileLock(os.path.join(parent_dir, "high_ground.txt.lock"), timeout=10)
    with lock:
        with open(os.path.join(parent_dir, "high_ground.txt"), "a") as f:
            f.write("You were the chosen one.")


def test_dataset_prep(dataset_path: str) -> None:
    """Test the dataset preparation.
    """
    datasets.utils.logging.set_verbosity_info()

    # Test the filelock library
    print("[DEBUG] Testing the filelock library...")
    test_filelock(dataset_path)
    print("[DEBUG] Filelock library test completed successfully!")

    # Checking status of offline mode
    hf_datasets_offline = os.getenv('HF_DATASETS_OFFLINE') == '1'
    transformers_offline = os.getenv('TRANSFORMERS_OFFLINE') == '1'
    hf_hub_offline = os.getenv('HF_HUB_OFFLINE') == '1'
    if hf_datasets_offline:
        print(f"[DEBUG] Offline Mode: `HF_DATASETS_OFFLINE` is set to {hf_datasets_offline}")
    if transformers_offline:
        print(f"[DEBUG] Offline Mode: `TRANSFORMERS_OFFLINE` is set to {transformers_offline}")
    if hf_hub_offline:
        print(f"[DEBUG] Offline Mode: `HF_HUB_OFFLINE` is set to {os.getenv('HF_HUB_OFFLINE')}")

    # Check if the given dataset is on the local machine
    dataset_path = os.path.join(os.getcwd(), dataset_path)
    if os.path.exists(dataset_path):
        print(f"[DEBUG] Dataset found on LOCAL MACHINE at {dataset_path}")
    elif hf_datasets_offline:
        print(f"[DEBUG] Dataset not found on local machine at {dataset_path}, but `HF_DATASETS_OFFLINE` is set to 1, so I will not try to download it from HF.")
        raise RuntimeError("[ERROR] Please set `HF_DATASETS_OFFLINE=0` to download the dataset from HF or move the dataset to the local machine.")
    else:
        raise RuntimeWarning(f"[WARNING] Dataset not found on local machine at {dataset_path}, will probably try to download it from HF.")


# Callback to inspect and modify training loop things with SFTTrainer
class CustomCallback(TrainerCallback):
    def __init__(self):
        print("[TRACE] Initializing CustomCallback...")

        self.training_timer = None
        self.epoch_timer = None
        self.step_timer = None
                
        print("[TRACE] Initialized CustomCallback.")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_timer = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_timer is not None:
            elapsed = time.time() - self.epoch_timer
            self.epoch_timer = None
            print(f"[TRACE] Finished epoch in {elapsed} seconds.")
        else:
            print("[ERROR] For some reason, self.epoch_timer was none. Can't print time elapsed.")

    def on_evaluate(self, args, state, control, **kwargs):
        print(f"[TRACE] Evaluating at {datetime.datetime.now()}")

    def on_init_end(self, args, state, control, **kwargs):
        pass
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

    def on_optimizer_step(self, args, state, control, **kwargs):
        # print("[TRACE] Did optimizer step.")
        pass

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        pass

    def on_predict(self, args, state, control, metrics, **kwargs):
        pass

    def on_prediction_step(self, args, state, control, **kwargs):
        pass

    def on_save(self, args, state, control, **kwargs):
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        print(f"[TRACE] Step begin at {datetime.datetime.now()}")
        self.step_timer = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_timer is not None:
            elapsed = time.time() - self.step_timer
            self.step_timer = None
            print(f"[TRACE] Finished step at {datetime.datetime.now()} in {elapsed} seconds.")
        else:
            print("[ERROR] For some reason, self.step_timer was none. Can't print time elapsed.")

    def on_substep_end(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_timer = time.time()
        print(f"[TRACE] Started training at {datetime.datetime.now()}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.training_timer is not None:
            elapsed = time.time() - self.training_timer
            print(f"[TRACE] Finished training at {datetime.datetime.now()} in {elapsed} seconds.")
        else:
            print("[ERROR] For some reason, self.training_timer was none. Can't print time elapsed.")


def main(script_args, training_args, model_args):
    # Set logging and verbosity levels
    if os.getenv('CUST_VERBOSITY') == 'DEBUG':
        print("[DEBUG] Setting verbosity to DEBUG")
        logging.basicConfig(level=logging.DEBUG)
        datasets.utils.logging.set_verbosity_debug()
    elif os.getenv('CUST_VERBOSITY') == 'INFO':
        print("[DEBUG] Setting verbosity to INFO")
        logging.basicConfig(level=logging.INFO)
        datasets.utils.logging.set_verbosity_info()
    elif os.getenv('CUST_VERBOSITY') == 'WARNING':
        print("[DEBUG] Setting verbosity to WARNING")
        logging.basicConfig(level=logging.WARNING)
        datasets.utils.logging.set_verbosity_warning()
    elif os.getenv('CUST_VERBOSITY') == 'ERROR':
        print("[DEBUG] Setting verbosity to ERROR")
        logging.basicConfig(level=logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
    else:
        print("[DEBUG] Setting verbosity to WARNING (default). Use CUST_VERBOSITY=(DEBUG|INFO|WARNING|ERROR) to change.")
        logging.basicConfig(level=logging.WARNING)
        datasets.utils.logging.set_verbosity_warning()

    # Print that I am alive
    try:
        import socket
        print(f"[TRACE] I am on hostname={socket.gethostname()} and I am ALIVE!")
    except Exception as e:
        print(f"[TRACE] I am alive, but I couldn't read my hostname for some reason: {e}")

    # Get world rank and local node rank
    global world_size
    global world_rank
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    node_rank = os.environ.get('LOCAL_RANK',-1)
    print(f"[DEBUG] World rank: {world_rank}, Node-local rank: {node_rank}")
    
    # Barrier to ensure all processes are ready
    print(f"[DEBUG] Rank {world_rank} is waiting for barrier 1...")
    dist.barrier()
    print(f"[DEBUG] Rank {world_rank} ({world_size} total ranks) passed barrier 1!")

    # Check if CUDA is available (if we're using a GPU)
    if not torch.cuda.is_available():
        print("[ERROR] CUDA/GPU is not available! Exiting...")
        exit(1)
    else:
        print(f"[DEBUG] CUDA is available! Rank {world_rank} is using {torch.cuda.device_count()} GPUs.")

    # Check if set to offline mode
    hf_datasets_offline = os.getenv('HF_DATASETS_OFFLINE') == '1'
    transformers_offline = os.getenv('TRANSFORMERS_OFFLINE') == '1'
    hf_hub_offline = os.getenv('HF_HUB_OFFLINE') == '1'
    
    # Get information about the job, or just use a default
    global job_name
    global job_id
    job_name = os.getenv("SLURM_JOB_NAME", default="UnknownJobName")
    job_id = os.getenv("SLURM_JOB_ID", default="UnknownJobId")
    
    # Test the dataset preparation, but only on the first process
    if world_rank == 0:
        print(f"[DEBUG] Running dataset test preparation on rank {world_rank}...")
        try: 
            test_dataset_prep(dataset_path=script_args.dataset_name)
        except Exception as e:
            print(f"[ERROR] Failed to run dataset test preparation: {e}")
        print("[DEBUG] Dataset test preparation completed successfully!")

    # Barrier to ensure all processes have finished the dataset test preparation
    print(f"[DEBUG] Rank {world_rank} is waiting for barrier 2...")
    dist.barrier()
    print(f"[DEBUG] Rank {world_rank} ({world_size} total GPUs) passed barrier 2!")
    
    ################
    # Model init kwargs & Tokenizer
    ################
    using_fsdp = True if os.getenv("USING_FSDP", default="0") == "1" else False
    print(f"[INFO] {using_fsdp=}")
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing or using_fsdp else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        local_files_only=hf_hub_offline,
    )
    # training_args.model_init_kwargs = model_kwargs  # IMPORTANT: Uncomment if not loading model separately from SFTTrainer!

    # Tokenizer
    print(f"[DEBUG] Loading tokenizer from {model_args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True, 
        local_files_only=hf_datasets_offline
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("[DEBUG] Tokenizer loaded successfully!")

    ################
    # Dataset
    ################
    print(f"[DEBUG] Loading dataset from {script_args.dataset_name}...")
    start = time.time()
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # dataset = load_from_disk(script_args.dataset_name)
    elapsed = time.time() - start
    print(f"[DEBUG] Dataset loaded successfully in {elapsed:.2f} seconds!")

    # ################
    # # Tokenize dataset
    # ################
    # print(f"[DEBUG] Tokenizing dataset...")
    # dataset = dataset.map(lambda x: tokenizer(x["text"], padding=True, truncation=True, max_length=4096), batched=True)
    # print(f"[DEBUG] Dataset tokenized successfully!")

    ################
    # Model
    ################
    print(f"[DEBUG] Loading model from {model_args.model_name_or_path}...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # low_cpu_mem_usage=True,  # HACK: Might help with CPU OOM issues while loading FSDP model? Doesn't work with ZeRO though.
        # torch_dtype=torch.bfloat16,
        **model_kwargs)
    elapsed = time.time() - start
    print(f"[DEBUG] Model loaded successfully in {elapsed:.2f} seconds!")

    # # Get some information about the model
    # for name, param in model.named_parameters():
    #     print(f"[DEBUG] MODEL INFO   {name}: {param.shape}")

    # # Freeze all parameters except for the last 100
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     if name.startswith("decoder.final_layer_norm") or i > len(model.parameters()) - 100:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    ################
    # Training
    ################
    print("[DEBUG] Initializing trainer...")
    trainer = SFTTrainer(
        # model=model_args.model_name_or_path,
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        formatting_func=dataset_format_router,
    )
    print("[DEBUG] Trainer initialized successfully!")

    # Wait for all processes to finish initializing the trainer
    print(f"[DEBUG] Rank {world_rank} is waiting for barrier 3...")
    start = time.time()
    dist.barrier()
    elapsed = time.time() - start
    print(f"[DEBUG] Rank {world_rank} ({world_size} total GPUs) passed barrier 3! (waited {elapsed:.2f} seconds)")

    # Check if we should resume from a checkpoint
    # FIXME: Actually implement this!
    if training_args.resume_from_checkpoint is not False and training_args.resume_from_checkpoint is not None and training_args.resume_from_checkpoint != "":
        print(f"[WARNING] Found resume checkpoint: {training_args.resume_from_checkpoint}, but THIS IS NOT CURRENTLY IMPLEMENTED!")
    else:
        print("[DEBUG] Not resuming from a checkpoint.")

    print("[DEBUG] Training model...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"[DEBUG] Training completed successfully in {elapsed:.2f} seconds!")

    # Wait for all processes to finish training
    print(f"[DEBUG] Rank {world_rank} is waiting for barrier 4...")
    dist.barrier()
    print(f"[DEBUG] Rank {world_rank} ({world_size} total GPUs) passed barrier 4!")
    
    # Save as a safetensors file
    if world_rank == 0:
        print(f"[DEBUG] Saving model to {training_args.output_dir}...")
        start = time.time()
        trainer.save_model(training_args.output_dir)
        elapsed = time.time() - start
        print(f"[DEBUG] Model saved successfully in {elapsed:.2f} seconds!")
    
    # Barrier
    print(f"[DEBUG] Rank {world_rank} is waiting for barrier 5...")
    dist.barrier()
    print(f"[DEBUG] Rank {world_rank} ({world_size} total GPUs) passed barrier 5!")

    print("[DEBUG] Main function completed successfully!")
    
    # Check end conditions
    global trace_has_been_saved
    if not trace_has_been_saved:
        print("[WARNING] Chrome trace may not have been saved!")


# Custom arguments
@dataclass
class CustomArguments:
    chrometrace_storage_dir: str = "./log"
    simpletrace_storage_dir: str = "./log"


if __name__ == "__main__":
    print("[DEBUG] Started training script!")

    # Set signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse CLI args
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_args, custom_args = parser.parse_args_and_config()

    # Print training args
    print(f"[DEBUG] Training args: {training_args}")

    # Get specific script args
    print(f"[DEBUG] Custom args (note: we probably won't use these because we're not profiling): {custom_args}")
    chrometrace_storage_dir
    simpletrace_storage_dir
    chrometrace_storage_dir = custom_args.chrometrace_storage_dir
    simpletrace_storage_dir = custom_args.simpletrace_storage_dir
    
    # Run main but handle KeyboardInterrupt
    try:
        main(script_args, training_args, model_args)
    except KeyboardInterrupt:
        print("[DEBUG] Exiting training script...")
        exit()

    print("[DEBUG] Training script completed successfully!")

