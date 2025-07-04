from peft import PromptTuningConfig, TaskType

def get_prompt_config():
    return PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init="RANDOM",
        num_virtual_tokens=16,
        tokenizer_name_or_path="bert-base-uncased"
    )
