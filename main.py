import torch
from peft import PeftConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

if __name__ == '__main__':

    checkpoint = "letter/checkpoint-1600"
    config = PeftConfig.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, checkpoint)
    print("[Info] Model loaded.")

    # prompt = '为"粤港澳大湾区人工智能与机器人大会"活动写一篇演讲讲者邀请函'
    prompt = 'Write an invitation letter for the keynote speakers at the "GBA Artificial Intelligence and Robotics Conference"'

    tokens = tokenizer(prompt, return_tensors="pt")

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=650,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True)

    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(output_text)
