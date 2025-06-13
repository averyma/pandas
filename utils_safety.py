from enum import Enum
from typing import List, Tuple
import logging
from utils_prompt_format import (
    LLAMA_GUARD_3_CATEGORY, SafetyCategory, AgentType,
    build_custom_prompt, create_single_turn_conversation,
    PROMPT_TEMPLATE_3,LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX
)

class LG3Cat(Enum):
    VIOLENT_CRIMES =  0
    NON_VIOLENT_CRIMES = 1
    SEX_CRIMES = 2
    CHILD_EXPLOITATION = 3
    DEFAMATION = 4
    SPECIALIZED_ADVICE = 5
    PRIVACY = 6
    INTELLECTUAL_PROPERTY = 7
    INDISCRIMINATE_WEAPONS = 8
    HATE = 9
    SELF_HARM = 10
    SEXUAL_CONTENT = 11
    ELECTIONS = 12
    CODE_INTERPRETER_ABUSE = 13

def get_lg3_categories(category_list: List[LG3Cat] = [], all: bool = False, custom_categories: List[SafetyCategory] = [] ):
    categories = list()
    if all:
        categories = list(LLAMA_GUARD_3_CATEGORY)
        categories.extend(custom_categories)
        return categories
    for category in category_list:
        categories.append(LLAMA_GUARD_3_CATEGORY[LG3Cat(category).value])
    categories.extend(custom_categories)
    return categories

def evaluate_safety(tokenizer, model, agent_type, prompt = "", category_list = [], categories = [], verbose = False):
    # prompt = [([prompt], AgentType.USER)]
    prompt = [([prompt])]
    if categories == []:
        if category_list == []:
            categories = get_lg3_categories(all = True)
        else:
            categories = get_lg3_categories(category_list)
    formatted_prompt = build_custom_prompt(
            agent_type = agent_type,
            conversations = create_single_turn_conversation(prompt[0], agent_type),
            # agent_type = AgentType.USER,
            # conversations = create_conversation(prompt[0]),
            categories=categories,
            category_short_name_prefix = LLAMA_GUARD_3_CATEGORY_SHORT_NAME_PREFIX,
            prompt_template = PROMPT_TEMPLATE_3,
            with_policy = True)
    if verbose:
        logging.info("Prompt:\n" + formatted_prompt)
    input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    prompt_len = input["input_ids"].shape[-1]
    output = model.generate(**input, max_new_tokens=100, pad_token_id=0,
                            eos_token_id=128009 # TO BE REMOVED
                            )
    results = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    if verbose:
        logging.info("Results:\n" + results)
    return results