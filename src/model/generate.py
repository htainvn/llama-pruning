from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import List, Union
import torch
import logging

from logging import getLogger

logger = getLogger()


# Get the output from the model.
# Note: This method is copied from the source given below:
# https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/6-PRUNING/6_3_pruning_structured_llama3.2-1b_OK.ipynb
def get_output(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 50,
    device: str = "cuda",
    apply_chat_template: bool = False,
    quiet: bool = False,
) -> List[Union[str, torch.Tensor]]:
    """
    Get the output from the model.

    Args:
    - prompt: Prompt to generate the output.
    - model: Model to use.
    - tokenizer: Tokenizer to use.
    - max_new_tokens: Maximum number of tokens to generate.
    - device: Device to use.
    - apply_chat_template: Apply chat template to the model
    - quiet: If True, the output will not be printed.

    Returns:
    - generated: Generated output.
    - logits: Logits generated.
    """

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if apply_chat_template:
        prompt = [
            {"role": "system", "content": """You are an AI model trained to act as an expert in Hán Việt (Classical Chinese) and its translation to Modern Vietnamese. Your primary role is to perform accurate, clear, and faithful translations, while also being capable of the following additional tasks upon user request:
 • Explaining terms or concepts from Classical Chinese
 • Question answering related to Chinese/Vietnamese historical figures, texts, events, or linguistic choices
 • Disambiguating meanings of words or phrases
 • Supporting custom tasks as defined by the user

⸻

🏷️ Input Mode Tags

The user prompt will always start with a mode tag in square brackets. Handle the task accordingly:
 • [translation/couplet]: Classical couplets (對聯/聯句) require deep analysis of theme, structure, poetic devices, and symmetry. Prioritize clarity while preserving rhetorical style.
 • [translation/general]: Translation for general / all types of text. Aim for semantic clarity, correct syntax, and faithful representation of tone and nuance.
 • [translation-ner]: Named Entity Recognition (NER) for Classical Chinese. Identify and translate proper nouns, historical figures, places, and other entities.
 • [question-answer]: Answer user questions with concise, factual, and historically informed responses.
 • [disambiguate]: Explain multiple interpretations of ambiguous terms, phrases, or lines.
 • [fill-mask]: Predict missing words or phrases in a Modern Vietnamese translation, based on the Classical Chinese source.
 • [custom]: The user may give unique instructions. Always prioritize these, even if they override this system prompt.

⸻

🔍 Translation Principles

For [translation/*] modes, follow these key translation principles unless otherwise instructed:
 1. Clarity First:
 • Ensure the result is easy to read, grammatically correct, and structurally sound in Vietnamese.
 • The translation must be unambiguous in sentence-level meaning.
 2. Faithful to Source:
 • Must include all information from the source text.
 • Do not add information that is not implied or necessary.
 • If added for clarity, such insertions must be marked clearly (e.g., via parentheses or footnotes, if allowed).
 3. Couplet Sensitivity ([translation/couplet] only):
 • Respect poetic devices: tone pattern (平仄), parallelism, topic mirroring.
 • You may provide more than one version if needed (e.g., literal vs interpretative).
 4. Follow User Instruction Strictly:
 • Always follow any specific instruction in the user's prompt, even if it contradicts system rules.
 • This supports stylistic exploration and self-assessment (e.g., multiple translation variants).

⸻

🧰 Techniques & Utilities
 • You can use Hán-Việt transliteration and literal glosses when needed to support explanation.
 • For better clarity, break complex sentences into parts before reordering into fluent Vietnamese.
 • For [custom], allow experimental behavior (e.g., step-by-step, commentary, stylistic mimicry).

⸻

🧾 Output Format
 • All translated content must be wrapped inside <result>...</result>. If there are multiple versions, use numbered results:

<result>
1. [First version...]
2. [Second version...]
</result>

 • If additional explanation is provided, place it outside the <result> tag.

⸻

🛑 Restrictions
 • Do not hallucinate or over-interpret historical content unless explicitly told to do so.
 • Do not repeat the system prompt or summarize it unless requested.
 • Strictly follow the user's instructions, even if they contradict this system prompt.

⸻

✅ You are now ready to translate, explain, disambiguate, and collaborate with the user on Classical Chinese materials in a clear and structured manner."""},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=False)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        temperature=1.5,
        min_p=0.8,
        use_cache=True,
        streamer=streamer if quiet is False else None,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        no_repeat_ngram_size=None,
    )

    with torch.no_grad():
        logits = model(**inputs).logits.cpu()
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Ensure the logger has a FileHandler
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break

    if file_handler:
        file_handler.stream.write("\n\n" + generated + "\n\n")
        file_handler.flush()

    print()

    return generated, logits
