import streamlit as st
import torch
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, REPETITION_PENALTY

def generate_answer(query, context_docs, gen_model, tokenizer):
    """Generates an answer using the LLM based on query and context."""
    if not context_docs:
        return "我找不到相关文档来回答您的问题。"
    if not gen_model or not tokenizer:
         st.error("生成模型或分词器不可用。")
         return "错误：生成组件未加载。"

    context = "\n\n---\n\n".join([doc['content'] for doc in context_docs]) # Combine retrieved docs

    prompt = f"""仅基于以下上下文文档，回答用户的问题。
如果答案未在上下文中找到，请明确说明，不要编造信息。

上下文文档：
{context}

用户问题：{query}

回答：
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id # Important for open-end generation
            )
        # Decode only the newly generated tokens, excluding the prompt
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "Sorry, I encountered an error while generating the answer." 