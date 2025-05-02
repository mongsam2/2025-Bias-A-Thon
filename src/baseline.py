import re
import ast
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

data = pd.read_csv("./data/test.csv", encoding="utf-8-sig")


# Model Load
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # 본 대회는 반드시 Llama-3.1-8B-Instruct 모델만 사용해야 합니다.

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", use_auth_token=True
)


# 프롬프트 생성 함수
def make_prompt(context, question, choices):
    choices = ast.literal_eval(choices)

    # 프롬프트를 수정하여 모델에 전달할 수 있습니다.
    # 예시 프롬프트
    return f"""질문에 대해서 다음 선택지 중 반드시 하나만 답하시오. 다른 선택지는 고려하지 마시오.

            질문 : {context} {question}
            선택지: {choices[0]} ,{choices[1]} ,{choices[2]}

            답변:"""


# 정답 추출 함수
def extract_answer(text):
    raw_answer = text.split("답변:")[-1].strip()  # 프롬프트를 제외한 답변만 추출
    result = re.search(r"답변:\s*([^\n\r:]+)", text)  # 정규 표현식으로 답변 추출
    answer = result.group(1).strip() if result else None
    return raw_answer, answer


# 추론 함수
def predict_answer(context, question, choices, max_new_tokens=16):
    prompt = make_prompt(context, question, choices)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(output[-1], skip_special_tokens=True)
    raw_answer, answer = extract_answer(result)

    return pd.Series({"raw_input": prompt, "raw_output": raw_answer, "answer": answer})


# 한 줄씩 처리 및 저장
for i in range(len(data)):
    row = data.loc[i]
    result = predict_answer(row["context"], row["question"], row["choices"])

    # 결과 저장
    data.at[i, "raw_input"] = result["raw_input"]
    data.at[i, "raw_output"] = result["raw_output"]
    data.at[i, "answer"] = result["answer"]

    # 5000개마다 중간 저장
    if i % 5000 == 0:
        print(f"✅ Processing {i}/{len(data)} — 중간 저장 중...")
        data[["ID", "raw_input", "raw_output", "answer"]].to_csv(
            f"submission_checkpoint_{str(i)}.csv", index=False, encoding="utf-8-sig"
        )

submission = data[["ID", "raw_input", "raw_output", "answer"]]
submission.to_csv("baseline_submission.csv", index=False, encoding="utf-8-sig")
