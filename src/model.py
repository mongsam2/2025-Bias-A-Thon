import re
import os
import ast
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaModel:

    def __init__(self, title, prompt_file_name):
        self.title = title
        self.prompt_file_name = prompt_file_name
        self.data = pd.read_csv("src/resources/test.csv", encoding="utf-8-sig")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            device_map="cuda",
            token=True,
            torch_dtype=torch.float16,
        )

    def run(self, is_test):
        # 실험 결과 OR 파이프라인 테스트용 폴더 생성
        submission_dir = (
            os.path.join("submissions", self.title)
            if not is_test
            else os.path.join("test", self.title)
        )
        os.makedirs(submission_dir, exist_ok=True)

        inference_data = self.data if not is_test else self.data.iloc[:10].copy()
        checkpoint_count = 5000 if not is_test else 5

        for i in tqdm(range(len(inference_data)), desc="Inference"):
            row = inference_data.loc[i]
            context = row["context"]
            question = row["question"]
            choices = ast.literal_eval(row["choices"])
            result = self.__inference(context, question, choices)

            # 결과 저장
            inference_data.at[i, "raw_input"] = result["raw_input"]
            inference_data.at[i, "raw_output"] = result["raw_output"]
            inference_data.at[i, "answer"] = result["answer"]

            # 5000개마다 중간 저장
            if i % checkpoint_count == 0:
                submission_path = os.path.join(submission_dir, f"checkpoint_{i}.csv")

                # print(f"✅ Processing {i}/{len(inference_data)} — 중간 저장 중...")

                inference_data[["ID", "raw_input", "raw_output", "answer"]].to_csv(
                    submission_path, index=False, encoding="utf-8-sig"
                )

        submission = inference_data[["ID", "raw_input", "raw_output", "answer"]]
        submission.to_csv(
            os.path.join(submission_dir, "submission.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    def __inference(self, context, question, choices, max_new_tokens=16):
        prompt = self.__set_prompt(context, question, choices)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device
        )  # 모델 입력용 토큰으로 변환

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        result = self.tokenizer.decode(output[-1], skip_special_tokens=True)
        raw_answer, answer = self.__extract_answer(result, choices)

        return pd.Series(
            {"raw_input": prompt, "raw_output": raw_answer, "answer": answer}
        )

    def __set_prompt(self, context, question, choices):
        file_path = os.path.join("src", "resources", self.prompt_file_name + ".txt")
        with open(file_path, "r", encoding="utf-8") as file:
            prompt_text = file.read()

        return (
            f"{prompt_text}\n"
            f"질문: {context} {question}\n"
            f"선택지: {choices[0]}, {choices[1]}, {choices[2]}\n"
        )

    def __extract_answer(self, text, choices):
        response = text.split("[풀어야 할 문제]\n")[1]
        raw_answer = response  # 예시 프롬프트를 제외한 답변만 추출
        response_splited = response.split("답변: ")[1]
        result = "알 수 없음" if len(response_splited) == 0 else response_splited[1]
        answer = "알 수 없음"

        for choice in choices:
            if choice in result:
                answer = choice

        return raw_answer, answer
