import ast
import os


def make_prompt(context, question, choices, text_file_name):
    choices = ast.literal_eval(choices)  # 문자열 안에 담겨있는 리스트를 변환

    prompt_text = load_prompt_text(text_file_name)
    return f"""{prompt_text}
    
            질문 : {context} {question}
            선택지: {choices[0]} ,{choices[1]} ,{choices[2]}"""


def load_prompt_text(file_name):

    file_path = os.path.join("code/resources", file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
