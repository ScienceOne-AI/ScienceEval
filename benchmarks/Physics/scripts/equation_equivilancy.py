import signal
from sympy import simplify, expand, trigsimp
from sympy.parsing.latex import parse_latex
import re
import os
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

import requests
def get_llm_response(messages, url, model, key):
    """ 获取大模型生成的原始内容 """
    url = url.rstrip("/") + "/chat/completions"
    response =  requests.post(url=url, json={
        "model": model,
        "messages": messages,
        'max_tokens': 10000,
    }, headers={
        "Authorization": f"Bearer {key}",
        "x-ark-moderation-scene": "skip-ark-moderation"
    },timeout=120)
    
    return response.json()
def timeout_handler(signum, frame):
    """Handler for timeout protection."""
    raise TimeoutError("SymPy computation took too long!")

def _extract_core_eq(expr: str) -> str:
    """Extract the right-hand side of an equation or implication from a LaTeX expression."""
    if "\implies" in expr:
        expr = expr.split("\implies")[-1].strip()
    if "=" in expr:
        expr = expr.split("=")[-1].strip()
    return expr

def _preprocess_latex(string: str) -> str:
    """Preprocess LaTeX to normalize format and separate variables."""
    if not string:
        return ""
    
    string = re.sub(r"_\{.*?\}", "", string)
    string = re.sub(r"_\\?\w", "", string)
    string = string.replace("\left", "").replace("\right", "").replace("\cdot", "*")
    return string

def _standardize_expr(expr):
    """Standardize a SymPy expression with timeout protection."""
    try:
        signal.signal(signal.SIGALRM, timeout_handler)  # Set timeout signal
        signal.alarm(10)  # Set timeout to 10 seconds
        result = simplify(expand(trigsimp(expr)))
        signal.alarm(0)  # Cancel timeout
        return result
    except TimeoutError:
        signal.alarm(0)  # Cancel timeout
        raise ValueError("SymPy computation timed out!")
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        raise ValueError(f"SymPy error: {e}")

def call_llm_to_compare(expr1: str, expr2: str, llm_answers,judge_api_url, judge_api_key,judge_model) -> bool:
    """Use an LLM to determine if two LaTeX expressions with text are equivalent."""
    try:
        # print('123')
        messages=[
                {"role": "system", "content": "You are an assistant that compares LaTeX expressions for equivalence."},
                {"role": "user", "content": f"Compare the following LaTeX expressions and check if the numerical part are same meaning content:\n\nExpression 1:\n{expr1}\n\nExpression 2:\n{expr2}.\n\n Return True if they are equivalent, otherwise return False. focus on numerical and mathematical content. If it's multiple choice answer like a b c d, focus only on the letters.This is the solution process of Expression 1, which may include the definitions of some symbols:\n{llm_answers}."}
            ]
        response = get_llm_response(messages=messages,url=judge_api_url, model=judge_model, key=judge_api_key)
        # print(response)
        answer = response["choices"][0]["message"].get("content")
        return "true" in answer.lower()
    except Exception as e:
        return False  # Default to False if LLM fails

def is_equiv(expr1: str, expr2: str, llm_answers, dataset_answers,judge_api_url, judge_api_key,judge_model,verbose: bool = False) -> dict:
    """
    Compare two LaTeX expressions for equivalence and handle errors gracefully.
    """
    result_data = {
        "input_expressions": {"expr1": expr1, "expr2": expr2},
        "preprocessed_expressions": {},
        "sympy_result": None,
        "llm_result": None,
        "final_result": None,
        "error": None,
    }
    
    try:
        # print('==========')
        if "\text" in expr1 or "\text" in expr2:
            # print('-------------')
            result_data["llm_result"] = call_llm_to_compare(expr1, expr2, llm_answers,judge_api_url, judge_api_key,judge_model)
            result_data["final_result"] = result_data["llm_result"]
            return result_data
        
        expr1_processed = _preprocess_latex(expr1)
        expr2_processed = _preprocess_latex(expr2)
        expr1_core = _extract_core_eq(expr1_processed)
        expr2_core = _extract_core_eq(expr2_processed)
        
        try:
            sympy_expr1 = _standardize_expr(parse_latex(expr1_core))
            sympy_expr2 = _standardize_expr(parse_latex(expr2_core))
            result_data["preprocessed_expressions"] = {"expr1": str(sympy_expr1), "expr2": str(sympy_expr2)}
            
            # Set timeout protection
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)  # Set 10-second timeout
            sympy_result = simplify(sympy_expr1 - sympy_expr2) == 0
            sympy_result = sympy_result or sympy_expr1.equals(sympy_expr2)
            signal.alarm(0)  # Cancel timeout
        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            result_data["error"] = "SymPy computation timed out!"
            sympy_result = None
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            result_data["error"] = str(e)
            sympy_result = None
        
        result_data["sympy_result"] = sympy_result
        
        if sympy_result is not None and sympy_result:
            result_data["final_result"] = sympy_result
        else:
            # print(sympy_result)
            result_data["llm_result"] = call_llm_to_compare(expr1, expr2, llm_answers,judge_api_url, judge_api_key,judge_model)
            result_data["final_result"] = result_data["llm_result"]
        
    except Exception as e:
        result_data["error"] = str(e)
    
    return result_data


# Example tests
if __name__ == "__main__":
    # Example 1: Mathematical expressions
    latex1 = r"x = \sqrt{2 \mu h R}"
    latex2 = r"\sqrt{2\mu Rh}"
    result = is_equiv(latex1, latex2, verbose=True)
    print(result)

    # Example 2: Expressions with text
    latex3 = r"\\text{(a) Electron spin-orbit coupling"
    latex4 = r"(a)"
    result = is_equiv(latex3, latex4, verbose=True)
    print(result)
