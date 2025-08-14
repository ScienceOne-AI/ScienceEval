import re
from typing import List

def preprocess_latex(latex_text: str) -> str:
    boxed_pattern = re.compile(r'(\\boxed|\\\\\[boxed)\{')
    matches = list(boxed_pattern.finditer(latex_text))
    if not matches:
        return latex_text 
    processed = latex_text
    
    for match in reversed(matches):
        start_idx = match.end()  
        start_tag = match.group(1)  
        end_tag = '}' if start_tag == r'\boxed' else r'}\\\\]'
        remaining_text = processed[start_idx:]
        
        open_count = 1  
        close_count = 0
        end_pos = -1
        
        for i, char in enumerate(remaining_text):
            if char == '{':
                open_count += 1
            elif char == '}':
                close_count += 1
                if open_count == close_count:
                    
                    if start_tag == r'\\\\\[boxed':
                        
                        if i + len(r'}\\\\]') <= len(remaining_text):
                            if remaining_text[i:i+len(r'}\\\\]')] == end_tag:
                                end_pos = start_idx + i + len(r'}\\\\]')
                                break
                    else:
                        end_pos = start_idx + i + 1 
                        break
        
        if end_pos == -1:
            
            need_close = open_count - close_count
            if need_close > 0:
                
                if start_tag == r'\\\\\[boxed':
                    
                    end_marker_pos = remaining_text.find(r'\\\\]')
                    if end_marker_pos != -1:
                        
                        insert_pos = start_idx + end_marker_pos
                        append_item = '}' * need_close
                        processed = processed[:insert_pos] + append_item + processed[insert_pos:]
                    else:
                        
                        insert_pos = start_idx + len(remaining_text)
                        append_item = '}' * need_close
                        processed = processed[:insert_pos] + append_item + processed[insert_pos:]
                else:
                    
                    line_break_pos = remaining_text.find('\n')
                    if line_break_pos != -1:
                        insert_pos = start_idx + line_break_pos
                        append_item = '}' * need_close
                        processed = processed[:insert_pos] + append_item + processed[insert_pos:]
                    else:
                        insert_pos = start_idx + len(remaining_text)
                        append_item = '}' * need_close
                        processed = processed[:insert_pos] + append_item + processed[insert_pos:]
                # print(f"预处理：在 {start_tag} 结构内补充了 {need_close} "+" 个 '}'")
    
    return processed


def extract_all_boxed_content_new(latex_response: str,latex_wrap=r'\\boxed{([^{}]*|{.*?})}') -> List[str]:
    """
    提取LaTeX文本中所有\boxed{}和\\\\[boxed{...}\\\\]内的内容，
    支持嵌套{}，使用优化的预处理确保边界正确。
    """
    processed_text = preprocess_latex(latex_response)
    
    pattern = re.compile(
        r'(?:\\boxed|\\\\\[boxed)\{(?:[^{}]++|{(?:[^{}]++|{(?:[^{}]++|{})*})*})\}(?:\\\\\])?',
        re.DOTALL
    )
    matches = pattern.findall(processed_text)
    
    extracted = []
    for match in matches:
        if match.startswith(r'\\\\[boxed{'):
            content = match[len(r'\\\\[boxed{'):]
            if content.endswith(r'\\\\]'):
                content = content[:-len(r'\\\\]')]
        else:
            content = match[len(r'\boxed{'):-1]  
        
        content = content.strip()
        if content:
            extracted.append(content)
    
    return extracted
                    
def extract_all_boxed_content(latex_response, latex_wrap=r'\\boxed{([^{}]*|{.*?})}'):
    """
    Extract all \boxed{} content from a LaTeX response, supporting nested {}.

    Args:
        latex_response (str): The LaTeX response text.
        latex_wrap (str): Regular expression pattern for matching \boxed{}.

    Returns:
        list: Extracted \boxed{} content.
    """
    # Define regex pattern to match nested \boxed{}
    latex_response = preprocess_latex(latex_response)
    pattern = re.compile(r'\\boxed{((?:[^{}]|{(?:[^{}]|{.*?})*})*)}|\\\\\[boxed{((?:[^{}]|{(?:[^{}]|{.*?})*})*)}\\\\\]', re.DOTALL)
    matches = pattern.findall(latex_response)  # Match all occurrences

    if not matches:
        return []
    # Flatten matches and remove empty strings
    return [match.strip() for sublist in matches for match in sublist if match.strip()]

def extract_final_answer(last_answer):
    """
    Extract the final answer from \boxed{}.

    Args:
        last_answer (str): LaTeX text containing \boxed{}.

    Returns:
        str: Extracted answer.
    """
    match = re.search(r'\\boxed{(.*?)}|\\\\\[boxed{(.*?)}\\\\\]', last_answer)
    if match:
        return next(group for group in match.groups() if group).strip()
    return last_answer

def extract_final_answer_list(last_answer):
    """
    Extract a list of answers from \boxed{} (for multi-part answers).

    Args:
        last_answer (str): LaTeX text containing \boxed{}.

    Returns:
        list: Extracted list of answers.
    """
    matches = re.findall(r'\\boxed{\\\[(.*?)\\\]}|\\\\\[boxed{\\\[(.*?)\\\]}\\\\\]', last_answer)
    if matches:
        return [item.strip() for sublist in matches for item in sublist if item for item in item.split(',')]
    return [extract_final_answer(last_answer)]

def extract_final_answer_allform(latex_response, answer_type=None, latex_wrap=r'\\boxed{(.*?)}'):
    """
    General method to extract all final answers.

    Args:
        latex_response (str): LaTeX response text.
        answer_type (str): Type of answer (float, list, math_expression).
        latex_wrap (str): Regular expression pattern for matching LaTeX content.

    Returns:
        list: Extracted answers.
    """
    boxed_content = extract_all_boxed_content(latex_response, latex_wrap)
    if not boxed_content:
        return []

    if answer_type == 'list':
        return [extract_final_answer_list(item) for item in boxed_content]
    return [extract_final_answer(item) for item in boxed_content]
