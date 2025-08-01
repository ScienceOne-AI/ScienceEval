from math_verify import parse, verify
from sympy import *
import math
from post_process import parse_math_answer, remove_not, cal_not,parse_not



def equiv(
    model_output,   # 模型输出的字符串
    answer,    # 标准答案的字符串
    rel_tol    # 相对误差容忍度（默认0.05, 即5%误差范围）
):
    print('\n====>:equiv判别')
    
    model_output=model_output.replace(',', '')
    try:
        ans=float(answer.strip())
        print('equiv   ans:',ans)
        model_val = float(model_output.strip())
        print('equiv   model_output:', model_val)
        first = math.isclose(model_val, ans, rel_tol=rel_tol)
    except Exception as e:
        print('equiv first 对比出错:', e)
        first=False
    try: 
        model=model_output.strip().split()[0]
        model_val = float(model.strip())
        second=math.isclose(model_val, ans, rel_tol=rel_tol)
    except Exception as e:
        print('equiv second 对比出错:', e)
        second=False
        
    print('equiv   first judge:', first)
    print('equiv   second judge:', second)
    return first or second


def good_verify(
        gold,                   # 标准答案
        answer,                 # 待评估答案
        float_rounding=6,       # 浮点数精度（保留几位小数）
        fuzzy_comparison=0.05   # 模糊比较（相较标准答案可以上下浮动的范围）
    ):
    gold=str(gold.strip())
    answer=str(answer.strip())
    is_equiv = equiv(model_output=answer, answer=gold,rel_tol=fuzzy_comparison)
    if 'e' in answer.lower() and 'e' in gold.lower():
        # 如果答案中包含科学计数法的'e'，则直接返回is_equiv
        print('答案中包含科学计数法的\'e\'，直接返回is_equiv')
        return is_equiv
    # 检测gold和answer是否以$开头和结尾，若没有就补上
    print('\n====>:good_verify判别')
    if not gold.startswith('$'):
        gold = '$' + gold
    if not gold.endswith('$'):
        gold = gold + '$'
    if not answer.endswith('$'):
        answer = answer + '$'
    if not answer.startswith('$'):
        answer = '$' + answer
   
    
    # 解析答案
    gold = parse(gold)  
    answer = parse(answer)
    print('gold:', gold)
    print('answer:', answer)
    
    if gold[1] == answer[1]:
        print('gold[1] == answer[1]')
        return True
    if gold[0] == answer[0]:
        print('gold[0] == answer[0]')
        return True

    gold_val = N(gold[0])
    boundary1 = (1 - fuzzy_comparison) * gold_val
    boundary2 = (1 + fuzzy_comparison) * gold_val

    # 确保left_nums < right_nums，无论gold_val是正数还是负数
    left_nums = min(boundary1, boundary2)
    right_nums = max(boundary1, boundary2)

    print('左边界:', left_nums, '    右边界:', right_nums)

    try:
        answer_val = N(answer[0])
        print('good_verify  answer_val:', answer_val)
        if left_nums <= answer_val <= right_nums:
            print('good_verify模糊比较通过')
            return True
    except Exception as e:
        print(f'数值比较失败: {e}')
        print(f'gold[0]: {gold[0]}, type: {type(gold[0])}')
        print(f'answer[0]: {answer[0]}, type: {type(answer[0])}')
        # 如果无法转换为数值，跳过数值比较
        pass
   
    # return verify(gold, answer, float_rounding=float_rounding) 
    _good_verify = verify(gold, answer, float_rounding=float_rounding)
    print(f'最终通过两个函数判断_good_verify： {_good_verify}， is_equiv: {is_equiv}')
    
    return _good_verify or is_equiv



def extract_boxed_content(
    text,  # 输入的文本字符串，包含可能的 \boxed{} 
):
    """
    匹配最后一个 \\boxed{} 中的内容，支持自动检测嵌套{}的情况。
    """
    import re
    
    # 找到所有 \boxed{ 的位置
    boxed_pattern = r'\\boxed\{'
    matches = list(re.finditer(boxed_pattern, text))
    print(f"找到 {len(matches)} 个 `\\boxed{{` 的位置，位置列表：{[m.start() for m in matches]}")
    if not matches:
        print("没有找到任何 \\boxed{ 字符串")
        return None
    
    # 获取最后一个 \boxed{ 的位置
    last_match = matches[-1]
    print(f"最后一个 `\\boxed{{` 的位置：{last_match.start()}")
    start_pos = last_match.end()  # \boxed{ 之后的位置
    print(f"开始匹配括号的位置：{start_pos}")
    
    # 从最后一个 \boxed{ 开始匹配括号
    brace_count = 1
    i = start_pos
    
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        # 找到了匹配的闭合括号
        content = text[start_pos:i-1]
        return content
    else:
        # 没有找到匹配的闭合括号
        print("最后一个 \\boxed{ 没有找到匹配的闭合括号")
        return None


def good_verify_with_unit(
    gold_with_unit,         # 标准答案，带有单位。且数值和单位之间用@@分隔
    answer_gpt_solution,    # 原始答案（gpt solution），有可能不仅包括所有回答，还有可能包括所有思考过程
    float_rounding=6,       # 浮点数精度（保留几位小数）
    fuzzy_comparison=0.05,   # 模糊比较（相较标准答案可以上下浮动的范围）
    in_boxed=True           # 是否在 \boxed{} 中提取答案，默认为True
):
    '''
    从模型返回值（gpt solution）中重新提取模型最终回答（gpt answer）
    然后将最终回答（gpt answer）先和不带单位的参考答案（gold）进行比较，如果相同则返回True，否则和带单位的参考答案（gold）比较，若相同，则返回True，否则返回False。
    '''
    
    # 提取最终回答（gpt answer）
    if in_boxed:
        answer = extract_boxed_content(answer_gpt_solution)  # 提取最后一个 \boxed{} 中的内容
    else:
        answer = answer_gpt_solution.strip()
    
    print(f'\n\n==================================================================================')
    
    
    if '@@' in gold_with_unit:
        gold = gold_with_unit.split('@@')[0].strip()   # 不带单位的参考答案
        unit_part = gold_with_unit.split('@@')[-1].strip().strip('$')
        gold_with_unit = gold + ' ' + unit_part  # 带正确单位的参考答案 没有@@分隔符
    else:
        # 如果没有@@分隔符，gold和 gold_with_unit相同
        gold = gold_with_unit
        gold_with_unit = gold_with_unit.strip()  # 确保去除多余空格
    print(f'===========================answer:{answer}, gold:{gold}, gold_with_unit:{gold_with_unit}===========================')
    print(f'通过cal_not和parse_not两个函数解析')
    answer=cal_not(parse_not(answer))
    # gold=cal_not((gold, unit_part))
    print(f'===========================answer:{answer}, gold:{gold}, gold_with_unit:{gold_with_unit}===========================')


    # 3.52e-19@@$10^{-19} \\mathrm{~J}$
    # 27}^{50} \\frac{\\binom{50}{k}}{2^{50}}
    
    # print(f'good_verify_with_unit  gold_with_unit: {gold_with_unit}, gold: {gold}, answer: {answer}')
    if answer is None:
        print("未能提取到有效的gpt answer")
        return False

    # 比较是否和不带单位的参考答案一致
    print('\n=========比较是否和不带单位的参考答案一致==========')
    result1 = good_verify(
        gold,                   # 标准答案
        answer,                 # 待评估答案，不带单位
        float_rounding,       # 浮点数精度（保留几位小数）
        fuzzy_comparison   # 模糊比较（相较标准答案可以上下浮动的范围）
    )
    if result1:
        print("不带单位的答案匹配成功")
        return True
    
    # 比较是否和带单位的参考答案一致
    print('\n=========比较是否和带单位的参考答案一致==========')
    result2 = good_verify(
        gold_with_unit,                   # 标准答案
        answer,                 # 待评估答案，带单位
        float_rounding,       # 浮点数精度（保留几位小数）
        fuzzy_comparison   # 模糊比较（相较标准答案可以上下浮动的范围）
    )
    if result2:
        print("带单位的答案匹配成功")
        return True
    
    print("最终答案与参考答案不匹配")
    return False

if __name__ == '__main__':
    answer_gold_list = [
        ("7.89 \\times 10^{-5}", "7.889122578500967e-05"),
    ]

    for answer, gold in answer_gold_list:
        result2 = good_verify_with_unit(gold_with_unit=gold, answer_gpt_solution=answer, in_boxed=False)
        print(f'<<<<<<<<最终判断: {result2}>>>>>>>>>>')
