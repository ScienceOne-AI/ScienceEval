import os
import json
import logging
import requests
import inspect
from openai import OpenAI
from retrying import retry
from datetime import datetime


class LLMJudgeChatBot:
    
    """
    参数:
    - model (str): 调用的模型名。
    - log_path(str): 日志文件地址
    - api_config : 字典，模型的传参。
    """

    def __init__(self,api_config,log_path="llm_judge.jsonl"):
        self.log_path = log_path
        self.url = api_config["api_url"]
        self.api_key = api_config["api_key"]
        self.model = api_config["model"]
        self.timeout = api_config.get("timeout",3600)
        self.client = OpenAI(
            base_url = self.url,
            api_key = self.api_key,
            default_query={"api-version":"preview"}
        )


    def get_retry_attempt(self):
        frame = inspect.currentframe()
        while frame:
            if 'attempt_number' in frame.f_locals:
                return frame.f_locals['attempt_number']
            frame = frame.f_back
        return 1


    @retry(stop_max_attempt_number=3, wait_fixed=10000)
    def generate_response(self, messages):
        """
        根据输入的消息生成GPT4的响应。
        
        返回:
        - str: 生成的响应内容。
        """
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                timeout = self.timeout
            )
            dct_input = {
                "model":self.model,
                "timeout":self.timeout, 
                "messages":messages
            }
            
            dct_output = response.model_dump() 

            # 存储调用日志      
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {
                'timestamp': current_time,
                'input': dct_input,
                'output': dct_output,
            }
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')


            # 处理模型输出结果
            choice = dct_output["choices"][0]
            content = choice["message"].get("content", None)
            response_content = choice["message"].get("content", None)
            reasoning_content = choice["message"].get("reasoning_content", None)
            finish_reason = choice.get("finish_reason", None)
            completion_tokens = dct_output["usage"].get("completion_tokens", None)
            usage = {
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason
                }
            # 输出失败
            if content == None and reasoning_content == None:
                return content,reasoning_content,response_content,usage
            # 不输出思考内容的模型答案都在content
            elif content != None and reasoning_content == None:
                content = content.strip()
                reasoning_content = content
                response_content = content
            # 输出思考内容的模型，但输出超长，答案在reasoning
            elif content == None and reasoning_content != None :
                reasoning_content = reasoning_content.strip()
                content = reasoning_content
                response_content = reasoning_content
            # 输出思考内容的模型，正常输出，答案分别存在content和reasoning
            elif content != None and reasoning_content != None:                
                content = content.strip()
                reasoning_content = reasoning_content.strip()
                response_content = f"<|thought_start|>\n{reasoning_content}\n<|thought_end|>\n{content}"
           
            # 如果思考内容和回答拼在一起
            if "</think>" in response_content:
                reasoning_content, content = response_content.split("</think>")
                reasoning_content = reasoning_content.replace("<think>","").strip()
                content = content.strip()
            return content,reasoning_content,response_content,usage
        except Exception as e:
            # 在控制台返回报错
            attempt = self.get_retry_attempt()
            logging.error(f"LLMJudge Warning: {e}, Modle: {self.model}, Message :{messages}, Retrying: {attempt}/3 ")
            raise

def get_llmjudge_result(messages,api_config,log_path="llm_judge.jsonl"):
    try:
        bot = LLMJudgeChatBot(api_config,log_path)
        content,reasoning_content,response_content,usage = bot.generate_response(messages)   
        return content,reasoning_content,response_content,usage
    except Exception as e:
        logging.error(f"LLMJudge Warning: {e}, Modle: {api_config["model"]}, Message :{messages}, Failed to retry 3 times")
        return "Error","Error","Error","Error"



if __name__ == '__main__':
    question = "What are the advantages of using melamine sponge as a precursor for N-doped carbon foam and its applications in electrochemistry?"
    gold_answer = "Melamine sponge (MS) serves as an excellent precursor for nitrogen-doped carbon foam (N-doped carbon foam) due to its unique combination of properties and advantages. Its high nitrogen content enables efficient and straightforward carbonization to produce N-doped carbon foam, which is particularly beneficial for enhancing electrochemical performance. The resulting material exhibits hydrophobicity, porosity, and conductivity, making it highly versatile for various applications in electrochemistry.\n\nKey advantages of using melamine sponge include its cost-effectiveness, environmental friendliness, and ease of synthesis compared to alternatives like polyurethane or graphene foam. The inherent porosity of MS-derived carbon foam allows for further modifications, such as heteroatom doping or coating with electrochemically active materials, which can significantly improve its performance. These characteristics expand its applicability in energy storage devices like supercapacitors and lithium-ion batteries, as well as in oxygen reduction reaction (ORR) catalysts, strain/stress sensors, and oil/water separation.\n\nIn summary, melamine sponge is a practical and efficient precursor for N-doped carbon foam, offering broad utility in energy storage, sensing, and environmental applications due to its favorable electrochemical properties, ease of modification, and sustainable production process."
    raw_answer = "Melamine “magic” sponge is an almost ideal starting material for turning a cheap, every-day article into a high-value electrochemical component.  \nThe main reasons and the most frequently reported electrochemical uses are summarized below.\n\n1. Why the sponge is a good precursor\n\n   • Intrinsic N source  \n     – Melamine is ≈ 66 wt % nitrogen (triazine –NH– and –C=N– groups).  \n     – After pyrolysis the nitrogen is automatically incorporated into the graphitic lattice (pyridinic, pyrrolic, graphitic N), so no extra dopant or post-treatment is necessary.\n\n   • Ready-made 3-D architecture  \n     – Commercial melamine sponge is an open-cell foam with micron-sized struts that carbonise almost without shrinkage.  \n     – The resulting material is a hierarchical macro/meso/micro-porous carbon foam that allows fast ion diffusion and electron transport.\n\n   • Self-standing, light and elastic  \n     – Density can fall below 10 mg cm-3 after carbonisation.  \n     – The carbon foam can be cut, bent or pressed directly into a device; no metal current collector, binder or conductive additive is required.\n\n   • Extremely cheap and scalable  \n     – The sponges cost only a few cents, are produced on the tonne scale and can be converted in a single furnace step (N₂/Ar, 700-1000 °C).\n\n   • Easy to functionalise  \n     – Soaking the sponge before pyrolysis lets one introduce S, P, B or metal salts so that multi-doped or metal-decorated foams are obtained in one pot.\n\n2. Electrochemical applications that exploit these advantages\n\n   1. Super-/pseudocapacitors  \n      • High surface area and N-induced pseudocapacitance give specific capacitances of 200–500 F g-1.  \n      • The monolithic, binder-free electrode shows negligible IR drop and 10000-cycle stability.\n\n   2. Metal-ion batteries (Li-ion, Na-ion, K-ion)  \n      • As anode: defect/N-rich sites store extra charge; the open network shortens ion paths and tolerates volume change.  \n      • As conductive skeleton for composite cathodes or for high-sulphur-loading Li-S batteries.\n\n   3. Metal–air / Zn–air batteries and fuel-cell cathodes  \n      • N-doped carbon foam is an efficient, metal-free ORR electrocatalyst (onset ≈0.89 V vs RHE, 4-e⁻ pathway).  \n      • The flexible sheet can serve directly as the breathable air cathode.\n\n   4. Water splitting / HER & OER  \n      • When additionally decorated with Ni, Co, Fe phosphides or sulphides, the 3-D N-doped carbon delivers low overpotentials and long-term mechanical integrity.\n\n   5. Capacitive de-ionisation and electro-adsorption  \n      • High surface area and hydrophilic N sites give large charge efficiency and fast salt removal.\n\n   6. Electrochemical sensors  \n      • Abundant edge-N sites and high conductivity provide low-potential detection of H₂O₂, dopamine, nitroaromatics, etc.\n\nIn short, melamine sponge offers a nitrogen-rich chemistry, an instant macro-porous template, mechanical stability and negligible cost, making it one of the most attractive precursors for producing N-doped carbon foams for a broad range of electrochemical technologies."
   
    system_prompt = """Your task is to evaluate the accuracy of LLM-generated answers to materials science questions by comparing them to expert-validated "gold" answers.\n\nFor each evaluation, you will receive:\n\t- A materials science question\n\t- A gold answer, based on authoritative domain knowledge\n\t- An LLM-generated inference answer, which you must assess\n\nYour goal is to evaluate how well the inference answer aligns with the gold answer in terms of factual accuracy, conceptual completeness, and relevance.\n\nUse the following evaluation rubric:\n\t- Correct: The inference answer fully captures all essential concepts from the gold answer, with no significant omissions or factual errors.\n\t- Mostly Correct: The inference answer conveys the main idea or correct conclusion, even if minor details are missing or slight inaccuracies are present. Additional non-conflicting information is acceptable.\n\t- Incorrect: The inference answer demonstrates substantial misunderstanding, includes major factual errors, or omits core concepts present in the gold answer.\n\nProvide a short justification for your rating, highlighting key similarities or discrepancies between the inference and gold answers. Output your response in the following JSON format:\n{{\n    "reasoning": "A concise explanation supporting your judgment.",\n    "judgment": "correct|mostly correct|incorrect"\n}"""
    user_prompt = f"""**Input Data**:
        - Material Science Question: {question}
        - Gold Answer: {gold_answer}
        - Student Answer: {raw_answer}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    api_config= {
        "api_url":"your_api_url",
        "model":"your_model_name",
        "api_key":"your_model_key",
        }
    content,reasoning_content,response_content,usage = get_llmjudge_result(messages,api_config)
    print(content)


