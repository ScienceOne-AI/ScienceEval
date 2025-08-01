import time
from typing import Any
from openai import OpenAI
import re
import traceback

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError


class ReasonCompletionSampler(SamplerBase):
    """
    Initialize the sampler with OpenAI API parameters.

    Args:
        args: An object containing OpenAI API credentials and generation parameters.
    """

    def __init__(
        self,
        args
    ):
        self.args = args
        self.system_message = None
        
        self.client = OpenAI(
            api_key= args.api_key,
            base_url= args.api_url,
            timeout= args.timeout,
            max_retries = 2,
        )
        

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {"url": f"data:image/{format};{encoding},{image}",},
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}
    
    @staticmethod
    def _split_reasoning(text: str) -> tuple[str, str]:
        m = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL)
        if m:
            return m.group(1), m.group(2)
        return None, text
    
    def __call__(self, message_list: MessageList) -> dict[str, Any]:
        """
        Generates a response using the OpenAI API.
        Returns a dictionary with reasoning, answer, and token usage stats.
        """
        # Prepend system message if it is set
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list

        trial = 0
        while trial < 3:
            try:
                start = time.time()
                params = {
                    "model": self.args.model,
                    "messages": message_list,
                    "max_tokens": self.args.max_tokens,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "presence_penalty": self.args.presence_penalty,
                }
                # Remove parameters with None values
                params = {k: v for k, v in params.items() if v is not None}                                
                resp = self.client.chat.completions.create(**params)
                elapsed = round(time.time() - start, 3)

                resp_dict = resp.model_dump()
                choice_dict = resp_dict.get("choices", [{}])[0]
                
                answer_content = choice_dict.get("message", {}).get("content")
                reasoning_content = choice_dict.get("message", {}).get("reasoning_content")
                
                # Fallback: try to split reasoning and answer manually
                if answer_content and not reasoning_content:
                    reasoning_content, answer_content = self._split_reasoning(answer_content)
                
                finish_reason  = choice_dict.get("finish_reason")
                completion_tokens  = resp_dict.get("usage", {}).get("completion_tokens")
                
                print(f"question: {message_list[-1]['content']}\n")
                print(f"reasoning_content: {reasoning_content}\n")
                print(f"answer_content: {answer_content}\n\n")
                if finish_reason != "stop":
                    print("⚠️ OpenAI API warning: output may be truncated due to token limit.")
                
                result = {
                    "reasoning_content": reasoning_content,
                    "answer_content": answer_content,
                    "finish_reason": finish_reason,
                    "completion_tokens": completion_tokens,
                    "elapsed_sec": elapsed
                }
                return result

            except Exception as e:
                traceback.print_exc()
                exception_backoff = 2 ** trial
                print(f"⚠️ Exception, retry #{trial} in {exception_backoff}s: {e}")
                time.sleep(exception_backoff)
                trial += 1
            
        print("❌ All retries failed.")
        return {}