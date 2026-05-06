"""
LLM Client Module - Encapsulate LLM API Calls
"""
import os
import dashscope
from dashscope import Generation
from typing import Dict, Any, List


class LLMClient:
    """LLM Client"""

    def __init__(self, config):
        self.model = config.llm.model
        self.api_key = config.llm.api_key
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens

        # Set API key
        dashscope.api_key = self.api_key

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response

        Args:
            messages: Message list
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        import time
        start_time = time.time()

        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                result_format="message",
                temperature=kwargs.get('temperature', self.temperature),
                top_p=0.8,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                timeout=300  # 5 minute timeout
            )

            elapsed = time.time() - start_time
            if elapsed > 60:
                print(f"[WARNING] LLM call took {elapsed:.1f}s, may be slow")

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise Exception(f"LLM API error: {response.code}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[ERROR] LLM call failed (took {elapsed:.1f}s): {str(e)}")
            raise Exception(f"LLM generation failed (after {elapsed:.1f}s): {str(e)}")

    def generate_with_function(self, messages: List[Dict[str, str]],
                                functions: List[Dict],
                                **kwargs) -> Dict[str, Any]:
        """
        Generate response with function calling

        Args:
            messages: Message list
            functions: Function definition list
            **kwargs: Additional parameters

        Returns:
            Dictionary containing response and function calls
        """
        # TODO: Implement function calling support
        content = self.generate(messages, **kwargs)
        return {
            'content': content,
            'function_call': None
        }
