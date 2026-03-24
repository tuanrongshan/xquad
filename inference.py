import ollama
from openai import OpenAI

class Inferencer:
    def __init__(self, framework='vllm', model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", base_url="http://localhost:4630/v1"):
        self._framework = framework
        self.model = model
        self._base_url = base_url

        if self._framework == 'ollama':
            self.client = ollama.Client(host=base_url)
        elif self._framework == 'vllm':
            self.client = OpenAI(base_url=base_url, api_key="")
        else:
            raise ValueError("Unsupported framework.")

    def model_inference(self, prompt: str, temperature: float = 0.8) -> str:
        if self._framework == 'vllm':
            return self._model_inference_vllm(prompt, temperature)
        elif self._framework == 'ollama':
            return self._model_inference_ollama(prompt, temperature)

    def _model_inference_ollama(self, prompt, temperature=0.8) -> str:
        response = self.client.chat(
            model="deepseek-r1:32b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        response = response['message']['content']
        if '</think>' in response:
            response = response.split('</think>', 1)[-1]
        return response

    def _model_inference_vllm(self, prompt: str, temperature: float = 0.8) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        response = response.choices[0].message.content
        if '</think>' in response:
            response = response.split('</think>', 1)[-1]
        return response