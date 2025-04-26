from typing import List
from starlette.requests import Request

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from ray import serve
import asyncio
import time

# pip install 'ray[serve]' transformers torch

@serve.deployment(num_replicas=4, max_ongoing_requests=16)
class BatchingGPT2Deployment:
    def __init__(self):
        model_id = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()


        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.max_length = 50

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.05)
    async def handle_batch(self, requests: List[Request]) -> List[dict]:
        texts = [r.query_params["text"] for r in requests]
        num_return_sequences = int(requests[0].query_params.get("num_return_sequences", 10))

        if not all(
            r.query_params.get("num_return_sequences", "10") == str(num_return_sequences)
            for r in requests
        ):
            return [{"error": "All requests in a batch must have the same num_return_sequences"}] * len(requests)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            start = time.time()
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            print(f"BATCH SIZE: {len(texts)} | SEQ: {num_return_sequences} | TOOK: {time.time() - start:.2f}s")

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        grouped_outputs = [
            decoded[i * num_return_sequences:(i + 1) * num_return_sequences]
            for i in range(len(texts))
        ]

        return [{"generated_texts": group} for group in grouped_outputs]

    async def __call__(self, request: Request) -> dict:
        return await self.handle_batch(request)

serve.run(BatchingGPT2Deployment.bind(), route_prefix="/")

# Sample request
import requests
response = requests.get(
    "http://localhost:8000/",
    params={"text": "Ray Serve is great because", "num_return_sequences": 3}
)
print(response.json())

import time
time.sleep(1000000)
