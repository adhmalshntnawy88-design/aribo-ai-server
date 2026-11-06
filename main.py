
import os
import json
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "Aribo_GPT_v0.4.pth")
TOKENIZER_PATH = os.path.join(BASE, "models", "tokenizer.json")

device = torch.device("cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer = json.load(f)

vocab = checkpoint["vocab"]
char_to_id = tokenizer["char_to_id"]
id_to_char = tokenizer["id_to_char"]

class AriboGPT(nn.Module):
    def __init__(self, vocab_size, settings):
        super().__init__()
        d = settings["d_model"]
        self.block_size = settings["block_size"]

        self.token_embedding = nn.Embedding(vocab_size, d)
        self.position_embedding = nn.Embedding(self.block_size, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=settings["n_heads"],
            dim_feedforward=d * 2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=settings["n_layers"]
        )

        self.output_layer = nn.Linear(d, vocab_size)

    def forward(self, idx):
        length = idx.shape[1]
        positions = torch.arange(length, device=idx.device)

        x = self.token_embedding(idx)
        x = x + self.position_embedding(positions)

        mask = torch.triu(
            torch.ones(length, length, device=idx.device),
            diagonal=1
        ).bool()

        x = self.transformer(x, mask=mask)
        return self.output_layer(x)

model = AriboGPT(len(vocab), checkpoint["settings"]).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

app = FastAPI(title="ARIBO AI API", version="0.5.0")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {
        "name": "ARIBO AI",
        "status": "online",
        "version": "0.5.0"
    }

@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "Aribo_GPT_v0.4"
    }

@app.post("/chat")
def chat(data: ChatRequest):
    message = data.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    if any(c not in char_to_id for c in message):
        raise HTTPException(
            status_code=422,
            detail="Unsupported characters in current tokenizer"
        )

    ids = [char_to_id[c] for c in message]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    generated = []

    with torch.no_grad():
        for _ in range(80):
            context = idx[:, -model.block_size:]
            logits = model(context)[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1)

            idx = torch.cat([idx, next_id], dim=1)
            generated.append(next_id.item())

    reply = "".join(id_to_char[str(i)] for i in generated)

    return {
        "reply": reply,
        "model": "Aribo_GPT_v0.4"
    }
