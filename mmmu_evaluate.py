import base64
import io
from typing import List, Dict
import openai
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import pandas as pd

client = openai.OpenAI(
    api_key = "YOUR_API_KEY_HERE",
    base_url = "https://api.your-provider.com/v1"
)

def pil_to_base64(img: Image.Image) -> str:
    if img.mode in ("P", "RGBA", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def query_model(
    text_prompt: str,
    images: List[Image.Image],
    model_name: str,
    max_tokens: int = None,
) -> str:
    if max_tokens is None:
        if model_name in {"gpt-5", "gemini-2.5-pro"}:
            max_tokens = 4096
        else:
            max_tokens = 32
    content = [{"type": "text", "text": text_prompt}]
    for img in images:
        b64 = pil_to_base64(img)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=1 if model_name == "gpt-5" else 0
    )
    return resp.choices[0].message.content.strip()

def extract_images(item) -> List[Image.Image]:
    images = []
    for i in range(1, 8):
        img_data = item.get(f"image_{i}")
        if img_data is None:
            continue
        if isinstance(img_data, dict) and "bytes" in img_data:
            img = Image.open(io.BytesIO(img_data["bytes"]))
            images.append(img)
        elif hasattr(img_data, "save"):
            images.append(img_data)
        else:
            print(f"未知的图片格式: {type(img_data)}")
    return images

def build_prompt(question: str, question_type: str, options_raw: str) -> str:
    if question_type == "multiple-choice":
        opts = [o.strip() for o in options_raw.split("|") if o.strip()]
        letters = "ABCDEF"
        lines = [f"Question: {question}"]
        for ch, opt in zip(letters, opts):
            lines.append(f"{ch}: {opt}")
        lines.append("Answer with the single correct option letter (A/B/C/…), no explanation.")
        return "\n".join(lines)
    else:
        return (
            f"Question: {question}\n"
            f"Answer with a concise number or word, no explanation."
        )

category_datasets: Dict[str, List[Dict]] = {
    "Agriculture": load_dataset(
        "parquet", data_files="Agriculture/validation-00000-of-00001.parquet"
    )["train"],
    "Computer_Science": load_dataset(
        "parquet", data_files="Computer_Science/validation-00000-of-00001.parquet"
    )["train"],
    "Math": load_dataset(
        "parquet", data_files="Math/validation-00000-of-00001.parquet"
    )["train"],
}

models = [
    "gpt-4o",
    "qwen-vl-plus",
    "google/gemini-2.5-flash",
    "claude-sonnet-4-20250514",
    "gpt-5",
    "claude-3-7-sonnet-20250219",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "gpt-4.1",
]

results = {m: {} for m in models}

for category, ds in category_datasets.items():
    for model_name in models:
        correct = 0
        total = len(ds)
        pbar = tqdm(total=total, desc=f"{category} | {model_name}")
        for item in ds:
            prompt = build_prompt(
                question=item["question"],
                question_type=item["question_type"],
                options_raw=item["options"]
            )
            images = extract_images(item)
            pred = query_model(prompt, images, model_name)
            gt = item["answer"]
            if pred.upper().startswith(gt.upper()):
                correct += 1
            pbar.update(1)
        pbar.close()
        results[model_name][category] = (correct, total)

data = []
for model in models:
    row = {"Model": model}
    for category in category_datasets:
        correct, total = results[model][category]
        accuracy = correct / total
        row[category] = f"{correct}/{total} ({accuracy:.2%})"
        row[f"{category}_score"] = accuracy
    data.append(row)

df = pd.DataFrame(data)

score_columns = [f"{category}_score" for category in category_datasets]
df['Average'] = df[score_columns].mean(axis=1)
df = df.sort_values('Average', ascending=False)

display_columns = ["Model"] + list(category_datasets.keys()) + ["Average"]
df_display = df[display_columns].copy()

df_display['Average'] = df_display['Average'].apply(lambda x: f"{x:.2%}")

print("\nEvaluation Results by Category")
print("=" * 80)
print(df_display.to_string(index=False))

df_display.to_csv("evaluation_results.csv", index=False)

print("\nResults saved to evaluation_results.csv")
