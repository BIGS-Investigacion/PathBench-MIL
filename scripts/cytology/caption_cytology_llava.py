"""
Genera captions con LLaVA-Med para imágenes de citología cervical.
Guarda resultados en results/cytology/captions_llava.csv
"""
from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
OUT_DIR  = ROOT / 'results/cytology'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES   = ['Carcinoma', 'Negativas']
MODEL_ID  = 'llava-hf/llava-1.5-7b-hf'
PROMPT    = "USER: <image>\nDescribe the morphological features visible in this cervical cytology image. Focus on cell characteristics, nuclear features, and any abnormalities. ASSISTANT:"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Cargando {MODEL_ID}...')
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map='auto',
)
model.eval()
print('Modelo listo.\n')

records = []
for cls in CLASSES:
    cls_dir = DATA_DIR / cls
    images  = sorted(cls_dir.glob('*.png')) + sorted(cls_dir.glob('*.jpg'))
    print(f'[{cls}] {len(images)} imágenes')
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        inputs = processor(text=PROMPT, images=img, return_tensors='pt').to(device, torch.float16)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )
        caption = processor.decode(output[0], skip_special_tokens=True)
        # Extraer solo la respuesta del ASSISTANT
        caption = caption.split('ASSISTANT:')[-1].strip()
        print(f'  {img_path.name}:\n    {caption}\n')
        records.append({
            'class':    cls,
            'filename': img_path.name,
            'caption':  caption,
        })

df = pd.DataFrame(records)
out_path = OUT_DIR / 'captions_llava.csv'
df.to_csv(out_path, index=False)
print(f'Guardado: {out_path}')