"""
Enrich precautions JSONs using Google Generative API (Gemini-like).

Usage (one-time run with pasted key):
In PowerShell, run:
$env:GEMINI_API_KEY="<YOUR_KEY>"; python enrich_precautions_gemini.py

The script will:
- Read `models/class_names.json` to get classes
- For each class, call the Generative API with a safe prompt requesting commonly used pesticide/fungicide/fertilizer product names and active ingredients for that crop+disease
- Backup existing files in `precautions_extra/backup/`
- Update/augment `precautions_extra/paddy_placeholders.json` and other files if present

Security: Do NOT write the API key to disk. The script reads it from env var `GEMINI_API_KEY`.
"""
import os
import json
import time
import requests
from pathlib import Path

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    print('Error: GEMINI_API_KEY environment variable not set. Exiting.')
    exit(1)

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES_PATH = BASE_DIR / 'models' / 'class_names.json'
EXTRA_DIR = BASE_DIR / 'precautions_extra'
BACKUP_DIR = EXTRA_DIR / 'backup'
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Google Generative API endpoint (text-bison) - using key in query param
# Candidate endpoints and model paths to try (order: preferred -> fallback)
ENDPOINT_CANDIDATES = [
    'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate',
    'https://generativelanguage.googleapis.com/v1/models/text-bison-001:generate',
    'https://generativelanguage.googleapis.com/v1beta2/models/text-bison@001:generate',
    'https://generativelanguage.googleapis.com/v1/models/text-bison@001:generate'
]


with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    classes = json.load(f)

# Load existing extras
extras = {}
for fname in os.listdir(EXTRA_DIR):
    if not fname.lower().endswith('.json'):
        continue
    path = EXTRA_DIR / fname
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                extras.update(data)
    except Exception as e:
        print('Warning: failed to load', path, e)

# For each class, if we already have an entry with an 'api_recommendation' field, skip
headers = {'Content-Type': 'application/json'}

def try_request(endpoint, headers, body):
    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
        return resp
    except Exception as e:
        return e


for cls in classes:
    print('\nProcessing class:', cls)
    existing = extras.get(cls, {})
    if existing.get('api_recommendation'):
        print('  - already enriched; skipping')
        continue

    # Craft prompt
    crop, _, disease = cls.partition('___')
    if not disease:
        disease = cls
    prompt_text = (
        f"You are an expert agronomist. Provide a concise list of up to 6 commonly used commercial product names (trade names) and their active ingredients, "
        f"typical application rates or formats (e.g., g/ha or ml/L), and a one-line safety note for treating {disease} on {crop} (crop: {crop}). "
        "If multiple product classes exist (fungicide/insecticide/bactericide/fertilizer), label them. Prioritize products commonly used in South Asia/India but include internationally-known options. "
        "Return the answer in plain text, with short bullet points. Do not include long regulatory statements."
    )

    body = {
        'prompt': {'text': prompt_text},
        'temperature': 0.2,
        'max_output_tokens': 512
    }

    # Try each endpoint candidate with two auth styles: key in query, and Bearer header
    success = False
    last_err = None
    for ep in ENDPOINT_CANDIDATES:
        # First try with key in query param
        url = ep + f'?key={API_KEY}'
        resp = try_request(url, headers, body)
        if isinstance(resp, Exception):
            last_err = resp
        else:
            if resp.status_code == 200:
                j = resp.json()
                success = True
            else:
                last_err = f'{resp.status_code}: {resp.text[:200]}'

        if success:
            pass
        else:
            # Try Bearer authorization style
            bearer_headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'}
            resp2 = try_request(ep, bearer_headers, body)
            if isinstance(resp2, Exception):
                last_err = resp2
            else:
                if resp2.status_code == 200:
                    j = resp2.json()
                    success = True
                else:
                    last_err = f'{resp2.status_code}: {resp2.text[:200]}'

        if success:
            # Parse the returned JSON to get text output in different possible shapes
            txt = None
            if isinstance(j, dict):
                if 'candidates' in j and isinstance(j['candidates'], list) and j['candidates']:
                    # older style
                    txt = j['candidates'][0].get('content', '')
                elif 'output' in j and isinstance(j['output'], list):
                    texts = []
                    for out in j['output']:
                        if out.get('type') == 'text' and out.get('text'):
                            texts.append(out.get('text'))
                    txt = '\n'.join(texts)
                elif 'responses' in j and isinstance(j['responses'], list):
                    # another variant
                    txt = '\n'.join([r.get('content', '') for r in j['responses'] if isinstance(r, dict)])
                else:
                    # fallback to stringifying
                    txt = json.dumps(j)

            extras.setdefault(cls, {})['api_recommendation'] = txt if txt else f'API returned ok but no text found. Raw: {str(j)[:400]}'
            print('  - enriched (len', len(extras[cls]['api_recommendation']), 'chars)')
            break

    if not success:
        print('  API error or no endpoint worked; last_err=', last_err)
        extras.setdefault(cls, {})['api_recommendation'] = f'API error or endpoint failure: {last_err}'

    # Be polite with rate limits
    time.sleep(1.0)

# Backup original files and write single merged JSON file
backup_path = BACKUP_DIR / f'precautions_extra_backup_{int(time.time())}.json'
with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(extras, f, indent=2, ensure_ascii=False)
print('\nBackup saved to', backup_path)

# Write combined JSON
out_path = EXTRA_DIR / 'precautions_enriched_all.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(extras, f, indent=2, ensure_ascii=False)

print('Enrichment complete. Wrote', out_path)
