import time
import os
import json
import sys
from predict import predict_disease

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(PROJECT_DIR, 'input.jpg')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'diagnostics')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'last_prediction.json')
TIMEOUT_SECONDS = 300
POLL_INTERVAL = 1.0

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Watching for image at: {INPUT_PATH}")
start = time.time()
while True:
    if os.path.exists(INPUT_PATH):
        print(f"Found image: {INPUT_PATH}")
        try:
            result = predict_disease(INPUT_PATH)
            # augment result with local path
            result['local_path'] = INPUT_PATH
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(json.dumps(result, indent=2))
            print(f"Saved prediction to: {OUTPUT_PATH}")
            sys.exit(0)
        except Exception as e:
            print(f"Prediction failed: {e}", file=sys.stderr)
            sys.exit(2)

    if time.time() - start > TIMEOUT_SECONDS:
        print(f"Timeout waiting for {INPUT_PATH} (waited {TIMEOUT_SECONDS}s).", file=sys.stderr)
        sys.exit(3)

    time.sleep(POLL_INTERVAL)
