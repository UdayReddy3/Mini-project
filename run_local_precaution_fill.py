import json
from precautions import DiseasePrecautionGenerator
from pathlib import Path

PREC_DIR = Path('precautions_extra')
FILES = [PREC_DIR / 'precautions_enriched_manual.json', PREC_DIR / 'precautions_enriched_all.json']

def fill_files():
    gen = DiseasePrecautionGenerator()
    total_updated = 0
    for f in FILES:
        if not f.exists():
            print(f"File not found: {f}")
            continue
        data = json.loads(f.read_text(encoding='utf-8'))
        updated = 0
        for key, entry in data.items():
            has_chemical = entry.get('chemical_treatment') and len(entry.get('chemical_treatment', [])) > 0
            has_natural = entry.get('natural_treatment') and len(entry.get('natural_treatment', [])) > 0
            if not has_chemical or not has_natural:
                smart = gen.get_smart_precautions(key)
                # Update missing fields only
                if smart.get('chemical_treatment') and (not has_chemical):
                    entry['chemical_treatment'] = smart['chemical_treatment']
                if smart.get('natural_treatment') and (not has_natural):
                    entry['natural_treatment'] = smart['natural_treatment']
                # Fill other metadata if missing
                if not entry.get('time_to_recovery') or entry.get('time_to_recovery') in ('N/A',''):
                    entry['time_to_recovery'] = smart.get('time_to_recovery', entry.get('time_to_recovery', 'N/A'))
                if not entry.get('yield_impact') or entry.get('yield_impact') in ('Unknown',''):
                    entry['yield_impact'] = smart.get('yield_impact', entry.get('yield_impact', 'Unknown'))
                if not entry.get('cost_effectiveness') or entry.get('cost_effectiveness') in ('N/A',''):
                    entry['cost_effectiveness'] = smart.get('cost_effectiveness', entry.get('cost_effectiveness', 'N/A'))
                updated += 1
        if updated:
            f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"File {f.name}: updated {updated} entries")
        total_updated += updated
    print(f"Total updated: {total_updated}")

if __name__ == '__main__':
    fill_files()
