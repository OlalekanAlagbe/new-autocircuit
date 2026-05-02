"""
Run all pending causal validation experiments.
Model: gemma-2-2b  |  SAE: gemmascope-transcoder-16k  |  strength = -20 suppress
"""
import json
import time
import requests

BASE_URL  = 'https://www.neuronpedia.org/api'
STEER_URL = f'{BASE_URL}/steer'
MODEL     = 'gemma-2-2b'
SAE       = 'gemmascope-transcoder-16k'
HEADERS   = {'Content-Type': 'application/json'}

PROMPTS = {
    'berlin':  'Paris is to France as Berlin is to',
    'rome':    'Paris is to France as Rome is to',
    'tokyo':   'Paris is to France as Tokyo is to',
    'teacher': 'Doctor is to hospital as teacher is to',
    'bird':    'Fish is to water as bird is to',
}

# ── core helper ───────────────────────────────────────────────────────────────

def fmt(f, strength):
    return {'modelId': MODEL, 'layer': f'{f["layer"]}-{SAE}',
            'index': f['index'], 'strength': strength}

def _post_with_retry(payload, retries=3, backoff=10):
    for attempt in range(retries):
        try:
            r = requests.post(STEER_URL, json=payload, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503):
                wait = backoff * (attempt + 1)
                print(f'    HTTP {r.status_code} — retrying in {wait}s')
                time.sleep(wait)
                continue
            return r
        except requests.RequestException as e:
            print(f'    Request error: {e} — retrying')
            time.sleep(backoff)
    return r


def steer_call(prompt, features, strength=-20, n_tokens=10):
    """
    Single API call — the response contains both DEFAULT and STEERED.
    Empty-features baseline calls return 500; always send at least one feature.
    """
    payload = {
        'prompt': prompt, 'modelId': MODEL,
        'features': [fmt(f, strength) for f in features],
        'temperature': 0.2, 'n_tokens': n_tokens,
        'freq_penalty': 1.0, 'seed': 42, 'strength_multiplier': 1.0,
    }
    r = _post_with_retry(payload)
    if r.status_code != 200:
        return r.status_code, '', ''
    body = r.json()
    time.sleep(0.8)
    return 200, body.get('DEFAULT', ''), body.get('STEERED', '')


def first_tok(text, prompt):
    s = text.replace('<bos>', '').strip()
    if s.lower().startswith(prompt.lower()):
        s = s[len(prompt):].strip()
    parts = s.split()
    return parts[0] if parts else '?'


def run(label, prompt_key, features, strength=-20, n_tokens=10):
    prompt = PROMPTS.get(prompt_key, prompt_key)
    print(f'  [{label}] {prompt_key}')
    status, default, steered = steer_call(prompt, features, strength, n_tokens)
    if status != 200:
        print(f'    !! HTTP {status}')
        return {'label': label, 'circuit': prompt_key, 'prompt': prompt,
                'api_status': status, 'default': '', 'steered': '',
                'default_first': '?', 'steered_first': '?', 'changed': False}
    d_tok = first_tok(default, prompt)
    s_tok = first_tok(steered, prompt)
    changed = d_tok.lower().rstrip('.,;') != s_tok.lower().rstrip('.,;')
    print(f'    default: {d_tok!r}  steered: {s_tok!r}  changed: {changed}')
    return {
        'label': label, 'circuit': prompt_key, 'prompt': prompt,
        'api_status': 200,
        'default': default.replace('<bos>', '').strip()[:70],
        'steered': steered.replace('<bos>', '').strip()[:70],
        'default_first': d_tok,
        'steered_first': s_tok,
        'changed': changed,
    }


# ── feature sets ─────────────────────────────────────────────────────────────

PHASE1 = [
    {'layer': 0,  'index': 11651},
    {'layer': 1,  'index': 11356},
    {'layer': 2,  'index': 11475},
    {'layer': 4,  'index': 10752},
    {'layer': 5,  'index': 9672},
]
PHASE2 = [
    {'layer': 5, 'index': 5793},
    {'layer': 5, 'index': 2141},
    {'layer': 8, 'index': 13766},
    {'layer': 9, 'index': 13344},
]
PHASE1_2   = PHASE1 + PHASE2
PHASE1_2_3 = PHASE1_2 + [{'layer': 13, 'index': 10969}]

SPECIFICITY = [
    ('berlin',  6,  3335,  'difficulty/challenges'),
    ('rome',    6,  2267,  'formal text/code'),
    ('rome',    4,  14857, 'code snippets'),
    ('tokyo',   6,  2267,  'formal text/code'),
    ('teacher', 4,  14857, 'code snippets'),
    ('teacher', 8,  13766, 'analogies or comparisons'),
    ('bird',    6,  2267,  'formal text/code'),
    ('bird',    5,  5793,  'analogies'),
]

# ── run ───────────────────────────────────────────────────────────────────────

results = {}

print('\n=== Phase 1 Collective (teacher + bird) ===')
for c in ('teacher', 'bird'):
    results[f'phase1_{c}'] = run('All Phase 1 (5 feat)', c, PHASE1)

print('\n=== Phase 1+2 Collective (all 5 circuits) ===')
for c in PROMPTS:
    results[f'phase1_2_{c}'] = run('Phase 1+2 (9 feat)', c, PHASE1_2)

print('\n=== Phase 1+2+3 Collective (all 5 circuits) ===')
for c in PROMPTS:
    results[f'phase1_2_3_{c}'] = run('Phase 1+2+3 (10 feat)', c, PHASE1_2_3)

print('\n=== Specificity Tests ===')
for circuit, layer, index, label in SPECIFICITY:
    key = f'spec_{circuit}_L{layer}_{index}'
    r = run(f'Specificity L{layer}/{index} ({label})',
            circuit, [{'layer': layer, 'index': index}])
    r['feature_label'] = label
    results[key] = r

print('\n=== Boost Test (L5/5793 at +20 on non-analogy prompt) ===')
BOOST_PROMPT = 'The weather in Berlin today is'
print(f'  prompt: "{BOOST_PROMPT}"')
status, default, steered = steer_call(
    BOOST_PROMPT, [{'layer': 5, 'index': 5793}], strength=20, n_tokens=15)
if status == 200:
    d_tok = first_tok(default, BOOST_PROMPT)
    s_tok = first_tok(steered,  BOOST_PROMPT)
    changed = d_tok.lower().rstrip('.,;') != s_tok.lower().rstrip('.,;')
    print(f'  default: {d_tok!r}  steered: {s_tok!r}  changed: {changed}')
    results['boost'] = {
        'label': 'Boost L5/5793 (+20) non-analogy prompt',
        'circuit': 'boost', 'prompt': BOOST_PROMPT, 'api_status': 200,
        'default': default.replace('<bos>', '').strip()[:80],
        'steered': steered.replace('<bos>', '').strip()[:80],
        'default_first': d_tok, 'steered_first': s_tok, 'changed': changed,
    }
else:
    print(f'  !! HTTP {status}')
    results['boost'] = {'api_status': status, 'changed': False,
                        'prompt': BOOST_PROMPT}

# ── save ─────────────────────────────────────────────────────────────────────

out = r'C:\Users\USER\Desktop\Olalekan Alagbe\steering_pending_results.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=True)

print(f'\nDone. {len(results)} results saved to steering_pending_results.json')
