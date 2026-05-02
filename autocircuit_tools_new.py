"""
AutoCircuit Graph Utils - Local Python Version
Adapted from the Colab notebook for local execution.
"""

import os
import json
import time
import requests
import networkx as nx
from pathlib import Path
from typing import Optional

# ── Load API Key ──
NEURONPEDIA_API_KEY = os.environ.get('NEURONPEDIA_API_KEY', 'sk-np-DinhN2coo82hQFJXSzNsAQtQl6RbvEj3XMVO6ZrLcME0')

# ── Configuration ──
BASE_URL = 'https://www.neuronpedia.org/api'
MODEL_ID = 'gemma-2-2b'
SAE_ID   = 'gemmascope-transcoder-16k'

HEADERS = {
    'Content-Type': 'application/json'
}
if NEURONPEDIA_API_KEY:
    HEADERS['X-Api-Key'] = NEURONPEDIA_API_KEY

# Steering uses a DIFFERENT model from graph analysis.
# The /api/steer endpoint only supports 'gemma-2b' with 'res-jb' SAE layers.
# MODEL_ID ('gemma-2-2b') remains unchanged for all graph/feature functions.
STEER_MODEL_ID = 'gemma-2b'

# The 13 most important named features from the paper, organised by phase.
# Use with get_quick_list_url() to browse them all on Neuronpedia.
CORE_CIRCUIT_FEATURES = [
    # Phase 1 — Circuit Template (syntactic scaffold)
    {'layer': 0,  'index': 11651, 'label': "the word 'to'"},
    {'layer': 1,  'index': 11356, 'label': "'to' followed by a verb"},
    {'layer': 2,  'index': 11475, 'label': "the word 'refers'"},
    {'layer': 4,  'index': 10752, 'label': "'to be' preceded by 'to'"},
    {'layer': 5,  'index': 9672,  'label': "the phrase 'it is to'"},
    # Phase 2 — Analogy Recognition Hub (the key relational features)
    {'layer': 5,  'index': 5793,  'label': 'analogies'},
    {'layer': 5,  'index': 2141,  'label': 'comparisons of public figures'},
    {'layer': 8,  'index': 13766, 'label': 'analogies or comparisons'},
    {'layer': 9,  'index': 13344, 'label': 'comparison between two things'},
    # Phase 3 — Relational Integration
    {'layer': 11, 'index': 15947, 'label': 'historical or social change'},
    {'layer': 13, 'index': 10969, 'label': 'comparisons between disciplines'},
    # High-recurrence formal text features (predicted to be causally inert)
    {'layer': 4,  'index': 14857, 'label': 'code snippets and license agreements'},
    {'layer': 6,  'index': 2267,  'label': 'programming code and legal jargon'},
]

# The 4 Phase 2 features — the heart of the paper's causal claim.
# Suppressing all 4 together should cause the model to output 'France'
# instead of the correct target country on capital analogy prompts.
PHASE_2_FEATURES = [
    {'layer': 5, 'index': 5793,  'label': 'analogies'},
    {'layer': 5, 'index': 2141,  'label': 'comparisons of public figures'},
    {'layer': 8, 'index': 13766, 'label': 'analogies or comparisons'},
    {'layer': 9, 'index': 13344, 'label': 'comparison between two things'},
]

# ── Local Storage Folders ──
GRAPHS_DIR   = Path('graphs')
CIRCUITS_DIR = Path('circuits')
GRAPHS_DIR.mkdir(exist_ok=True)
CIRCUITS_DIR.mkdir(exist_ok=True)


def generate_graph(prompt: str, slug: str = None, save: bool = True) -> dict:
    """
    Request Neuronpedia to generate an attribution graph for a given prompt.
    """
    if slug is None:
        slug = prompt.lower().replace(' ', '_')[:40].strip('_')

    payload = {
        'modelId': MODEL_ID,
        'prompt': prompt,
        'slug': slug,
        'maxFeatureNodes': 3000,
        'desiredLogitProb': 0.95,
        'nodeThreshold': 0.8,
        'edgeThreshold': 0.85,
    }

    print(f'[generate_graph] Requesting graph for: "{prompt}"')
    response = requests.post(f'{BASE_URL}/graph/generate', json=payload, headers=HEADERS)

    if response.status_code != 200:
        raise RuntimeError(
            f'Graph generation failed: {response.status_code} — {response.text}'
        )

    graph_data = response.json()

    if save:
        path = GRAPHS_DIR / f'{slug}.json'
        with open(path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f'[generate_graph] Saved to {path}')

    return graph_data


def fetch_existing_graph(slug: str) -> dict:
    """
    Fetch an already-generated graph from Neuronpedia by its slug.
    """
    url = f'{BASE_URL}/graph/{slug}'
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise RuntimeError(f'Failed to fetch graph "{slug}": {response.status_code}')

    graph_data = response.json()
    path = GRAPHS_DIR / f'{slug}.json'
    with open(path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f'[fetch_existing_graph] Saved "{slug}" -> {path}')
    return graph_data


def load_graph(path: str) -> nx.DiGraph:
    """
    Load a saved attribution graph JSON and convert it to a NetworkX DiGraph.
    """
    with open(path) as f:
        data = json.load(f)

    G = nx.DiGraph()

    metadata = data.get('metadata', {})
    G.graph['slug']          = metadata.get('slug', '')
    G.graph['prompt']        = metadata.get('prompt', '')
    G.graph['prompt_tokens'] = metadata.get('prompt_tokens', [])
    G.graph['model']         = metadata.get('scan', MODEL_ID)

    for node in data.get('nodes', []):
        node_id = node['node_id']
        G.add_node(node_id,
            feature         = node.get('feature'),
            layer           = node.get('layer'),
            ctx_idx         = node.get('ctx_idx'),
            feature_type    = node.get('feature_type', ''),
            influence       = node.get('influence', 0.0),
            activation      = node.get('activation', 0.0),
            is_target_logit = node.get('is_target_logit', False),
            clerp           = node.get('clerp', ''),
            token_prob      = node.get('token_prob', 0.0),
        )

    for link in data.get('links', []):
        G.add_edge(link['source'], link['target'], weight=link.get('weight', 0.0))

    print(f'[load_graph] "{G.graph["prompt"]}"')
    print(f'             {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    return G


def get_graph_summary(G: nx.DiGraph) -> dict:
    """
    Return a high-level summary of the graph.
    """
    influences = [attrs.get('influence', 0.0) for _, attrs in G.nodes(data=True)]

    layers = sorted(
        set(str(attrs.get('layer')) for _, attrs in G.nodes(data=True)
            if attrs.get('layer') is not None),
        key=lambda x: int(x) if x.lstrip('-').isdigit() else 999
    )

    logit_count = sum(
        1 for _, attrs in G.nodes(data=True) if attrs.get('is_target_logit')
    )

    return {
        'prompt'         : G.graph.get('prompt', ''),
        'model'          : G.graph.get('model', ''),
        'num_nodes'      : G.number_of_nodes(),
        'num_edges'      : G.number_of_edges(),
        'layers_present' : layers,
        'num_logit_nodes': logit_count,
        'top_5_nodes'    : get_top_nodes(G, n=5),
        'max_influence'  : round(max(influences), 4) if influences else 0,
        'avg_influence'  : round(sum(influences) / len(influences), 4) if influences else 0,
    }


def get_top_nodes(G: nx.DiGraph, n: int = 20, exclude_types: list = None) -> list:
    """
    Return the top-N most influential nodes in the graph.
    """
    exclude_types = exclude_types or []
    nodes = []

    for node_id, attrs in G.nodes(data=True):
        if attrs.get('feature_type') in exclude_types:
            continue
        nodes.append({
            'node_id'        : node_id,
            'feature'        : attrs.get('feature'),
            'layer'          : attrs.get('layer'),
            'ctx_idx'        : attrs.get('ctx_idx'),
            'feature_type'   : attrs.get('feature_type'),
            'influence'      : round(attrs.get('influence', 0.0), 4),
            'activation'     : round(attrs.get('activation', 0.0), 4),
            'is_target_logit': attrs.get('is_target_logit', False),
            'clerp'          : attrs.get('clerp', ''),
        })

    nodes.sort(key=lambda x: x['influence'], reverse=True)
    return nodes[:n]


def get_edges_for_node(G: nx.DiGraph, node_id: str, top_k: int = 10) -> dict:
    """
    Get all incoming and outgoing edges for a specific node.
    """
    incoming = [
        {'source': u, 'weight': round(data['weight'], 4)}
        for u, v, data in G.in_edges(node_id, data=True)
    ]
    outgoing = [
        {'target': v, 'weight': round(data['weight'], 4)}
        for u, v, data in G.out_edges(node_id, data=True)
    ]

    incoming.sort(key=lambda x: abs(x['weight']), reverse=True)
    outgoing.sort(key=lambda x: abs(x['weight']), reverse=True)

    return {
        'node_id'   : node_id,
        'incoming'  : incoming[:top_k],
        'outgoing'  : outgoing[:top_k],
        'in_degree' : G.in_degree(node_id),
        'out_degree': G.out_degree(node_id),
    }


def get_nodes_by_layer(G: nx.DiGraph, layer: str) -> list:
    """
    Return all nodes at a specific transformer layer, sorted by influence.
    """
    nodes = [
        {
            'node_id'   : nid,
            'feature'   : attrs.get('feature'),
            'ctx_idx'   : attrs.get('ctx_idx'),
            'influence' : round(attrs.get('influence', 0.0), 4),
            'activation': round(attrs.get('activation', 0.0), 4),
            'clerp'     : attrs.get('clerp', ''),
        }
        for nid, attrs in G.nodes(data=True)
        if str(attrs.get('layer')) == str(layer)
    ]
    nodes.sort(key=lambda x: x['influence'], reverse=True)
    return nodes


def compare_graphs(graphs: list, min_appearances: int = None) -> list:
    """
    Find nodes (features) that appear consistently across multiple graphs.
    """
    if min_appearances is None:
        min_appearances = len(graphs) // 2 + 1

    registry = {}

    for G in graphs:
        slug = G.graph.get('slug', 'unknown')
        for node_id, attrs in G.nodes(data=True):
            if attrs.get('feature_type') not in ('cross layer transcoder', 'transcoder'):
                continue
            layer   = attrs.get('layer')
            feature = attrs.get('feature')
            if layer is None or feature is None:
                continue

            key = (str(layer), str(feature))
            if key not in registry:
                registry[key] = []
            registry[key].append({
                'graph_slug': slug,
                'node_id'   : node_id,
                'influence' : attrs.get('influence', 0.0),
                'activation': attrs.get('activation', 0.0),
            })

    results = []
    for (layer, feature), occurrences in registry.items():
        if len(occurrences) < min_appearances:
            continue
        avg_inf = sum(o['influence']  for o in occurrences) / len(occurrences)
        avg_act = sum(o['activation'] for o in occurrences) / len(occurrences)
        results.append({
            'layer'         : layer,
            'feature'       : feature,
            'appearances'   : len(occurrences),
            'out_of'        : len(graphs),
            'avg_influence' : round(avg_inf, 4),
            'avg_activation': round(avg_act, 4),
            'graph_slugs'   : [o['graph_slug'] for o in occurrences],
            'node_ids'      : [o['node_id']    for o in occurrences],
        })

    results.sort(key=lambda x: (x['appearances'], x['avg_influence']), reverse=True)
    return results


def label_node(node_id: str, model_id: str = MODEL_ID, sae_id: str = SAE_ID) -> dict:
    """
    Translate a node's feature ID into human-readable English via Neuronpedia.
    """
    parts = node_id.split('_')
    if len(parts) < 2:
        return {'node_id': node_id, 'explanation': None, 'error': 'Unparseable node_id'}

    layer_str  = parts[0]
    feature_id = parts[1]

    if layer_str.upper() == 'E' or not feature_id.isdigit():
        return {
            'node_id'    : node_id,
            'layer'      : layer_str,
            'feature'    : feature_id,
            'explanation': f'[{layer_str} node — embedding or error term, no SAE feature]',
            'examples'   : [],
        }

    url = f'{BASE_URL}/feature/{model_id}/{sae_id}/{feature_id}'
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return {
            'node_id'    : node_id,
            'layer'      : layer_str,
            'feature'    : feature_id,
            'explanation': None,
            'error'      : f'API error {response.status_code}',
        }

    data = response.json()
    explanations = data.get('explanations', [])
    explanation  = explanations[0].get('description') if explanations else None

    return {
        'node_id'        : node_id,
        'layer'          : layer_str,
        'feature'        : feature_id,
        'explanation'    : explanation,
        'max_activation' : data.get('maxActApprox'),
        'positive_tokens': data.get('pos_str', [])[:10],
        'negative_tokens': data.get('neg_str', [])[:10],
        'examples'       : [
            {
                'tokens'    : act.get('tokens'),
                'max_value' : act.get('maxValue'),
                'peak_token': act['tokens'][act.get('maxValueTokenIndex', -1)]
                              if act.get('tokens') else None
            }
            for act in data.get('activations', [])[:3]
        ],
    }


def label_nodes_batch(node_ids: list, delay: float = 0.3) -> list:
    """
    Label multiple nodes with a delay between requests to respect rate limits.
    """
    results = []
    for i, nid in enumerate(node_ids):
        print(f'  Labeling {i+1}/{len(node_ids)}: {nid}')
        result = label_node(nid)
        results.append(result)
        time.sleep(delay)
    return results


def steer_feature(prompt: str, features: list,
                  strength_multiplier: float = 4.0,
                  n_tokens: int = 10,
                  temperature: float = 0.2) -> dict:
    """
    Steer one or more SAE features and observe how the model output changes.

    Uses STEER_MODEL_ID ('gemma-2b') and 'res-jb' layer format internally.
    The global MODEL_ID ('gemma-2-2b') is NOT used here — do not change it.

    Parameters
    ----------
    prompt : str
        The input prompt.
    features : list of dicts, each with:
        'layer'    : int   e.g. 5
        'index'    : int   e.g. 5793
        'strength' : float e.g. -20 to suppress, +20 to boost
        Pass an empty list [] for a baseline run with no steering.
    strength_multiplier : float
        Scales all strengths globally. Effective strength = strength x multiplier.
    n_tokens : int
        Number of tokens to generate.
    temperature : float
        Sampling temperature.

    Returns
    -------
    dict with keys: prompt, baseline_output, steered_output, changed, features_used
    """
    url = f'{BASE_URL}/steer'
    headers = {'Content-Type': 'application/json'}

    # Convert to the format the steer API expects.
    # Layer MUST be '{layer}-res-jb', not the transcoder format.
    formatted_features = [
        {
            'modelId' : STEER_MODEL_ID,
            'layer'   : f'{f["layer"]}-res-jb',
            'index'   : f['index'],
            'strength': f['strength'],
        }
        for f in features
    ]

    # Step 1: baseline — run with no features steered
    baseline_payload = {
        'prompt'             : prompt,
        'modelId'            : STEER_MODEL_ID,
        'features'           : [],
        'temperature'        : temperature,
        'n_tokens'           : n_tokens,
        'freq_penalty'       : 1.0,
        'seed'               : 42,
        'strength_multiplier': strength_multiplier,
    }
    baseline_resp = requests.post(url, json=baseline_payload, headers=headers)
    if baseline_resp.status_code != 200:
        print(f'[steer_feature] Baseline call failed: {baseline_resp.status_code}')
        print(f'Response: {baseline_resp.text[:500]}')
        return {'error': f'Baseline API error {baseline_resp.status_code}',
                'prompt': prompt}

    baseline_output = baseline_resp.json().get('output', '')

    # Step 2: steered — run with features applied
    steer_payload = {
        'prompt'             : prompt,
        'modelId'            : STEER_MODEL_ID,
        'features'           : formatted_features,
        'temperature'        : temperature,
        'n_tokens'           : n_tokens,
        'freq_penalty'       : 1.0,
        'seed'               : 42,
        'strength_multiplier': strength_multiplier,
    }
    steer_resp = requests.post(url, json=steer_payload, headers=headers)
    if steer_resp.status_code != 200:
        print(f'[steer_feature] Steer call failed: {steer_resp.status_code}')
        print(f'Response: {steer_resp.text[:500]}')
        return {'error': f'Steer API error {steer_resp.status_code}',
                'prompt': prompt}

    steered_output = steer_resp.json().get('output', '')

    result = {
        'prompt'         : prompt,
        'baseline_output': baseline_output,
        'steered_output' : steered_output,
        'changed'        : baseline_output.strip() != steered_output.strip(),
        'features_used'  : features,
    }

    print(f'Prompt   : "{prompt}"')
    print(f'Baseline : "{baseline_output.strip()}"')
    print(f'Steered  : "{steered_output.strip()}"')
    print(f'Changed  : {result["changed"]}')

    return result


def get_feature_embed_url(layer: int, feature_index: int) -> str:
    """
    Returns the Neuronpedia iframe embed URL for a given feature.
    Example: layer=5, index=5793 ->
    'https://neuronpedia.org/gemma-2-2b/5-gemmascope-transcoder-16k/5793?embed=true'
    """
    return (
        f'https://neuronpedia.org/{MODEL_ID}/'
        f'{layer}-gemmascope-transcoder-16k/{feature_index}?embed=true'
    )


def get_feature_iframe(layer: int, feature_index: int,
                       height: int = 300, width: int = 540) -> str:
    """
    Returns a complete HTML iframe string ready to paste into a webpage.
    """
    url = get_feature_embed_url(layer, feature_index)
    return (
        f'<iframe src="{url}" '
        f'title="Neuronpedia Feature L{layer} #{feature_index}" '
        f'style="height:{height}px;width:{width}px;'
        f'border:1px solid #ddd;border-radius:6px;"></iframe>'
    )


def get_quick_list_url(features: list, list_name: str) -> str:
    """
    Build a Neuronpedia Quick List URL from a list of features.
    Open the returned URL in a browser to see all features on one page.

    Parameters
    ----------
    features : list of dicts with keys 'layer' and 'index'
    list_name : str — the title shown on the quick list page
    """
    import urllib.parse

    formatted = [
        {
            'modelId': MODEL_ID,
            'layer'  : f'{f["layer"]}-gemmascope-transcoder-16k',
            'index'  : str(f['index']),
        }
        for f in features
    ]

    url = 'https://neuronpedia.org/quick-list/'
    url += '?name=' + urllib.parse.quote(list_name)
    url += '&features=' + urllib.parse.quote(json.dumps(formatted))
    return url


def run_causal_validation():
    """
    Run the 3 experiments that directly test the paper's core causal claim.

    The claim: Phase 2 features collectively implement relational transfer
    in analogical reasoning. Suppressing all 4 together should cause the
    model to output 'France' (the source pair) instead of 'Germany'
    (the correct target country).

    Experiment 1 — Baseline
        No steering. Confirm the model outputs 'Germany'.

    Experiment 2 — Suppress all 4 Phase 2 features simultaneously
        If the claim holds, the output should become 'France'.
        This is the paper's core causal necessity test.

    Experiment 3 — Boost L5 #5793 on a non-analogy prompt
        Force-inject the 'analogies' feature into a weather prompt.
        If the feature is causally sufficient, output should shift
        toward analogy-like structure.

    Total: 6 API calls (2 per experiment — baseline + steered).

    Note: Uses gemma-2b + res-jb SAE (the only model/SAE the steer API
    supports). Feature indices come from the paper's transcoder analysis
    and may not transfer perfectly across SAE types.
    """
    ANALOGY_PROMPT     = 'Paris is to France as Berlin is to'
    NON_ANALOGY_PROMPT = 'The weather in Berlin today is'

    print('=' * 60)
    print('CAUSAL VALIDATION — 3 EXPERIMENTS')
    print('Core claim: suppressing Phase 2 features collapses')
    print('analogical reasoning and produces "France" instead')
    print('=' * 60)

    # Experiment 1: Baseline — no steering
    print('\nExperiment 1: Baseline (no steering)')
    print('Expected: Germany')
    exp1 = steer_feature(prompt=ANALOGY_PROMPT, features=[])
    print(f'RESULT: "{exp1.get("baseline_output", "").strip()}"')

    # Experiment 2: Suppress all 4 Phase 2 features at once
    print('\nExperiment 2: Suppress all 4 Phase 2 features simultaneously')
    print('Expected: France (model loses relational transfer,')
    print('falls back to the source-pair association)')
    suppress_features = [
        {'layer': f['layer'], 'index': f['index'], 'strength': -20}
        for f in PHASE_2_FEATURES
    ]
    exp2 = steer_feature(prompt=ANALOGY_PROMPT, features=suppress_features)
    steered = exp2.get('steered_output', '')
    print(f'RESULT: "{steered.strip()}"')
    if 'france' in steered.lower():
        print('CLAIM SUPPORTED: output contains "France"')
    elif exp2.get('changed'):
        print('PARTIAL: output changed but does not contain "France"')
    else:
        print('NO EFFECT: output did not change')

    # Experiment 3: Boost analogy feature on a non-analogy prompt
    print('\nExperiment 3: Boost L5 #5793 ("analogies") on a weather prompt')
    print('Expected: output shifts toward analogy-like structure')
    exp3 = steer_feature(
        prompt=NON_ANALOGY_PROMPT,
        features=[{'layer': 5, 'index': 5793, 'strength': 20}],
    )
    print(f'Baseline : "{exp3.get("baseline_output", "").strip()}"')
    print(f'Steered  : "{exp3.get("steered_output", "").strip()}"')
    if exp3.get('changed'):
        print('Output changed — feature has causal influence on this model')
    else:
        print('No change — feature index may not transfer across SAE types')

    print('\n' + '=' * 60)
    print('Done. 3 experiments, 6 total API calls.')
    print('=' * 60)

    return {'exp1': exp1, 'exp2': exp2, 'exp3': exp3}


def save_circuit(name: str, nodes: list, description: str,
                 prompt_category: str, source_graphs: list,
                 validation_results: list = None) -> str:
    """
    Save a discovered circuit hypothesis to disk.
    """
    circuit = {
        'name'              : name,
        'description'       : description,
        'prompt_category'   : prompt_category,
        'source_graphs'     : source_graphs,
        'num_source_graphs' : len(source_graphs),
        'nodes'             : nodes,
        'validation_results': validation_results or [],
        'created_at'        : time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    filename = name.replace(' ', '_').lower() + '.json'
    path = CIRCUITS_DIR / filename
    with open(path, 'w') as f:
        json.dump(circuit, f, indent=2)

    print(f'[save_circuit] Saved "{name}" -> {path}')
    return str(path)


def build_prompt_dataset() -> dict:
    """
    Returns structured prompts organised by reasoning category.
    """
    return {
        'factual_recall': [
            'The capital of France is',
            'The capital of Germany is',
            'The capital of Japan is',
            'The capital of England is',
            'The capital of Italy is',
            'The capital of Spain is',
            'The capital of Brazil is',
            'The capital of Canada is',
            'The capital of Australia is',
            'The capital of China is',
        ],
        'arithmetic': [
            '12 plus 5 equals',
            '8 plus 3 equals',
            '15 plus 4 equals',
            '7 plus 9 equals',
            '20 plus 6 equals',
            '3 times 4 equals',
            '6 times 7 equals',
            '9 times 3 equals',
        ],
        'linguistic': [
            'The synonym of happy is',
            'The synonym of fast is',
            'The synonym of cold is',
            'The opposite of hot is',
            'The opposite of dark is',
            'The opposite of large is',
        ],
        'analogical': [
            'Paris is to France as Berlin is to',
            'Paris is to France as Rome is to',
            'Paris is to France as Tokyo is to',
            'Doctor is to hospital as teacher is to',
            'Fish is to water as bird is to',
        ],
    }


def test_connection() -> bool:
    """
    Test connection to Neuronpedia API.
    """
    print('Testing Neuronpedia API connection...')
    print(f'API key loaded: {"YES" if NEURONPEDIA_API_KEY else "NO (set NEURONPEDIA_API_KEY env var)"}')

    try:
        url = f'{BASE_URL}/graph/generate'
        payload = {
            'modelId': MODEL_ID,
            'prompt': 'test',
            'slug': 'connection_test',
            'maxFeatureNodes': 3000,
        }
        response = requests.post(url, json=payload, headers=HEADERS, timeout=30)
        if response.status_code in (200, 400):
            print('Connection successful! API is reachable.')
            return True
        else:
            print(f'Connection test returned status {response.status_code}')
            print(f'Response: {response.text[:500]}')
            return False
    except Exception as e:
        print(f'Connection failed: {e}')
        return False


if __name__ == '__main__':
    if test_connection():
        print('\nNeuronpedia is connected and ready!')
        print(f'Graphs folder  : {GRAPHS_DIR.resolve()}')
        print(f'Circuits folder: {CIRCUITS_DIR.resolve()}')
        print('\nTo run the 3 core causal validation experiments:')
        print('  run_causal_validation()')
        print('\nTo browse all core circuit features on Neuronpedia:')
        print(get_quick_list_url(CORE_CIRCUIT_FEATURES,
                                 'Analogical Reasoning Core Circuit'))
    else:
        print('\nConnection test failed. Please check your API key.')
