import os
import sys
import json
import re
import math
import hashlib
import logging
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import Generator, Dict, List, Any, Set, Tuple, Optional

# ================================================================
# 1. INTEGRATED CONFIGURATION (IKF / VOYAGER-FOCUSED)
# ================================================================
CONFIG = {
    # --- Harvester Settings ---
    "BASE_URL": "https://spdf.gsfc.nasa.gov/pub/",
    "TARGET_DIRS": [
        "documents/",
        "catalogs/",
        "software/",
        "models/"
    ],
    "MAX_DEPTH": 5,
    "DELAY_SECONDS": 0.2,
    "USER_AGENT": "NASA-SPDF-Knowledge-Miner/2.1 (IKF-VOYAGER)",
    # --- Miner Settings ---
    "ENTROPY_THRESHOLD": 3.5,
    "MIN_CONTENT_LENGTH": 50,
    "OUTPUT_JSON": "NASA_SPDF_IKF_Knowledge_Graph.json",
    "OUTPUT_MD": "NASA_SPDF_IKF_Report.md",
    "LOG_LEVEL": logging.INFO,
    # IKF/VOYAGER FOCUS: keywords to rank/filter content
    "FOCUS_KEYWORDS": [
        "voyager", "pioneer", "heliopause", "heliosheath",
        "magnetic turbulence", "alfven", "alfvén", "plasma",
        "solar wind", "interstellar medium", "memory dump",
        "telemetry", "bit error", "cosmic ray"
    ]
}

# --- Physics Constants & Units ---
SI_BASE = {
    'kg': {'M': 1}, 'm': {'L': 1}, 's': {'T': 1},
    'A': {'I': 1}, 'K': {'K': 1}, 'mol': {'N': 1}
}
SI_DERIVED = {
    'newton': {'M': 1, 'L': 1, 'T': -2},
    'joule': {'M': 1, 'L': 2, 'T': -2},
    'watt': {'M': 1, 'L': 2, 'T': -3},
    'pascal': {'M': 1, 'L': -1, 'T': -2},
    'hertz': {'T': -1},
    'coulomb': {'T': 1, 'I': 1},
    'volt': {'M': 1, 'L': 2, 'T': -3, 'I': -1},
    'tesla': {'M': 1, 'T': -2, 'I': -1},
    'nanotesla': {'M': 1, 'T': -2, 'I': -1},
    'entropy': {'M': 1, 'L': 2, 'T': -2, 'K': -1}
}

logging.basicConfig(
    level=CONFIG["LOG_LEVEL"],
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("NASA_INTEGRATOR")

# ================================================================
# 2. PHYSICS & MATH ENGINE
# ================================================================
class DimensionalValidator:
    @staticmethod
    def get_dimensional_signature(text: str) -> Optional[str]:
        text = text.lower()
        for unit, dims in SI_DERIVED.items():
            if unit in text:
                return DimensionalValidator._format_dims(dims)
        for unit, dims in SI_BASE.items():
            if re.search(rf"\b{unit}\b", text):
                return DimensionalValidator._format_dims(dims)
        return None

    @staticmethod
    def _format_dims(dims: Dict[str, int]) -> str:
        sig = []
        for k in sorted(dims.keys()):
            val = dims[k]
            if val == 1:
                sig.append(k)
            elif val != 0:
                sig.append(f"{k}^{val}")
        return " ".join(sig)

# ================================================================
# 3. NETWORK & STREAMING ENGINE
# ================================================================
class EntropyGate:
    @staticmethod
    def calculate(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        return -sum((c / length) * math.log(c / length, 2)
                    for c in counts.values())

    @staticmethod
    def is_signal(text: str) -> bool:
        if len(text) < CONFIG["MIN_CONTENT_LENGTH"]:
            return False
        return EntropyGate.calculate(text) > CONFIG["ENTROPY_THRESHOLD"]

class UnifiedStreamer:
    """
    Handles remote Text/HTML content.
    """
    @staticmethod
    def _focus_score(text: str) -> int:
        """IKF/VOYAGER FOCUS: rough relevance score."""
        tl = text.lower()
        return sum(1 for kw in CONFIG["FOCUS_KEYWORDS"] if kw in tl)

    @staticmethod
    def parse_nasa_content(content_text: str, source_url: str) -> Generator[Dict, None, None]:
        # If HTML, strip tags
        if source_url.endswith('.html') or '<html' in content_text.lower():
            soup = BeautifulSoup(content_text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ')
        else:
            text = content_text

        paragraphs = re.split(r'\n\s*\n', text)
        for p in paragraphs:
            clean_p = p.strip()
            if not EntropyGate.is_signal(clean_p):
                continue
            score = UnifiedStreamer._focus_score(clean_p)
            if score == 0:
                # Skip paragraphs unrelated to Voyager/IKF/plasma
                continue
            yield {
                "role": "system_archive",
                "time": datetime.now().isoformat(),
                "content": clean_p,
                "source": source_url,
                "focus_score": score
            }

# ================================================================
# 4. KNOWLEDGE MINING ENGINE
# ================================================================
class KnowledgeMiner:
    PATTERNS = {
        'MATH': re.compile(r'(?:\$\$(.*?)\$\$|\\\[(.*?)\\\])', re.DOTALL),
        'DEF': re.compile(
            r'(?i)(?P<term>[A-Z][a-zA-Z0-9\s\-]+)\s*'
            r'(?:is defined as|:=|refers to|represents)\s*'
            r'(?P<def>.{10,300})'
        ),
        'CODE': re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL),
        'LOGIC': re.compile(
            r'(?i)(?:implies|therefore|consequently|leads to)\s+(.{10,150})'
        ),
        'SIM_PARAM': re.compile(
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*'
            r'([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+)'
        )
    }

    @staticmethod
    def _tag_domain(content: str) -> str:
        """IKF/VOYAGER FOCUS: classify artifact domain."""
        cl = content.lower()
        if any(k in cl for k in ["voyager", "pioneer", "heliopause", "heliosheath"]):
            return "SPACECRAFT_BOUNDARY"
        if any(k in cl for k in ["alfven", "alfvén", "plasma", "solar wind"]):
            return "PLASMA_TURBULENCE"
        if any(k in cl for k in ["memory dump", "telemetry", "bit error", "sram"]):
            return "TELEMETRY_MEMORY"
        return "GENERAL"

    @staticmethod
    def mine(content: str, meta: Dict) -> List[Dict]:
        artifacts: List[Dict] = []
        source = meta['source']
        domain = KnowledgeMiner._tag_domain(content)

        # 1. Math
        for m in KnowledgeMiner.PATTERNS['MATH'].finditer(content):
            eq = m.group(1) or m.group(2)
            if eq and len(eq) > 5:
                dim_sig = DimensionalValidator.get_dimensional_signature(content)
                artifacts.append({
                    "id": hashlib.md5(eq.encode()).hexdigest()[:8],
                    "type": "MATH",
                    "content": eq.strip(),
                    "dim_signature": dim_sig or "Unknown",
                    "domain": domain,
                    "source": source
                })

        # 2. Parameters
        for m in KnowledgeMiner.PATTERNS['SIM_PARAM'].finditer(content):
            key, val = m.groups()
            if key.lower() in ['i', 'j', 'x', 'y', 'width', 'height', 'version']:
                continue
            artifacts.append({
                "id": hashlib.md5(f"{key}{val}".encode()).hexdigest()[:8],
                "type": "PARAM",
                "key": key,
                "value": val,
                "domain": domain,
                "source": source
            })

        # 3. Definitions
        for m in KnowledgeMiner.PATTERNS['DEF'].finditer(content):
            term = m.group('term').strip()
            definition = m.group('def').strip()
            artifacts.append({
                "id": hashlib.md5(term.encode()).hexdigest()[:8],
                "type": "CONCEPT",
                "term": term,
                "definition": definition,
                "domain": domain,
                "source": source
            })

        return artifacts

# ================================================================
# 5. SPDF REMOTE HARVESTER
# ================================================================
class SPDFHarvester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.visited: Set[str] = set()
        self.queue: deque = deque()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': CONFIG['USER_AGENT']})

        for target in CONFIG['TARGET_DIRS']:
            full_url = urljoin(base_url, target)
            self.queue.append((full_url, 0))

    def is_valid_link(self, href: str) -> bool:
        if not href or href.startswith('?') or href in ['/', '../', './']:
            return False
        return True

    def is_knowledge_file(self, href: str) -> bool:
        exts = ('.txt', '.html', '.md', '.cat', '.sf', '.h', '.c', '.pro')
        href_l = href.lower()
        if href_l.endswith(exts):
            return True
        return False

    def fetch_and_yield(self) -> Generator[Dict, None, None]:
        logger.info(f"Starting Scan of {self.base_url}")
        while self.queue:
            current_url, depth = self.queue.popleft()
            if current_url in self.visited:
                continue
            self.visited.add(current_url)
            time.sleep(CONFIG['DELAY_SECONDS'])
            try:
                response = self.session.get(current_url, timeout=10)
                if response.status_code != 200:
                    continue
                content_type = response.headers.get('Content-Type', '')

                # A. Directory
                if 'text/html' in content_type and current_url.endswith('/'):
                    if depth >= CONFIG['MAX_DEPTH']:
                        continue
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a'):
                        href = link.get('href')
                        if not self.is_valid_link(href):
                            continue
                        full_link = urljoin(current_url, href)
                        if href.endswith('/'):
                            if full_link.startswith(self.base_url) and full_link not in self.visited:
                                self.queue.append((full_link, depth + 1))
                        elif self.is_knowledge_file(href):
                            self.queue.append((full_link, 999))

                # B. File download marker
                elif depth == 999:
                    logger.info(f"Mining Knowledge from: {current_url}")
                    yield from UnifiedStreamer.parse_nasa_content(
                        response.text, current_url
                    )

            except Exception as e:
                logger.error(f"Error at {current_url}: {e}")

# ================================================================
# 6. GRAPH & REPORTING
# ================================================================
class GraphSynthesizer:
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict] = []
        self.concept_index: Dict[str, List[str]] = defaultdict(list)

    def ingest(self, artifacts: List[Dict]):
        for art in artifacts:
            self.nodes[art['id']] = art
            if art['type'] == 'CONCEPT':
                self.concept_index[art['term'].lower()].append(art['id'])

    def build_edges(self):
        logger.info("Building semantic knowledge graph...")
        for art_id, art in self.nodes.items():
            blob = str(art.get('content', '') or art.get('definition', '')).lower()
            for term, ids in self.concept_index.items():
                if term in blob and art['type'] != 'CONCEPT':
                    for target_id in ids:
                        if target_id != art_id:
                            self.edges.append({
                                "source": art_id,
                                "target": target_id,
                                "rel": "cites"
                            })

    def generate_report(self) -> str:
        md = f"# NASA SPDF IKF Knowledge Report\n"
        md += f"**Generated:** {datetime.now().isoformat()} | "
        md += f"**Nodes:** {len(self.nodes)}\n\n"

        md += "## 1. Plasma / Alfvénic Equations\n"
        math_nodes = [n for n in self.nodes.values()
                      if n['type'] == 'MATH' and n.get('domain') == 'PLASMA_TURBULENCE']
        for n in math_nodes:
            if n['dim_signature'] != 'Unknown':
                md += f"- `${n['content']}$`\n  - *Dims*: `{n['dim_signature']}`\n"
                md += f"  - *Src*: {n['source']}\n"

        md += "\n## 2. Spacecraft / Boundary Concepts\n"
        concepts = [n for n in self.nodes.values()
                    if n['type'] == 'CONCEPT' and n.get('domain') == 'SPACECRAFT_BOUNDARY']
        concepts.sort(key=lambda x: len(x['definition']), reverse=True)
        for n in concepts[:30]:
            md += f"- **{n['term']}**: {n['definition']} (src: {n['source']})\n"

        md += "\n## 3. Telemetry / Memory Parameters\n"
        params = [n for n in self.nodes.values()
                  if n['type'] == 'PARAM' and n.get('domain') == 'TELEMETRY_MEMORY']
        seen = set()
        for n in params:
            key = n['key']
            if key in seen:
                continue
            md += f"- `{key}` = `{n['value']}` ([link]({n['source']}))\n"
            seen.add(key)

        return md

# ================================================================
# MAIN EXECUTION
# ================================================================
def main():
    start_time = time.time()
    harvester = SPDFHarvester(CONFIG["BASE_URL"])
    graph = GraphSynthesizer()

    print("--- NASA SPDF IKF KNOWLEDGE MINER ---")
    print(f"Targeting: {CONFIG['TARGET_DIRS']}")
    total_artifacts = 0

    try:
        for msg in harvester.fetch_and_yield():
            artifacts = KnowledgeMiner.mine(
                msg['content'], {"source": msg['source']}
            )
            if artifacts:
                graph.ingest(artifacts)
                total_artifacts += len(artifacts)
                print(f" + {len(artifacts)} artifacts from {msg['source'].split('/')[-1]}")
            if len(graph.nodes) > 2000:
                print("Hit demo limit of 2000 nodes.")
                break
    except KeyboardInterrupt:
        print("\nStopping scan gracefully...")

    print(f"\nScanning complete. {total_artifacts} artifacts extracted.")
    graph.build_edges()

    output_data = {
        "meta": {
            "timestamp": time.time(),
            "source": CONFIG["BASE_URL"]
        },
        "nodes": list(graph.nodes.values()),
        "edges": graph.edges
    }
    with open(CONFIG["OUTPUT_JSON"], 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    with open(CONFIG["OUTPUT_MD"], 'w', encoding='utf-8') as f:
        f.write(graph.generate_report())

    print("\nSUCCESS!")
    print(f"1. Knowledge Graph: {CONFIG['OUTPUT_JSON']}")
    print(f"2. Markdown Report: {CONFIG['OUTPUT_MD']}")
    print(f"Total Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    # Dependencies: pip install requests beautifulsoup4
    main()