import os
import re
import csv
import json
import math
import time
import hashlib
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator, Tuple
from collections import Counter, defaultdict, deque
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

CONFIG = {
    "BASE_URL": "https://spdf.gsfc.nasa.gov/pub/",
    "TARGET_DIRS": ["documents/", "catalogs/", "software/", "models/"],
    "MAX_DEPTH": 6,
    "DELAY_SECONDS": 0.2,
    "USER_AGENT": "SPDF-Conceptual-Miner/4.0",
    "ENTROPY_THRESHOLD": 3.0,
    "MIN_CONTENT_LENGTH": 24,
    "CHECKPOINT_EVERY": 1000,
    "DASHBOARD_EVERY": 300,
    "STATE_FILE": "output/miner_state.json",
    "LOG_FILE": "output/miner.log",
    "OUTPUT_DIR": "output/spdf_miner",
    "TEXT_EXTS": (".txt", ".asc", ".cat", ".lbl", ".htm", ".html", ".md", ".pro", ".sf", ".fmt", ".tab", ".csv", ".dat", ".h", ".c"),
    "FOCUS_KEYWORDS": [
        "voyager", "pioneer", "parker", "ulysses", "helios", "cassini", "galileo",
        "mag", "pws", "plasma", "solar wind", "heliopause", "heliosheath",
        "alfven", "alfvén", "turbulence", "reconnection", "shock", "boundary",
        "telemetry", "memory", "dump", "packet", "catalog", "idl", "model"
    ],
    "MODEL_BUCKETS": {
        "boundary_plasma": ["heliopause", "heliosheath", "boundary", "termination shock", "bow shock"],
        "alfvenic_transport": ["alfven", "alfvén", "poynting", "wave", "torsional", "switchback"],
        "turbulence_intermittency": ["turbulence", "intermittent", "multifractal", "spectrum", "anisotropy", "cross helicity"],
        "telemetry_memory": ["telemetry", "memory", "dump", "packet", "bit", "fds", "sram"],
        "instrument_calibration": ["calibration", "idl", "procedure", "catalog", "label", "format"]
    }
}

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SPDF_CONCEPTUAL_MINER")


class StateManager:
    def __init__(self, path: str):
        self.path = path
        self.state = {
            "visited": [],
            "processed_files": [],
            "records": 0,
            "files_seen": 0,
            "last_update": None
        }
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.state = json.load(f)
            except Exception:
                pass

    def save(self, visited, processed, records, files_seen):
        self.state = {
            "visited": list(visited),
            "processed_files": list(processed),
            "records": records,
            "files_seen": files_seen,
            "last_update": datetime.now().isoformat()
        }
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2)


class EntropyGate:
    @staticmethod
    def calculate(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        n = len(text)
        return -sum((c / n) * math.log(c / n, 2) for c in counts.values())

    @staticmethod
    def is_signal(text: str) -> bool:
        return len(text) >= CONFIG["MIN_CONTENT_LENGTH"] and EntropyGate.calculate(text) > CONFIG["ENTROPY_THRESHOLD"]


class LegacyNumberParser:
    @staticmethod
    def parse_num(s: str):
        s = s.strip().rstrip(',;')
        if not s:
            return s
        if re.fullmatch(r'[-+]?0[xX][0-9A-Fa-f]+', s):
            return int(s, 16)
        if re.fullmatch(r'[0-9A-Fa-f]{4,16}', s) and re.search(r'[A-Fa-f]', s):
            return int(s, 16)
        if re.search(r'[DdEe][+-]?\d+', s):
            try:
                return float(s.replace('D', 'E').replace('d', 'e'))
            except Exception:
                return s
        try:
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return s


class TextInterpreter:
    @staticmethod
    def classify_source(url: str) -> str:
        ul = url.lower()
        if ul.endswith('.pro'):
            return 'idl_source'
        if any(ul.endswith(x) for x in ('.cat', '.lbl', '.fmt')):
            return 'metadata'
        if any(ul.endswith(x) for x in ('.tab', '.csv', '.dat', '.asc')):
            return 'structured_table'
        if any(ul.endswith(x) for x in ('.htm', '.html', '.md', '.txt', '.sf')):
            return 'documentation'
        return 'other'

    @staticmethod
    def tag_mission(blob: str) -> str:
        blob = blob.lower()
        for m in ["voyager", "pioneer", "parker", "ulysses", "helios", "cassini", "galileo", "wind", "ace"]:
            if m in blob:
                return m
        return 'unknown'

    @staticmethod
    def tag_instrument(blob: str) -> str:
        blob = blob.lower()
        for inst in ["mag", "pws", "crs", "pls", "epi", "fds", "sweap", "fields"]:
            if re.search(rf'\b{inst}\b', blob):
                return inst
        return 'unknown'

    @staticmethod
    def assign_models(blob: str) -> List[str]:
        blob = blob.lower()
        out = []
        for model, kws in CONFIG["MODEL_BUCKETS"].items():
            if any(k in blob for k in kws):
                out.append(model)
        return out or ['general']

    @staticmethod
    def html_to_text(content: str) -> str:
        if '<html' in content.lower():
            soup = BeautifulSoup(content, 'html.parser')
            for t in soup(['script', 'style']):
                t.extract()
            return soup.get_text('\n')
        return content


class Decoder:
    @staticmethod
    def decode(source: str, text: str) -> Generator[Dict[str, Any], None, None]:
        text = TextInterpreter.html_to_text(text)
        stype = TextInterpreter.classify_source(source)
        mission = TextInterpreter.tag_mission(source + ' ' + text[:5000])
        instrument = TextInterpreter.tag_instrument(source + ' ' + text[:5000])

        if stype == 'idl_source':
            for i, line in enumerate(text.splitlines(), 1):
                raw = line.rstrip()
                if not raw.strip():
                    continue
                if m := re.match(r'^\s*pro\s+([A-Za-z0-9_]+)', raw, re.I):
                    rec = {
                        'record_type': 'idl_procedure', 'source': source, 'line_no': i,
                        'name': m.group(1), 'raw': raw, 'mission': mission, 'instrument': instrument
                    }
                    rec['models'] = TextInterpreter.assign_models(raw)
                    yield rec
                for var, val in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\n]+)', raw):
                    rec = {
                        'record_type': 'idl_assignment', 'source': source, 'line_no': i,
                        'field': var, 'raw_value': val.strip(), 'value': LegacyNumberParser.parse_num(val.strip()),
                        'raw': raw, 'mission': mission, 'instrument': instrument
                    }
                    rec['models'] = TextInterpreter.assign_models(raw)
                    yield rec

        for i, line in enumerate(text.splitlines(), 1):
            raw = line.rstrip()
            if not raw.strip():
                continue

            m = re.match(r'^\s*([0-9A-Fa-f]{4,8})\s*[: ]\s*((?:[0-9A-Fa-f]{2,8}\s+){1,16})(.*)$', raw)
            if m:
                base = int(m.group(1), 16)
                words = m.group(2).split()
                tail = m.group(3).strip()
                for j, w in enumerate(words):
                    rec = {
                        'record_type': 'memory_word', 'source': source, 'line_no': i,
                        'address': f"0x{base + j:06X}", 'word_index': j,
                        'raw_value': w, 'value': LegacyNumberParser.parse_num(w),
                        'comment': tail, 'raw': raw, 'mission': mission, 'instrument': instrument
                    }
                    rec['models'] = TextInterpreter.assign_models(raw + ' ' + tail)
                    yield rec
                continue

            cols = re.split(r'\s{2,}|\t+', raw.strip())
            if len(cols) >= 2:
                parsed = [LegacyNumberParser.parse_num(c) for c in cols]
                rec = {
                    'record_type': 'table_row', 'source': source, 'line_no': i,
                    'columns': cols, 'parsed_columns': parsed, 'raw': raw,
                    'mission': mission, 'instrument': instrument
                }
                rec['models'] = TextInterpreter.assign_models(raw)
                yield rec
            elif EntropyGate.is_signal(raw):
                rec = {
                    'record_type': 'text_line', 'source': source, 'line_no': i,
                    'raw': raw, 'mission': mission, 'instrument': instrument
                }
                rec['models'] = TextInterpreter.assign_models(raw)
                yield rec


class ConceptMiner:
    PATTERNS = {
        'equation': re.compile(r'(?:\$\$(.*?)\$\$|\\\[(.*?)\\\])', re.DOTALL),
        'definition': re.compile(r'(?i)(?P<term>[A-Z][A-Za-z0-9\s\-]+)\s+(?:is defined as|refers to|represents)\s+(?P<definition>.{10,300})'),
        'logic': re.compile(r'(?i)(?:implies|therefore|consequently|leads to)\s+(.{10,180})'),
        'parameter': re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)')
    }

    @staticmethod
    def mine(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = rec.get('raw', '')
        artifacts = []
        for m in ConceptMiner.PATTERNS['equation'].finditer(raw):
            eq = m.group(1) or m.group(2)
            if eq and len(eq.strip()) > 5:
                artifacts.append({'artifact_type': 'equation', 'content': eq.strip()})
        for m in ConceptMiner.PATTERNS['definition'].finditer(raw):
            artifacts.append({'artifact_type': 'definition', 'term': m.group('term').strip(), 'definition': m.group('definition').strip()})
        for m in ConceptMiner.PATTERNS['logic'].finditer(raw):
            artifacts.append({'artifact_type': 'logic', 'content': m.group(1).strip()})
        for m in ConceptMiner.PATTERNS['parameter'].finditer(raw):
            artifacts.append({'artifact_type': 'parameter', 'field': m.group(1), 'value': LegacyNumberParser.parse_num(m.group(2))})
        return artifacts


class DatasetWriter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.records_path = os.path.join(out_dir, 'all_records.jsonl')
        self.artifacts_path = os.path.join(out_dir, 'all_artifacts.jsonl')
        self.records_f = open(self.records_path, 'a', encoding='utf-8')
        self.artifacts_f = open(self.artifacts_path, 'a', encoding='utf-8')
        self.model_files = {}
        self.stats = Counter()
        self.model_stats = Counter()

    def _get_model_file(self, model: str):
        if model not in self.model_files:
            path = os.path.join(self.out_dir, f'{model}.jsonl')
            self.model_files[model] = open(path, 'a', encoding='utf-8')
        return self.model_files[model]

    def write_record(self, rec: Dict[str, Any]):
        rec = dict(rec)
        rec['record_id'] = hashlib.md5((rec.get('source', '') + rec.get('raw', '') + str(rec.get('line_no', ''))).encode()).hexdigest()[:12]
        self.records_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self.stats['records'] += 1
        for model in rec.get('models', ['general']):
            self._get_model_file(model).write(json.dumps(rec, ensure_ascii=False) + '\n')
            self.model_stats[model] += 1

    def write_artifact(self, art: Dict[str, Any], rec: Dict[str, Any]):
        out = dict(art)
        out['source'] = rec.get('source')
        out['mission'] = rec.get('mission')
        out['instrument'] = rec.get('instrument')
        out['models'] = rec.get('models', ['general'])
        self.artifacts_f.write(json.dumps(out, ensure_ascii=False) + '\n')
        self.stats['artifacts'] += 1

    def close(self):
        self.records_f.close()
        self.artifacts_f.close()
        for f in self.model_files.values():
            f.close()


class Dashboard:
    def __init__(self, out_dir: str, writer: DatasetWriter):
        self.out_dir = out_dir
        self.writer = writer
        self.path = os.path.join(out_dir, 'dashboard.html')

    def render(self, extra: Dict[str, Any]):
        models = ''.join(f"<li>{k}: {v}</li>" for k, v in self.writer.model_stats.most_common())
        html = f"""
<!doctype html>
<html><head><meta charset='utf-8'><meta http-equiv='refresh' content='30'>
<title>SPDF Miner Dashboard</title>
<style>
body{{font-family:Arial,sans-serif;background:#111;color:#eee;padding:24px}}
.card{{background:#1b1b1b;border:1px solid #333;border-radius:12px;padding:16px;margin:12px 0}}
.bar{{height:14px;background:#2a2a2a;border-radius:7px;overflow:hidden}}
.fill{{height:14px;background:linear-gradient(90deg,#0ea5e9,#22c55e)}}
.small{{color:#aaa;font-size:14px}}
</style></head>
<body>
<h1>SPDF Conceptual Miner</h1>
<div class='card'><b>Status:</b> running<br><span class='small'>Updated: {datetime.now().isoformat()}</span></div>
<div class='card'><b>Total records:</b> {self.writer.stats['records']}<br><b>Total artifacts:</b> {self.writer.stats['artifacts']}<br><b>Files seen:</b> {extra.get('files_seen', 0)}</div>
<div class='card'><b>Progress signal</b><div class='bar'><div class='fill' style='width:{min(100, extra.get('progress_pct', 0))}%'></div></div></div>
<div class='card'><b>Model datasets</b><ul>{models}</ul></div>
<div class='card'><b>Last source</b><div class='small'>{extra.get('last_source', '')}</div></div>
</body></html>
"""
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(html)


class Harvester:
    def __init__(self, base_url: str, state: StateManager):
        self.base_url = base_url
        self.visited = set(state.state.get('visited', []))
        self.processed = set(state.state.get('processed_files', []))
        self.queue = deque()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': CONFIG['USER_AGENT']})
        for target in CONFIG['TARGET_DIRS']:
            self.queue.append((urljoin(base_url, target), 0))
        self.files_seen = state.state.get('files_seen', 0)

    def is_valid_link(self, href: Optional[str]) -> bool:
        return bool(href and not href.startswith('?') and href not in ['/', '../', './'])

    def is_text_candidate(self, href: str) -> bool:
        return href.lower().endswith(CONFIG['TEXT_EXTS'])

    def fetch(self) -> Generator[Tuple[str, str], None, None]:
        while self.queue:
            current_url, depth = self.queue.popleft()
            if current_url in self.visited:
                continue
            self.visited.add(current_url)
            time.sleep(CONFIG['DELAY_SECONDS'])
            try:
                r = self.session.get(current_url, timeout=15)
                if r.status_code != 200:
                    continue
                ctype = r.headers.get('Content-Type', '')
                if 'text/html' in ctype and current_url.endswith('/'):
                    if depth >= CONFIG['MAX_DEPTH']:
                        continue
                    soup = BeautifulSoup(r.text, 'html.parser')
                    for a in soup.find_all('a'):
                        href = a.get('href')
                        if not self.is_valid_link(href):
                            continue
                        full = urljoin(current_url, href)
                        if href.endswith('/'):
                            if full.startswith(self.base_url) and full not in self.visited:
                                self.queue.append((full, depth + 1))
                        elif self.is_text_candidate(href) and full not in self.processed:
                            self.queue.append((full, 999))
                elif depth == 999:
                    self.files_seen += 1
                    yield current_url, r.text
                    self.processed.add(current_url)
            except Exception as e:
                logger.error(f'Error at {current_url}: {e}')


def main():
    state = StateManager(CONFIG['STATE_FILE'])
    writer = DatasetWriter(CONFIG['OUTPUT_DIR'])
    dashboard = Dashboard(CONFIG['OUTPUT_DIR'], writer)
    harvester = Harvester(CONFIG['BASE_URL'], state)

    files_processed = 0
    last_dashboard = time.time()
    last_source = ''

    try:
        for source, content in harvester.fetch():
            last_source = source
            blob = (source + ' ' + content[:10000]).lower()
            if not any(k in blob for k in CONFIG['FOCUS_KEYWORDS']):
                continue
            logger.info(f'Processing {source}')
            for rec in Decoder.decode(source, content):
                writer.write_record(rec)
                for art in ConceptMiner.mine(rec):
                    writer.write_artifact(art, rec)
            files_processed += 1

            if files_processed % CONFIG['CHECKPOINT_EVERY'] == 0:
                state.save(harvester.visited, harvester.processed, writer.stats['records'], harvester.files_seen)
                logger.info('Checkpoint saved')

            if time.time() - last_dashboard >= CONFIG['DASHBOARD_EVERY']:
                dashboard.render({
                    'files_seen': harvester.files_seen,
                    'last_source': last_source,
                    'progress_pct': min(100, (files_processed % 10000) / 100.0)
                })
                last_dashboard = time.time()

    except KeyboardInterrupt:
        logger.info('Interrupted, saving state...')
    finally:
        state.save(harvester.visited, harvester.processed, writer.stats['records'], harvester.files_seen)
        dashboard.render({
            'files_seen': harvester.files_seen,
            'last_source': last_source,
            'progress_pct': 100 if files_processed else 0
        })
        writer.close()
        logger.info('Finished')


if __name__ == '__main__':
    main()
