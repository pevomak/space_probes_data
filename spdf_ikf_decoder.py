import os
import re
import csv
import json
import math
import time
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any, Tuple
from collections import Counter, defaultdict, deque
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

CONFIG = {
    "BASE_URL": "https://spdf.gsfc.nasa.gov/pub/",
    "TARGET_DIRS": [
        "documents/",
        "catalogs/",
        "software/",
        "models/"
    ],
    "MAX_DEPTH": 5,
    "DELAY_SECONDS": 0.2,
    "USER_AGENT": "SPDF-IKF-Decoder/3.0",
    "ENTROPY_THRESHOLD": 3.2,
    "MIN_CONTENT_LENGTH": 30,
    "MAX_RECORDS": 200000,
    "OUTPUT_DIR": "output",
    "OUTPUT_JSONL": "output/decoded_records.jsonl",
    "OUTPUT_CSV": "output/decoded_records.csv",
    "OUTPUT_SUMMARY": "output/decoded_summary.md",
    "LOG_LEVEL": logging.INFO,
    "FOCUS_KEYWORDS": [
        "voyager", "pioneer", "heliopause", "heliosheath", "interstellar",
        "solar wind", "plasma", "alfven", "alfvén", "mag", "pws", "crs",
        "telemetry", "memory", "dump", "packet", "catalog", "idl", "pro"
    ],
    "TEXT_EXTS": (".txt", ".asc", ".cat", ".lbl", ".cdf", ".htm", ".html", ".md", ".pro", ".sf", ".fmt", ".tab", ".csv", ".dat", ".h", ".c"),
}

logging.basicConfig(level=CONFIG["LOG_LEVEL"], format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("SPDF_IKF_DECODER")


class EntropyGate:
    @staticmethod
    def calculate(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        return -sum((c / length) * math.log(c / length, 2) for c in counts.values())

    @staticmethod
    def is_signal(text: str) -> bool:
        if len(text) < CONFIG["MIN_CONTENT_LENGTH"]:
            return False
        return EntropyGate.calculate(text) > CONFIG["ENTROPY_THRESHOLD"]


class LegacyNumberParser:
    @staticmethod
    def parse_num(s: str) -> Any:
        s = s.strip().rstrip(',;')
        if not s:
            return s
        if re.fullmatch(r'[-+]?0[xX][0-9A-Fa-f]+', s):
            return int(s, 16)
        if re.fullmatch(r'[0-9A-Fa-f]{4,16}', s) and re.search(r'[A-Fa-f]', s):
            return int(s, 16)
        if re.fullmatch(r'0[0-7]+', s) and len(s) > 1:
            try:
                return int(s, 8)
            except Exception:
                pass
        if re.search(r'[DdEe][+-]?\d+', s):
            try:
                return float(s.replace('D', 'E').replace('d', 'e'))
            except Exception:
                pass
        if re.fullmatch(r'[-+]?\d+', s):
            try:
                return int(s)
            except Exception:
                pass
        if re.fullmatch(r'[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[Ee][+-]?\d+)?', s):
            try:
                return float(s)
            except Exception:
                pass
        return s


class FileClassifier:
    @staticmethod
    def classify_url(url: str) -> str:
        ul = url.lower()
        if ul.endswith('.pro'):
            return 'idl_source'
        if any(ul.endswith(x) for x in ('.cat', '.lbl', '.fmt')):
            return 'metadata_table'
        if any(ul.endswith(x) for x in ('.tab', '.csv', '.dat', '.asc')):
            return 'structured_text'
        if any(ul.endswith(x) for x in ('.htm', '.html', '.md', '.txt', '.sf')):
            return 'documentation'
        return 'other'


class MissionTagger:
    @staticmethod
    def tag(text: str, source: str) -> Dict[str, str]:
        blob = f"{source} {text}".lower()
        mission = 'unknown'
        instrument = 'unknown'
        domain = 'general'

        if 'voyager' in blob:
            mission = 'voyager'
        elif 'pioneer' in blob:
            mission = 'pioneer'

        if any(k in blob for k in ['heliopause', 'heliosheath', 'interstellar']):
            domain = 'boundary_plasma'
        elif any(k in blob for k in ['solar wind', 'plasma', 'alfven', 'alfvén', 'turbulence']):
            domain = 'plasma_turbulence'
        elif any(k in blob for k in ['memory', 'telemetry', 'dump', 'packet']):
            domain = 'telemetry_memory'
        elif any(k in blob for k in ['catalog', 'label', 'metadata']):
            domain = 'metadata'

        for inst in ['mag', 'pws', 'crs', 'pls', 'lecp', 'epi', 'fds']:
            if re.search(rf'\b{inst}\b', blob):
                instrument = inst
                break

        return {'mission': mission, 'instrument': instrument, 'domain': domain}


class ContentDecoder:
    @staticmethod
    def html_to_text(content_text: str) -> str:
        if '<html' in content_text.lower():
            soup = BeautifulSoup(content_text, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.extract()
            return soup.get_text('\n')
        return content_text

    @staticmethod
    def decode_idl_source(text: str, source: str) -> Generator[Dict[str, Any], None, None]:
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            raw = line.rstrip()
            if not raw.strip():
                continue
            m = re.match(r'^\s*pro\s+([a-zA-Z0-9_]+)', raw, re.I)
            if m:
                yield {
                    'record_type': 'idl_procedure',
                    'source': source,
                    'line_no': i,
                    'name': m.group(1),
                    'raw': raw
                }
            for var, val in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\n]+)', raw):
                parsed = LegacyNumberParser.parse_num(val.strip())
                yield {
                    'record_type': 'idl_assignment',
                    'source': source,
                    'line_no': i,
                    'field': var,
                    'raw_value': val.strip(),
                    'value': parsed,
                    'raw': raw
                }
            if ';' in raw:
                comment = raw.split(';', 1)[1].strip()
                if len(comment) > 10:
                    yield {
                        'record_type': 'idl_comment',
                        'source': source,
                        'line_no': i,
                        'comment': comment,
                        'raw': raw
                    }

    @staticmethod
    def decode_hex_or_memory(text: str, source: str) -> Generator[Dict[str, Any], None, None]:
        for i, line in enumerate(text.splitlines(), 1):
            raw = line.rstrip()
            m = re.match(r'^\s*([0-9A-Fa-f]{4,8})\s*[: ]\s*((?:[0-9A-Fa-f]{2,8}\s+){1,16})(.*)$', raw)
            if not m:
                continue
            base_addr = int(m.group(1), 16)
            words = m.group(2).split()
            tail = m.group(3).strip()
            for j, word in enumerate(words):
                yield {
                    'record_type': 'memory_word',
                    'source': source,
                    'line_no': i,
                    'address': f"0x{base_addr + j:06X}",
                    'word_index': j,
                    'raw_value': word,
                    'value': LegacyNumberParser.parse_num(word),
                    'comment': tail,
                    'raw': raw
                }

    @staticmethod
    def decode_fixed_width_or_table(text: str, source: str) -> Generator[Dict[str, Any], None, None]:
        lines = text.splitlines()
        header = None
        for i, line in enumerate(lines, 1):
            raw = line.rstrip()
            if not raw.strip():
                continue
            cols = re.split(r'\s{2,}|\t+', raw.strip())
            if len(cols) >= 3 and header is None and not any(re.search(r'\d', c) for c in cols[:2]):
                header = cols
                yield {
                    'record_type': 'table_header',
                    'source': source,
                    'line_no': i,
                    'columns': header,
                    'raw': raw
                }
                continue
            if len(cols) >= 2:
                parsed = [LegacyNumberParser.parse_num(c) for c in cols]
                rec = {
                    'record_type': 'table_row',
                    'source': source,
                    'line_no': i,
                    'columns': cols,
                    'parsed_columns': parsed,
                    'raw': raw
                }
                if header and len(header) == len(cols):
                    rec['mapped'] = {header[k]: parsed[k] for k in range(len(cols))}
                yield rec

    @staticmethod
    def decode_text_blocks(text: str, source: str) -> Generator[Dict[str, Any], None, None]:
        blocks = re.split(r'\n\s*\n', text)
        for idx, block in enumerate(blocks, 1):
            b = block.strip()
            if not EntropyGate.is_signal(b):
                continue
            yield {
                'record_type': 'text_block',
                'source': source,
                'block_no': idx,
                'raw': b
            }

    @staticmethod
    def decode(content_text: str, source: str) -> Generator[Dict[str, Any], None, None]:
        text = ContentDecoder.html_to_text(content_text)
        kind = FileClassifier.classify_url(source)

        yielded = False
        if kind == 'idl_source':
            for rec in ContentDecoder.decode_idl_source(text, source):
                yielded = True
                yield rec

        for rec in ContentDecoder.decode_hex_or_memory(text, source):
            yielded = True
            yield rec

        if kind in ('metadata_table', 'structured_text', 'documentation', 'other'):
            for rec in ContentDecoder.decode_fixed_width_or_table(text, source):
                yielded = True
                yield rec

        if not yielded:
            for rec in ContentDecoder.decode_text_blocks(text, source):
                yield rec


class SPDFHarvester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.visited = set()
        self.queue = deque()
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': CONFIG['USER_AGENT']})
        for target in CONFIG['TARGET_DIRS']:
            self.queue.append((urljoin(base_url, target), 0))

    def is_valid_link(self, href: Optional[str]) -> bool:
        return bool(href and not href.startswith('?') and href not in ['/', '../', './'])

    def is_text_candidate(self, href: str) -> bool:
        return href.lower().endswith(CONFIG['TEXT_EXTS'])

    def fetch_documents(self) -> Generator[Tuple[str, str], None, None]:
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
                        elif self.is_text_candidate(href):
                            self.queue.append((full, 999))
                elif depth == 999:
                    yield current_url, r.text
            except Exception as e:
                logger.error(f"Error at {current_url}: {e}")


class RecordWriter:
    def __init__(self):
        self.jsonl_path = CONFIG['OUTPUT_JSONL']
        self.csv_path = CONFIG['OUTPUT_CSV']
        self.summary_path = CONFIG['OUTPUT_SUMMARY']
        self.fieldnames = [
            'record_id', 'record_type', 'mission', 'instrument', 'domain', 'source',
            'line_no', 'block_no', 'name', 'field', 'address', 'word_index',
            'raw_value', 'value', 'comment', 'columns', 'mapped', 'raw'
        ]
        self.counts = Counter()
        self.sources = Counter()
        self.domains = Counter()
        self.missions = Counter()
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.csv_writer.writeheader()
        self.jsonl_file = open(self.jsonl_path, 'w', encoding='utf-8')

    def write(self, rec: Dict[str, Any]):
        rec = dict(rec)
        rec['record_id'] = hashlib.md5((rec.get('source', '') + rec.get('raw', '') + str(rec.get('line_no', ''))).encode()).hexdigest()[:12]
        tags = MissionTagger.tag(rec.get('raw', ''), rec.get('source', ''))
        rec.update(tags)
        self.counts[rec['record_type']] += 1
        self.sources[rec['source']] += 1
        self.domains[rec['domain']] += 1
        self.missions[rec['mission']] += 1

        self.jsonl_file.write(json.dumps(rec, ensure_ascii=False) + '\n')

        row = {k: rec.get(k) for k in self.fieldnames}
        for key in ('columns', 'mapped', 'value'):
            if isinstance(row.get(key), (list, dict)):
                row[key] = json.dumps(row[key], ensure_ascii=False)
        self.csv_writer.writerow(row)

    def close(self):
        self.csv_file.close()
        self.jsonl_file.close()
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            f.write('# SPDF Decoded Summary\n\n')
            f.write(f'Generated: {datetime.now().isoformat()}\n\n')
            f.write('## Record types\n')
            for k, v in self.counts.most_common():
                f.write(f'- {k}: {v}\n')
            f.write('\n## Missions\n')
            for k, v in self.missions.most_common():
                f.write(f'- {k}: {v}\n')
            f.write('\n## Domains\n')
            for k, v in self.domains.most_common():
                f.write(f'- {k}: {v}\n')
            f.write('\n## Top sources\n')
            for k, v in self.sources.most_common(25):
                f.write(f'- {k}: {v}\n')


def main():
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    harvester = SPDFHarvester(CONFIG['BASE_URL'])
    writer = RecordWriter()
    total = 0
    try:
        for source, content in harvester.fetch_documents():
            src_l = source.lower()
            if not any(k in src_l or k in content.lower()[:5000] for k in CONFIG['FOCUS_KEYWORDS']):
                continue
            logger.info(f'Decoding {source}')
            for rec in ContentDecoder.decode(content, source):
                writer.write(rec)
                total += 1
                if total >= CONFIG['MAX_RECORDS']:
                    logger.info('Reached MAX_RECORDS limit')
                    break
            if total >= CONFIG['MAX_RECORDS']:
                break
    finally:
        writer.close()
    print(f'Decoded records: {total}')
    print(f'JSONL: {CONFIG["OUTPUT_JSONL"]}')
    print(f'CSV: {CONFIG["OUTPUT_CSV"]}')
    print(f'Summary: {CONFIG["OUTPUT_SUMMARY"]}')


if __name__ == '__main__':
    main()
