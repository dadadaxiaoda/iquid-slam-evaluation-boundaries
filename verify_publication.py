from pathlib import Path
import ast,json,hashlib
root=Path(__file__).resolve().parent
for p in root.rglob('*.py'): ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
for p in root.rglob('*.json'): json.loads(p.read_text(encoding='utf-8'))
for part in json.loads((root/'artifacts/archive_parts.json').read_text()):
 p=root/part['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 assert h.hexdigest()==part['sha256'],p
print('Python syntax, JSON and all archive part hashes verified.')
