from pathlib import Path
import json,hashlib,zipfile,shutil,tempfile
root=Path(__file__).resolve().parent
with tempfile.TemporaryFile() as merged:
 for part in json.loads((root/'artifacts/archive_parts.json').read_text()):
  p=root/part['path']; h=hashlib.sha256()
  with p.open('rb') as f:
   while True:
    b=f.read(1024*1024)
    if not b: break
    h.update(b);merged.write(b)
  assert h.hexdigest()==part['sha256'],p
 merged.seek(0)
 target=root/'restored_artifacts';target.mkdir(exist_ok=True)
 with zipfile.ZipFile(merged) as z:
  for entry in z.infolist():
   dest=(target/entry.filename).resolve()
   assert target.resolve() in dest.parents,entry.filename
  z.extractall(target)
print('Artifacts restored; no training performed.')
