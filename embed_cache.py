# embed_cache.py
import os, sqlite3, hashlib, json
os.makedirs("cache", exist_ok=True)
DB = os.path.join("cache", "embeddings.sqlite")

_conn = sqlite3.connect(DB)
_conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
_conn.execute("PRAGMA journal_mode=WAL;")
_conn.execute("PRAGMA synchronous=NORMAL;")

def _key(text: str, model: str) -> str:
    # 텍스트 전처리(공백 정규화) + 모델명 포함 키
    norm = " ".join(text.strip().split())
    h = hashlib.sha256((model + "||" + norm).encode("utf-8")).hexdigest()
    return h

def get_cached(text: str, model: str) -> list | None:
    k = _key(text, model)
    row = _conn.execute("SELECT v FROM emb WHERE k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else None

def set_cached(text: str, model: str, vec: list[float]) -> None:
    k = _key(text, model)
    _conn.execute("REPLACE INTO emb (k, v) VALUES (?, ?)", (k, json.dumps(vec)))
    _conn.commit()
