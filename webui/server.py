"""
server.py  — NOVA Web UI backend (stdlib only, no extra deps)
-------------------------------------------------------------
- เสิร์ฟหน้าเว็บ index.html
- /api/chat : สตรีมคำตอบจากโมเดล (base Qwen + LoRA adapter) แบบ token-by-token
- /api/health : บอกว่าโมเดลโหลดได้ไหม
- ถ้าโหลดโมเดลไม่ได้ (ยังไม่มี GPU/adapter) -> ทำงานแบบ demo อัตโนมัติ

รัน:  .venv/Scripts/python.exe webui/server.py
แล้วเปิด http://localhost:8000
"""
import json, os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE_MODEL = os.environ.get("NOVA_BASE", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER = os.environ.get("NOVA_ADAPTER", os.path.join(ROOT, "outputs", "qwen-ie-engineer-lora"))
PORT = int(os.environ.get("NOVA_PORT", "8000"))

_model = _tok = None
_load_error = None
_lock = threading.Lock()


def try_load():
    """โหลดโมเดลครั้งเดียวแบบ lazy; คืน True ถ้าสำเร็จ"""
    global _model, _tok, _load_error
    if _model is not None or _load_error is not None:
        return _model is not None
    with _lock:
        if _model is not None or _load_error is not None:
            return _model is not None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available (อัปเดตไดรเวอร์ NVIDIA ก่อน)")
            tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, dtype=torch.float16, device_map="auto")
            if os.path.isdir(ADAPTER):
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, ADAPTER)
                print(f"[nova] loaded base + LoRA adapter: {ADAPTER}")
            else:
                print(f"[nova] adapter not found ({ADAPTER}); using base model only")
            model.eval()
            _model, _tok = model, tok
            return True
        except Exception as e:
            _load_error = str(e)
            print(f"[nova] model load failed -> demo mode: {e}")
            return False


def stream_model(messages, write):
    """สตรีมคำตอบจริงจากโมเดล ผ่าน TextIteratorStreamer"""
    import torch
    from transformers import TextIteratorStreamer
    ids = _tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(_model.device)
    streamer = TextIteratorStreamer(_tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(input_ids=ids, max_new_tokens=512, do_sample=False, streamer=streamer)
    threading.Thread(target=_model.generate, kwargs=kw).start()
    for chunk in streamer:
        write(chunk)


DEMO = ("🛰️ *(Demo mode — โมเดลยังไม่พร้อม)*\n\n"
        "สาเหตุ: {err}\n\nเมื่ออัปเดตไดรเวอร์ NVIDIA และเทรนโมเดลเสร็จ "
        "เซิร์ฟเวอร์จะใช้โมเดลจริงอัตโนมัติครับ")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # เงียบ log
        pass

    def _send(self, code, ctype, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, "text/html; charset=utf-8", f.read())
        elif self.path == "/api/health":
            loaded = try_load()
            self._send(200, "application/json",
                       json.dumps({"model_loaded": loaded, "error": _load_error}).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/api/chat":
            return self._send(404, "text/plain", b"not found")
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        messages = data.get("messages", [])

        # โมเดลยังไม่พร้อม -> ส่ง 503 ให้ frontend ใช้ demo ที่สวยกว่าเอง
        if not try_load():
            self._send(503, "text/plain; charset=utf-8",
                       DEMO.format(err=_load_error or "unknown").encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def write(s):
            try:
                self.wfile.write(s.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

        try:
            stream_model(messages, write)
        except Exception as e:
            write(f"\n[error] {e}")


if __name__ == "__main__":
    print(f"[nova] serving on http://localhost:{PORT}  (model: {BASE_MODEL})")
    print("[nova] checking model availability…")
    try_load()
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
