"""Anthropic Messages API 호환 프록시 — 내부적으로 `claude -p`(Claude Code CLI)를 호출한다.

목적
  구독(Pro/Max)으로 `claude login` 된 머신에서, 마치 Anthropic API 키로 호출하는 것처럼
  표준 Anthropic 엔드포인트(/v1/messages 등)를 제공한다. 클라이언트(기존 LLM 서버, SDK,
  LangChain 등)는 base_url 만 이 프록시로 바꾸면 코드 변경 없이 구독 인증을 그대로 쓴다.

  클라이언트 ── HTTP(Anthropic 형식) ──▶ 이 프록시 ── subprocess ──▶ claude -p ──▶ Claude(구독)

핵심 발견(이 구현의 근거)
  - `claude -p --output-format json` 은 result / stop_reason / usage(input/output tokens) /
    session_id 를 그대로 준다 → 비스트리밍 응답을 그대로 매핑.
  - `claude -p --output-format stream-json --verbose --include-partial-messages` 의
    {"type":"stream_event","event":{...}} 라인은 **이미 네이티브 Anthropic 스트리밍 이벤트**다
    (message_start / content_block_start / content_block_delta / ... / message_stop).
    → event 만 꺼내 SSE 로 그대로 흘려보낸다.

지원 엔드포인트
  POST /v1/messages                 Messages API (stream / non-stream)
  POST /v1/messages/count_tokens    토큰 카운트(근사치)
  GET  /v1/models, /v1/models/{id}  모델 목록
  POST /v1/complete                 (레거시) Text Completions → 내부적으로 messages 로 변환
  GET  /healthz                     헬스체크

미지원/주의 (CLI 표면의 한계 — 무시하거나 근사 처리)
  - tools / tool_choice : CLI 가 자체적으로 도구를 실행하는 구조라 '클라이언트 정의 도구'
    호출은 전달되지 않는다. 정의는 무시한다(모델은 텍스트로만 답).
  - temperature / top_p / top_k : CLI 에 해당 플래그 없음 → 무시.
  - max_tokens : CLI 가 자체 관리 → 무시(인터페이스 호환용으로만 받음).
  - stop_sequences : 비스트리밍 응답에서 후처리로 잘라준다(스트리밍은 미적용).

기존 프로젝트 코드는 일절 건드리지 않는다 — 이 폴더(llm_proxy_temp)만으로 독립 실행된다.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("llm_proxy")

# ── 설정 (환경변수) ───────────────────────────────────────────────────────────
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY")            # 설정 시 x-api-key/Authorization 검증
CLI_TIMEOUT = float(os.environ.get("CLI_TIMEOUT_SEC", "600"))
# claude -p 를 실행할 작업 디렉터리. 거래 프로젝트의 CLAUDE.md/스킬을 매 호출 로드하지 않도록
# 중립 임시 디렉터리에서 돌린다(인증은 ~/.claude 전역이라 영향 없음).
WORKDIR = os.environ.get("PROXY_CLI_CWD") or tempfile.mkdtemp(prefix="llm-proxy-cwd-")

# 잘 알려진 Anthropic 모델 별칭 → CLI 가 아는 현행 모델로 매핑. 미지정/미스는 그대로 통과
# (CLI 가 미지원 모델은 폴백 처리). "@effort" 접미사(claude-opus-4-8@max)는 그대로 보존.
MODEL_ALIASES = {
    "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
    "claude-3-7-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-opus-latest": "claude-opus-4-8",
    "claude-3-5-haiku-latest": "claude-haiku-4-5-20251001",
    "claude-3-haiku-20240307": "claude-haiku-4-5-20251001",
}
KNOWN_MODELS = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",
]

app = FastAPI(title="claude-cli-anthropic-proxy", version="0.1.0")


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _check_auth(x_api_key: str | None, authorization: str | None) -> None:
    if not PROXY_API_KEY:
        return  # 검증 비활성(아무 키나 허용) — 드롭인 편의용
    supplied = x_api_key
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:]
    if supplied != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail={"type": "authentication_error",
                                                     "message": "invalid api key"})


def _resolve_bin() -> str:
    exe = shutil.which(CLAUDE_BIN)
    if exe is None:
        raise HTTPException(status_code=500, detail={
            "type": "api_error",
            "message": f"'{CLAUDE_BIN}' 실행파일을 찾을 수 없습니다. Claude Code CLI 설치 후 "
                       "`claude login`(구독) 또는 ANTHROPIC_API_KEY 인증이 필요합니다."})
    return exe


def _text_from_content(content) -> str:
    """Anthropic content(문자열 | 블록 배열)에서 텍스트만 추출."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


def _extract_images(content) -> list[tuple[str, str]]:
    """content 블록에서 (media_type, base64) 이미지들을 추출. (base64 source 만 지원)"""
    out: list[tuple[str, str]] = []
    if not isinstance(content, list):
        return out
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            src = block.get("source", {})
            if src.get("type") == "base64":
                out.append((src.get("media_type", "image/png"), src.get("data", "")))
    return out


def _build_prompt(messages: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    """Anthropic messages 배열을 claude -p 용 단일 프롬프트로 접는다.

    마지막 user 메시지를 본문으로, 그 앞 대화는 (이전 대화) 블록으로 직렬화한다.
    이미지는 마지막 user 메시지 것만 수집(임시파일→Read 도구로 노출)."""
    if not messages:
        return "", []
    last = messages[-1]
    body = _text_from_content(last.get("content"))
    images = _extract_images(last.get("content"))
    prior = messages[:-1]
    if prior:
        folded = "\n\n".join(
            f"[{m.get('role', 'user')}]\n{_text_from_content(m.get('content'))}" for m in prior)
        body = f"(이전 대화)\n{folded}\n\n(새 메시지)\n{body}"
    return body, images


def _map_model(model: str | None) -> str:
    if not model:
        return "claude-sonnet-4-6"
    base, sep, effort = model.partition("@")
    base = MODEL_ALIASES.get(base, base)
    return f"{base}@{effort}" if sep else base


def _build_argv(exe: str, model: str, system: str | None, *, stream: bool,
                allow_read: bool, web_search: bool) -> list[str]:
    base, _, effort = _map_model(model).partition("@")
    if stream:
        argv = [exe, "-p", "--output-format", "stream-json", "--verbose",
                "--include-partial-messages", "--model", base]
    else:
        argv = [exe, "-p", "--output-format", "json", "--model", base]
    if effort.strip():
        argv += ["--effort", effort.strip().lower()]
    allowed = []
    if allow_read:
        allowed.append("Read")
    if web_search:
        allowed.append("WebSearch")
    if allowed:
        argv += ["--allowedTools", ",".join(allowed)]
    if system:
        argv += ["--append-system-prompt", system]
    return argv


def _write_temp_images(images: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """이미지를 임시 PNG 로 저장하고, 프롬프트에 붙일 안내 문구를 만든다."""
    paths: list[str] = []
    for _mt, b64 in images:
        if not b64:
            continue
        fd, p = tempfile.mkstemp(suffix=".png", prefix="proxy-img-")
        with os.fdopen(fd, "wb") as fp:
            fp.write(base64.b64decode(b64))
        paths.append(p)
    note = ""
    if paths:
        listing = "\n".join(f"- {p}" for p in paths)
        note = f"[첨부 이미지 — 반드시 Read 도구로 직접 열어 확인하라]\n{listing}\n\n"
    return note, paths


# ── 엔드포인트 ───────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "bin": shutil.which(CLAUDE_BIN), "cwd": WORKDIR}


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {"data": [{"type": "model", "id": m, "display_name": m, "created_at": now}
                     for m in KNOWN_MODELS], "has_more": False,
            "first_id": KNOWN_MODELS[0], "last_id": KNOWN_MODELS[-1]}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    return {"type": "model", "id": model_id, "display_name": model_id,
            "created_at": int(time.time())}


@app.post("/v1/messages/count_tokens")
async def count_tokens(req: Request):
    body = await req.json()
    text = ""
    sys = body.get("system")
    if sys:
        text += _text_from_content(sys)
    for m in body.get("messages", []):
        text += "\n" + _text_from_content(m.get("content"))
    # 근사치: 대략 4자 ≈ 1토큰 (정확 카운트는 CLI 표면 미제공)
    return {"input_tokens": max(1, len(text) // 4)}


@app.post("/v1/messages")
async def messages(req: Request,
                   x_api_key: str | None = Header(default=None, alias="x-api-key"),
                   authorization: str | None = Header(default=None)):
    _check_auth(x_api_key, authorization)
    exe = _resolve_bin()
    body = await req.json()

    model = body.get("model")
    system = _text_from_content(body.get("system")) or None
    stream = bool(body.get("stream"))
    msgs = body.get("messages", [])
    web_search = _wants_web_search(body)

    prompt, images = _build_prompt(msgs)
    img_note, tmp_paths = _write_temp_images(images)
    prompt = img_note + prompt

    argv = _build_argv(exe, model or "", system, stream=stream,
                       allow_read=bool(tmp_paths), web_search=web_search)
    resolved_model = _map_model(model).partition("@")[0]

    if stream:
        return StreamingResponse(
            _stream_response(argv, prompt, tmp_paths, resolved_model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
    return await _json_response(argv, prompt, tmp_paths, resolved_model,
                                body.get("stop_sequences"))


def _wants_web_search(body: dict) -> bool:
    for t in body.get("tools", []) or []:
        if isinstance(t, dict) and str(t.get("type", "")).startswith("web_search"):
            return True
    return False


async def _run_cli(argv: list[str], prompt: str):
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=WORKDIR,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    return proc


def _cleanup(paths: list[str]):
    for p in paths:
        with suppress(OSError):
            os.unlink(p)


def _map_stop_reason(r: str | None) -> str | None:
    return r or "end_turn"


async def _json_response(argv, prompt, tmp_paths, model, stop_sequences):
    try:
        proc = await _run_cli(argv, prompt)
        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")), timeout=CLI_TIMEOUT)
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                proc.kill()
            raise HTTPException(status_code=504, detail={
                "type": "api_error", "message": f"claude CLI timeout ({CLI_TIMEOUT}s)"})
    finally:
        _cleanup(tmp_paths)

    if proc.returncode != 0:
        msg = (err_b or b"").decode("utf-8", "replace")[:500]
        raise HTTPException(status_code=502, detail={
            "type": "api_error", "message": f"claude CLI error: {msg}"})
    raw = (out_b or b"").decode("utf-8", "replace").strip()
    if not raw:
        raise HTTPException(status_code=502, detail={
            "type": "api_error", "message": "claude CLI empty output (auth/limit?)"})
    data = json.loads(raw)
    if data.get("is_error"):
        raise HTTPException(status_code=502, detail={
            "type": "api_error", "message": str(data)[:500]})

    text = (data.get("result") or "").strip()
    stop_reason = _map_stop_reason(data.get("stop_reason"))
    matched_stop = None
    for s in (stop_sequences or []):
        idx = text.find(s)
        if idx != -1:
            text = text[:idx]
            stop_reason, matched_stop = "stop_sequence", s
            break

    u = data.get("usage", {}) or {}
    return JSONResponse({
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "usage": {
            "input_tokens": u.get("input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
        },
    })


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def _stream_response(argv, prompt, tmp_paths, model):
    """stream-json 의 stream_event(=네이티브 Anthropic 이벤트)를 그대로 SSE 로 중계한다."""
    proc = await _run_cli(argv, prompt)
    if proc.stdin:
        proc.stdin.write(prompt.encode("utf-8"))
        with suppress(Exception):
            proc.stdin.close()

    saw_message_stop = False
    saw_message_start = False
    try:
        assert proc.stdout
        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=CLI_TIMEOUT)
            except asyncio.TimeoutError:
                break
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "stream_event":
                ev = obj.get("event") or {}
                etype = ev.get("type")
                if not etype:
                    continue
                if etype == "message_start":
                    saw_message_start = True
                if etype == "message_stop":
                    saw_message_stop = True
                yield _sse(etype, ev)
            elif obj.get("type") == "result" and obj.get("is_error"):
                # 오류 결과 → error 이벤트
                yield _sse("error", {"type": "error", "error": {
                    "type": "api_error", "message": str(obj)[:500]}})
        # CLI 가 message_stop 을 안 줬다면 보강(클라이언트 종료 신호 보장)
        if saw_message_start and not saw_message_stop:
            yield _sse("message_delta", {"type": "message_delta",
                                         "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                         "usage": {"output_tokens": 0}})
            yield _sse("message_stop", {"type": "message_stop"})
        if not saw_message_start:
            err = (await proc.stderr.read()).decode("utf-8", "replace")[:500] if proc.stderr else ""
            yield _sse("error", {"type": "error", "error": {
                "type": "api_error", "message": f"claude CLI produced no events: {err}"}})
    finally:
        with suppress(ProcessLookupError):
            if proc.returncode is None:
                proc.kill()
        with suppress(Exception):
            await proc.wait()
        _cleanup(tmp_paths)


# ── 레거시 Text Completions (/v1/complete) ───────────────────────────────────
@app.post("/v1/complete")
async def complete(req: Request,
                   x_api_key: str | None = Header(default=None, alias="x-api-key"),
                   authorization: str | None = Header(default=None)):
    _check_auth(x_api_key, authorization)
    exe = _resolve_bin()
    body = await req.json()
    # "\n\nHuman: ... \n\nAssistant:" 프롬프트 → 단순히 본문으로 전달
    prompt = body.get("prompt", "")
    argv = _build_argv(exe, body.get("model") or "", None, stream=False,
                       allow_read=False, web_search=False)
    try:
        proc = await _run_cli(argv, prompt)
        out_b, err_b = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")), timeout=CLI_TIMEOUT)
    except asyncio.TimeoutError:
        with suppress(ProcessLookupError):
            proc.kill()
        raise HTTPException(status_code=504, detail={"type": "api_error", "message": "timeout"})
    if proc.returncode != 0:
        raise HTTPException(status_code=502, detail={
            "type": "api_error", "message": (err_b or b"").decode("utf-8", "replace")[:500]})
    data = json.loads((out_b or b"").decode("utf-8", "replace").strip())
    return {
        "type": "completion",
        "id": "compl_" + uuid.uuid4().hex[:24],
        "completion": (data.get("result") or ""),
        "stop_reason": _map_stop_reason(data.get("stop_reason")),
        "model": _map_model(body.get("model")).partition("@")[0],
    }


def _find_free_port(host: str, desired: int, *, span: int = 50) -> int:
    """desired 포트가 비어있으면 그대로, 막혀있으면 위로 올라가며 빈 포트를 찾는다."""
    import socket
    for port in range(desired, desired + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"{desired}~{desired + span} 사이에 빈 포트가 없습니다.")


def main() -> None:
    """콘솔 엔트리포인트 (`claude-api-proxy`)."""
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    # 기본 8799 (8787 은 headroom 등 다른 OpenAI/Anthropic 프록시가 흔히 쓰므로 피함).
    desired = int(os.environ.get("PORT", "8799"))
    port = _find_free_port(host, desired)
    if port != desired:
        log.warning("포트 %s 가 사용 중 → %s 로 대체합니다.", desired, port)
    base_url = f"http://{host}:{port}"
    banner = (
        "\n" + "=" * 60 + "\n"
        f"  claude-api-proxy 가동:  {base_url}\n"
        "  클라이언트(기존 서버)에서 코드 수정 없이 아래만 설정하세요:\n"
        f"    set  ANTHROPIC_BASE_URL={base_url}        (Windows CMD)\n"
        f"    $env:ANTHROPIC_BASE_URL='{base_url}'      (PowerShell)\n"
        f"    export ANTHROPIC_BASE_URL={base_url}      (bash)\n"
        "    ANTHROPIC_API_KEY 는 아무 값이나 무방 (프록시가 구독 인증 사용)\n"
        + "=" * 60 + "\n"
    )
    print(banner, flush=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
