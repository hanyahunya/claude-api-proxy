# claude-cli → Anthropic API 프록시 (llm_proxy_temp)

구독(Pro/Max)으로 `claude login` 된 머신에서, **Anthropic API 키로 호출하는 것처럼** 표준
Anthropic 엔드포인트를 제공한다. 내부적으로는 `claude -p`(Claude Code CLI)를 subprocess 로
호출한다. 클라이언트는 `base_url` 만 이 프록시로 바꾸면 코드 변경 없이 구독 인증을 쓴다.

```
클라이언트(Anthropic SDK / 기존 LLM 서버 / LangChain)
        │  HTTP (Anthropic 형식)
        ▼
   이 프록시 (FastAPI)
        │  subprocess: claude -p
        ▼
   Claude (구독 인증, ~/.claude)
```

> 기존 프로젝트 코드는 일절 변경하지 않는다. 이 폴더만으로 독립 실행된다.

## 설치 & 실행

전제: `claude` CLI 설치 + `claude login`(구독) 또는 `ANTHROPIC_API_KEY` 인증 완료.

**pipx (권장 — 한 줄 설치):**
```bash
pipx install git+https://github.com/hanyahunya/claude-api-proxy
claude-api-proxy            # 기본 127.0.0.1:8787 (PORT 환경변수로 변경)
```

**소스에서 직접:**
```bash
git clone https://github.com/hanyahunya/claude-api-proxy
cd claude-api-proxy
pip install -r requirements.txt
python server.py
```

**Windows 원클릭:** `run.bat` (venv 자동 생성·설치 후 실행)

## 클라이언트 예시 (공식 SDK — 그대로 동작 검증됨)

```python
from anthropic import Anthropic
c = Anthropic(api_key="anything", base_url="http://127.0.0.1:8787")
m = c.messages.create(model="claude-sonnet-4-6", max_tokens=100,
                      messages=[{"role": "user", "content": "안녕"}])
print(m.content[0].text)

# 스트리밍도 동작
with c.messages.stream(model="claude-sonnet-4-6", max_tokens=100,
                       messages=[{"role": "user", "content": "1부터 5까지"}]) as s:
    for t in s.text_stream:
        print(t, end="")
```

OpenAI/JS/curl 등 base_url 만 바꾸면 동일하게 사용 가능.

## 지원 엔드포인트

| 메서드 | 경로 | 비고 |
|---|---|---|
| POST | `/v1/messages` | Messages API. `stream:true` 시 네이티브 SSE 이벤트 중계 |
| POST | `/v1/messages/count_tokens` | 토큰 카운트 (**근사치**, 4자≈1토큰) |
| GET | `/v1/models`, `/v1/models/{id}` | 모델 목록 |
| POST | `/v1/complete` | 레거시 Text Completions |
| GET | `/healthz` | 헬스체크 |

### 매핑되는 기능
- `system`(문자열/블록 배열) → `--append-system-prompt`
- `messages`(다중 턴) → 단일 프롬프트로 접어 전달(마지막 user 가 본문, 앞은 "이전 대화")
- 이미지 블록(`source.type=base64`) → 임시 PNG 저장 후 Read 도구로 노출
- `stream` → `--output-format stream-json` 의 `stream_event` 를 그대로 SSE 중계
- 모델 별칭(`claude-3-5-sonnet-latest` 등) → 현행 모델 매핑, 미스는 통과
- 모델 `@effort` 접미사(`claude-opus-4-8@max`) → `--effort`
- `stop_sequences` → 비스트리밍 응답에서 후처리로 잘라줌
- `usage.input_tokens/output_tokens`(+캐시 토큰) → 응답에 그대로 전달
- 인증: `PROXY_API_KEY` 설정 시 `x-api-key`/`Authorization: Bearer` 검증

## 한계 (CLI 표면의 구조적 제약)
- **tools / tool_choice (클라이언트 정의 도구)**: CLI 가 도구를 *자체 실행*하는 구조라
  클라이언트 쪽 도구 호출(tool_use 블록 왕복)은 전달되지 않는다. 정의는 무시됨.
  단, `tools` 에 `web_search` 타입이 있으면 CLI 의 WebSearch 를 허용한다.
- **temperature / top_p / top_k / max_tokens**: CLI 에 해당 제어가 없어 무시(호환용 수용).
- **count_tokens**: 정확 카운트 API 가 없어 글자수 기반 근사치.
- 첫 호출은 CLI 가 시스템/캐시를 구성하느라 지연이 있을 수 있음(이후 캐시로 빨라짐).

## 설정 (환경변수)
`.env.example` 참고: `PORT`, `PROXY_API_KEY`, `CLAUDE_BIN`, `CLI_TIMEOUT_SEC`,
`PROXY_CLI_CWD`, `LOG_LEVEL`.
