"""Display routes — REST endpoints, WebSocket, and face HTML page."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from auth import KeyStore, check_ws_auth
from display_state import DisplayBroadcaster, VALID_EXPRESSIONS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["display"])


# -- Request models ----------------------------------------------------------

class TextRequest(BaseModel):
    text: str
    size: str = Field(default="large", pattern="^(small|medium|large)$")


class FaceRequest(BaseModel):
    expression: str


class ImageRequest(BaseModel):
    image_b64: str
    mime_type: str = "image/png"


# -- Face HTML ---------------------------------------------------------------

FACE_HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>🤖 Robot Face</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100vw; height: 100vh; overflow: hidden; background: #1a1a2e; color: #eee;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    cursor: none; user-select: none; display: flex; flex-direction: column; align-items: center;
    transition: background 0.8s ease; }

  /* Status-based background colors */
  body.bg-idle { background: #1a1a2e; }
  body.bg-executing { background: #0d253a; }
  body.bg-rewinding { background: #2e1f0a; }
  body.bg-error { background: #2e0a0a; }

  /* Top status bar */
  .status-bar { width: 100%; padding: 18px 24px; display: flex; align-items: center; justify-content: center;
    gap: 16px; flex-shrink: 0; z-index: 20; }
  .status-label { font-size: 28px; color: #888; transition: all 0.4s ease;
    text-transform: capitalize; letter-spacing: 1px; }
  .status-label.idle { color: #4caf50; }
  .status-label.executing { color: #64b5f6; }
  .status-label.rewinding { color: #ffb74d; }
  .status-label.error { color: #ef5350; }
  .queue-badge { font-size: 16px; color: #999; background: rgba(255,255,255,0.08); padding: 4px 14px;
    border-radius: 12px; transition: opacity 0.4s ease; }
  .queue-badge.hidden { opacity: 0; pointer-events: none; }

  /* Face container */
  .face-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center;
    transition: all 0.5s ease; flex: 1; width: 100%; }
  .face-wrap.corner { position: fixed; top: 20px; right: 20px; width: 120px; height: 120px; flex: none; z-index: 10; }

  /* Face */
  .face { position: relative; width: 420px; height: 420px; transition: all 0.5s ease; }
  .corner .face { width: 100px; height: 100px; }

  /* Eyes */
  .eye { position: absolute; top: 35%; width: 22%; height: 26%; background: #eee; border-radius: 50%;
    transition: all 0.4s ease; }
  .eye-left { left: 22%; }
  .eye-right { right: 22%; }
  .eye .pupil { position: absolute; bottom: 20%; left: 50%; transform: translateX(-50%);
    width: 45%; height: 45%; background: #1a1a2e; border-radius: 50%; transition: all 0.4s ease; }

  /* Mouth */
  .mouth { position: absolute; bottom: 20%; left: 50%; transform: translateX(-50%);
    width: 40%; height: 18%; transition: all 0.4s ease; }

  /* Expressions */
  /* happy — big smile arc, squinty happy eyes, green */
  .face.happy .mouth { border-bottom: 6px solid #eee; border-radius: 0 0 50% 50%; height: 22%; }
  .face.happy .eye { height: 18%; top: 38%; border-radius: 50% 50% 50% 50% / 20% 20% 80% 80%; background: #4caf50; }
  .face.happy .eye .pupil { bottom: 15%; }

  /* thinking — small o mouth, eyes look up-right, blue */
  .face.thinking .mouth { width: 16%; height: 16%; border: 4px solid #eee; border-radius: 50%; }
  .face.thinking .eye { background: #64b5f6; }
  .face.thinking .eye .pupil { bottom: 60%; left: 65%; }
  .face.thinking .eye-left { height: 22%; }
  .face.thinking .eye-right { height: 22%; }

  /* sad — frown arc, red */
  .face.sad .mouth { border-top: 6px solid #eee; border-radius: 50% 50% 0 0; height: 14%; bottom: 24%; }
  .face.sad .eye { height: 20%; top: 38%; background: #ef5350; }
  .face.sad .eye .pupil { bottom: 10%; }

  /* neutral — straight line mouth */
  .face.neutral .mouth { height: 0; border-bottom: 4px solid #aaa; border-radius: 0; width: 30%; }
  .face.neutral .eye { height: 22%; }

  /* excited — big open mouth, wide eyes */
  .face.excited .mouth { width: 30%; height: 24%; border: 5px solid #eee; border-radius: 50%;
    background: rgba(255,255,255,0.05); }
  .face.excited .eye { height: 30%; width: 24%; }
  .face.excited .eye .pupil { width: 40%; height: 40%; }

  /* concerned — tight mouth, circly spiraling eyes */
  .face.concerned .mouth { height: 0; border-bottom: 5px solid #ffb74d; border-radius: 0; width: 25%; }
  .face.concerned .eye { background: transparent; border: none; animation: none; overflow: visible; }
  .face.concerned .eye .pupil { width: 50%; height: 50%; background: #ffb74d; border-radius: 50%;
    animation: orbit 1.2s linear infinite; }
  .face.concerned .eye::before, .face.concerned .eye::after {
    content: ''; position: absolute; border-radius: 50%; border: 3px solid #ffb74d; background: transparent;
    top: 50%; left: 50%; transform: translate(-50%, -50%); animation: spin-back 2s linear infinite; }
  .face.concerned .eye::before { width: 90%; height: 90%; }
  .face.concerned .eye::after { width: 55%; height: 55%; animation-direction: reverse; animation-duration: 1.4s; }
  @keyframes orbit { from { transform: rotate(0deg) translateX(35%) rotate(0deg); }
    to { transform: rotate(360deg) translateX(35%) rotate(-360deg); } }
  @keyframes spin-back { from { transform: translate(-50%, -50%) rotate(0deg); }
    to { transform: translate(-50%, -50%) rotate(-360deg); } }

  /* Blink animation */
  @keyframes blink {
    0%, 94%, 100% { transform: scaleY(1); }
    96% { transform: scaleY(0.05); }
  }
  .eye { animation: blink 4s infinite; animation-delay: var(--blink-delay, 0s); }
  .eye-right { --blink-delay: 0.15s; }

  .corner + .content-area { padding-top: 0; }

  /* Content area (text/image pushed by SDK) */
  .content-area { display: none; flex: 1; width: 100%; align-items: center; justify-content: center;
    padding: 40px; overflow: hidden; }
  .content-area.active { display: flex; }

  .content-text { font-size: 48px; color: #eee; text-align: center; max-width: 90%;
    line-height: 1.3; word-wrap: break-word; }
  .content-text.small { font-size: 28px; }
  .content-text.medium { font-size: 38px; }
  .content-text.large { font-size: 48px; }

  .content-image { max-width: 90%; max-height: 80vh; object-fit: contain; border-radius: 12px; }

  /* Connection indicator */
  .conn-dot { position: fixed; top: 12px; left: 12px; width: 10px; height: 10px;
    border-radius: 50%; background: #4caf50; transition: background 0.3s; z-index: 100; }
  .conn-dot.disconnected { background: #f44336; }

  /* Mute toggle */
  .mute-btn { position: fixed; bottom: 12px; left: 12px; width: 36px; height: 36px;
    background: rgba(255,255,255,0.1); border: none; border-radius: 50%; color: #aaa;
    font-size: 18px; cursor: pointer; z-index: 100; display: flex; align-items: center;
    justify-content: center; transition: background 0.3s, color 0.3s; }
  .mute-btn:hover { background: rgba(255,255,255,0.2); color: #eee; }
  .mute-btn.muted { color: #f44336; }
</style>
</head>
<body>

<div class="conn-dot" id="connDot"></div>
<button class="mute-btn" id="muteBtn" title="Toggle voice">&#x1f50a;</button>

<div class="status-bar">
  <div class="status-label idle" id="statusLabel">Idle</div>
  <div class="queue-badge hidden" id="queue"></div>
</div>

<div class="face-wrap" id="faceWrap">
  <div class="face happy" id="face">
    <div class="eye eye-left"><div class="pupil"></div></div>
    <div class="eye eye-right"><div class="pupil"></div></div>
    <div class="mouth"></div>
  </div>
</div>

<div class="content-area" id="contentArea">
  <div class="content-text large" id="contentText" style="display:none;"></div>
  <img class="content-image" id="contentImage" style="display:none;" />
</div>

<script>
(function() {
  const face = document.getElementById('face');
  const faceWrap = document.getElementById('faceWrap');
  const statusLabel = document.getElementById('statusLabel');
  const contentArea = document.getElementById('contentArea');
  const contentText = document.getElementById('contentText');
  const contentImage = document.getElementById('contentImage');
  const queueEl = document.getElementById('queue');
  const connDot = document.getElementById('connDot');

  const muteBtn = document.getElementById('muteBtn');

  let ws = null;
  let reconnectTimer = null;
  let prevStatus = null;
  let isFirstMessage = true;
  let muted = localStorage.getItem('faceMuted') === 'true';

  // Apply initial mute state
  if (muted) { muteBtn.classList.add('muted'); muteBtn.innerHTML = '&#x1f507;'; }

  muteBtn.addEventListener('click', function() {
    muted = !muted;
    localStorage.setItem('faceMuted', muted);
    muteBtn.classList.toggle('muted', muted);
    muteBtn.innerHTML = muted ? '&#x1f507;' : '&#x1f50a;';
  });

  function speak(text) {
    if (muted || typeof speechSynthesis === 'undefined') return;
    speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    u.pitch = 1.0;
    speechSynthesis.speak(u);
  }

  const STATUS_LABELS = {
    idle: 'Idle',
    executing: 'Executing...',
    rewinding: 'Rewinding...',
    error: 'Error',
  };

  function setFace(expression) {
    face.className = 'face ' + expression;
  }

  var STATUS_VOICE = {
    executing: 'Executing code',
    rewinding: 'Rewinding',
    error: 'Error',
    idle: 'Ready',
  };

  function setStatus(status, queueLen, holder) {
    statusLabel.textContent = STATUS_LABELS[status] || status;
    statusLabel.className = 'status-label ' + status;

    // Background color per status
    document.body.className = document.body.className.replace(/bg-\S+/g, '').trim();
    document.body.classList.add('bg-' + (status || 'idle'));

    // Voice announcement on status transition (skip initial snapshot)
    if (!isFirstMessage && status !== prevStatus && STATUS_VOICE[status]) {
      // Skip "Ready" if coming from null (initial)
      if (!(status === 'idle' && prevStatus === null)) {
        speak(STATUS_VOICE[status]);
      }
    }
    prevStatus = status;

    if (queueLen > 0) {
      queueEl.textContent = queueLen + (queueLen === 1 ? ' person waiting' : ' people waiting');
      queueEl.classList.remove('hidden');
    } else {
      queueEl.classList.add('hidden');
    }
  }

  function showContent(hasContent) {
    if (hasContent) {
      faceWrap.classList.add('corner');
      contentArea.classList.add('active');
    } else {
      faceWrap.classList.remove('corner');
      contentArea.classList.remove('active');
      contentText.style.display = 'none';
      contentImage.style.display = 'none';
    }
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case 'snapshot':
        isFirstMessage = true;
        setFace(msg.face);
        setStatus(msg.robot_status, msg.queue_length, msg.current_holder);
        isFirstMessage = false;
        if (msg.text) {
          contentText.textContent = msg.text;
          contentText.className = 'content-text ' + (msg.text_size || 'large');
          contentText.style.display = '';
          contentImage.style.display = 'none';
          showContent(true);
        } else if (msg.image_b64) {
          contentImage.src = 'data:' + (msg.image_mime || 'image/png') + ';base64,' + msg.image_b64;
          contentImage.style.display = '';
          contentText.style.display = 'none';
          showContent(true);
        } else {
          showContent(false);
        }
        break;

      case 'face':
        setFace(msg.face);
        break;

      case 'text':
        contentText.textContent = msg.text;
        contentText.className = 'content-text ' + (msg.text_size || 'large');
        contentText.style.display = '';
        contentImage.style.display = 'none';
        showContent(true);
        break;

      case 'image':
        contentImage.src = 'data:' + (msg.image_mime || 'image/png') + ';base64,' + msg.image_b64;
        contentImage.style.display = '';
        contentText.style.display = 'none';
        showContent(true);
        break;

      case 'status':
        setFace(msg.face);
        setStatus(msg.robot_status, msg.queue_length, msg.current_holder);
        break;

      case 'clear':
        showContent(false);
        setFace('happy');
        break;

      case 'announce':
        speak(msg.text);
        break;
    }
  }

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws/display');

    ws.onopen = function() {
      connDot.classList.remove('disconnected');
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    };

    ws.onmessage = function(e) {
      try { handleMessage(JSON.parse(e.data)); } catch(err) { console.error('ws parse error', err); }
    };

    ws.onclose = function() {
      connDot.classList.add('disconnected');
      reconnectTimer = setTimeout(connect, 2000);
    };

    ws.onerror = function() {
      ws.close();
    };
  }

  connect();
})();
</script>
</body></html>
"""


# -- Router factory ----------------------------------------------------------

def create_router(display: DisplayBroadcaster, key_store: KeyStore | None = None) -> APIRouter:
    """Create display router with REST, WebSocket, and HTML endpoints."""

    @router.get("/face", response_class=HTMLResponse, include_in_schema=False)
    async def face_page(request: Request):
        """Serve the robot face HTML page."""
        # Inject API key into WebSocket URL if provided via query param
        api_key = request.query_params.get("api_key", "")
        if api_key:
            html = FACE_HTML.replace(
                "'/ws/display'",
                f"'/ws/display?api_key={api_key}'",
            )
            # Also handle the template literal form
            html = html.replace(
                "location.host + '/ws/display'",
                f"location.host + '/ws/display?api_key={api_key}'",
            )
        else:
            html = FACE_HTML
        return html

    @router.post("/display/text")
    async def display_text(req: TextRequest):
        """Show text on the face display."""
        display.set_text(req.text, req.size)
        return {"ok": True}

    @router.post("/display/face")
    async def display_face(req: FaceRequest):
        """Change the face expression."""
        if req.expression not in VALID_EXPRESSIONS:
            from fastapi import HTTPException
            raise HTTPException(400, f"Invalid expression: {req.expression}. Valid: {sorted(VALID_EXPRESSIONS)}")
        display.set_face(req.expression)
        return {"ok": True}

    @router.post("/display/image")
    async def display_image(req: ImageRequest):
        """Show an image on the face display."""
        # Basic size check (~2MB base64 ~ 1.5MB raw)
        if len(req.image_b64) > 3_000_000:
            from fastapi import HTTPException
            raise HTTPException(413, "Image too large (max ~2MB)")
        display.set_image(req.image_b64, req.mime_type)
        return {"ok": True}

    @router.post("/display/clear")
    async def display_clear():
        """Clear display content and revert face to default."""
        display.clear_content()
        return {"ok": True}

    @router.websocket("/ws/display")
    async def ws_display(ws: WebSocket):
        """WebSocket for live display updates to face GUI."""
        if key_store is not None:
            ok, _name, _role = check_ws_auth(ws, key_store)
            if not ok:
                await ws.close(code=4001, reason="Unauthorized")
                return
        await display.connect(ws)
        try:
            while True:
                # Keep connection alive; client doesn't need to send
                await ws.receive_text()
        except WebSocketDisconnect:
            display.disconnect(ws)
        except Exception:
            display.disconnect(ws)

    return router
