"""
ui_server.py

FastAPI/WebSocket browser interface for the classroom assistant robot.
The server runs in a daemon thread started by main.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from intent_parser import parse_request as parse_intent
from ui_manager import UIManager
from voice_transmission import VoiceTransmissionManager


def _configure_firewall(port: int) -> None:
    """Add a Windows firewall inbound rule for the UI/audio port.

    Without this rule, Windows may silently drop the Raspberry Pi's POST
    requests to ``/api/audio`` arriving through the Tailscale/LAN interface.
    The Pi's TCP SYNs then get no reply and the Pi stalls until its own
    client-side timeout expires (previously a 20 second hang per prompt).
    """
    if sys.platform != "win32":
        return
    rule_name = f"RobotUIServer_Port{port}"
    try:
        check = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule_name}"],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="replace",
        )
        if check.returncode == 0 and "No rules match" not in check.stdout:
            print(f"[FIREWALL] Inbound rule already exists: port {port}")
            return
        subprocess.run(
            ["netsh", "advfirewall", "firewall", "add", "rule",
             f"name={rule_name}", "dir=in", "action=allow",
             "protocol=TCP", f"localport={port}"],
            capture_output=True, timeout=5,
            encoding="utf-8", errors="replace",
        )
        print(f"[FIREWALL] Added inbound rule: port {port}")
    except Exception as error:
        print(f"[FIREWALL WARNING] Could not configure port {port}: {error}")
        print(
            "[FIREWALL WARNING] Run once as administrator: "
            f'netsh advfirewall firewall add rule name="{rule_name}" '
            f"dir=in action=allow protocol=TCP localport={port}"
        )


def _generate_ssl_cert_paths() -> tuple[str | None, str | None]:
    """Generate a self-signed SSL certificate for HTTPS support.

    Mobile browsers require HTTPS for ``getUserMedia`` (microphone access).
    This function tries openssl first, then falls back to the ``cryptography``
    library.  Certificates are cached in ``~/.voice_transmission_server/``.

    Returns:
        ``(cert_path, key_path)`` or ``(None, None)`` if generation fails.
    """
    cert_dir = Path.home() / ".voice_transmission_server"
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    if cert_path.exists() and key_path.exists():
        print(f"[SSL] Using existing certificate: {cert_path}")
        return str(cert_path), str(key_path)

    cert_dir.mkdir(parents=True, exist_ok=True)

    # ---- Method 1: openssl ------------------------------------------
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_path), "-out", str(cert_path),
                "-days", "3650", "-nodes",
                "-subj", "/CN=RobotControlCenter",
            ],
            capture_output=True, check=True, timeout=30,
        )
        print(f"[SSL] Certificate generated (openssl): {cert_path}")
        return str(cert_path), str(key_path)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        pass

    # ---- Method 2: cryptography library -----------------------------
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime as dt

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "RobotControlCenter"),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(dt.datetime.utcnow())
            .not_valid_after(
                dt.datetime.utcnow() + dt.timedelta(days=3650)
            )
            .sign(key, hashes.SHA256())
        )
        key_path.write_bytes(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        print(f"[SSL] Certificate generated (cryptography): {cert_path}")
        return str(cert_path), str(key_path)
    except ImportError:
        pass

    # ---- No SSL available -------------------------------------------
    print(
        "[SSL WARNING] Cannot generate SSL certificate "
        "(openssl and cryptography not available)."
    )
    print(
        "[SSL WARNING] Mobile browsers will NOT be able to access "
        "the microphone — HTTPS is required for getUserMedia()."
    )
    return None, None


def _detect_ips() -> tuple[str | None, list[str]]:
    """Detect local network IPs, distinguishing Tailscale (100.64.0.0/10)
    from LAN addresses.

    Returns:
        ``(tailscale_ip, lan_ips)``
    """
    import socket as _socket

    all_ips: list[str] = []

    # Method 1: psutil (most reliable, no DNS)
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            for addr in addrs:
                if (
                    addr.family == _socket.AF_INET
                    and not addr.address.startswith("127.")
                ):
                    all_ips.append(addr.address)
    except (ImportError, Exception):
        pass

    # Method 2: UDP connect probe
    for probe in ("100.64.0.1", "8.8.8.8"):
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as s:
                s.settimeout(1)
                s.connect((probe, 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    all_ips.append(ip)
        except OSError:
            pass

    # Deduplicate
    seen: set[str] = set()
    unique = [ip for ip in all_ips if not (ip in seen or seen.add(ip))]  # type: ignore[func-returns-value]

    # Classify
    tailscale_ip: str | None = None
    lan_ips: list[str] = []
    for ip in unique:
        parts = ip.split(".")
        if (
            len(parts) == 4
            and ip.startswith("100.")
            and 64 <= int(parts[1]) <= 127
        ):
            tailscale_ip = ip
        else:
            lan_ips.append(ip)

    return tailscale_ip, lan_ips


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Robot Control Center</title>
<style>
*{box-sizing:border-box;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
body{margin:0;background:#f4f7fb;color:#172033}
header{padding:22px 28px;background:#172033;color:white;display:flex;justify-content:space-between;align-items:center}
header h1{margin:0;font-size:1.55rem}.online{color:#4ade80}.offline{color:#f87171}
main{padding:22px;max-width:1400px;margin:auto}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}
.card{background:white;border-radius:15px;padding:18px;box-shadow:0 4px 15px rgba(15,23,42,.07)}
.label{color:#64748b;font-size:.82rem}.value{font-size:1.25rem;font-weight:700;margin-top:7px}
.content{display:grid;grid-template-columns:2fr 1fr;gap:18px}
h2{margin:0 0 13px;font-size:1.08rem}
table{width:100%;border-collapse:collapse;font-size:.88rem}
th,td{padding:10px 8px;border-bottom:1px solid #edf0f5;text-align:left}
th{background:#f8fafc;position:sticky;top:0}.table-wrap{max-height:470px;overflow:auto}
.empty{text-align:center;color:#94a3b8;padding:35px}
.actions{display:grid;gap:12px}
button{border:0;border-radius:11px;padding:14px;font-weight:700;cursor:pointer;font-size:.95rem}
.stop{background:#dc2626;color:white}.stop:hover{background:#b91c1c}
.audio{background:#7c3aed;color:white}.audio:hover:not(:disabled){background:#6d28d9}
.mic{background:#059669;color:white}.mic:hover:not(:disabled){background:#047857}
.mic.active{background:#dc2626;color:white}.mic.active:hover:not(:disabled){background:#b91c1c}
.load{background:#2563eb;color:white}.load:hover:not(:disabled){background:#1d4ed8}
.unload{background:#16a34a;color:white}.unload:hover:not(:disabled){background:#15803d}
button:disabled{background:#cbd5e1;color:#64748b;cursor:not-allowed;opacity:.72}
.log{margin-top:15px;background:#0f172a;color:#dbeafe;padding:12px;border-radius:10px;min-height:95px;font-family:monospace;font-size:.82rem;white-space:pre-wrap}
.badge{display:inline-block;padding:3px 9px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:.75rem}
@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}.content{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>🤖 Robot Control Center</h1>
  <div><span id="wsStatus" class="offline">● WebSocket disconnected</span></div>
</header>
<main>
  <section class="grid">
    <div class="card"><div class="label">Robot State</div><div class="value" id="robotState">INITIALIZING</div></div>
    <div class="card"><div class="label">Pi Connection</div><div class="value" id="piStatus">Offline</div></div>
    <div class="card"><div class="label">Scan Direction</div><div class="value" id="scanDirection">-</div></div>
    <div class="card"><div class="label">Route Node</div><div class="value" id="routeNode">-</div></div>
  </section>

  <section class="content">
    <div class="card">
      <h2>📋 Request History <span class="badge" id="requestCount">0</span></h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Time</th><th>ID</th><th>Request</th><th>Description</th><th>Position</th></tr></thead>
          <tbody id="requestBody"><tr><td colspan="5" class="empty">Waiting for requests...</td></tr></tbody>
        </table>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>🎛 Control</h2>
        <div class="actions">
          <button
            class="audio"
            id="audioEnableButton"
            onclick="enablePhoneAudio()"
          >
            🔊 Enable Phone Audio
          </button>

          <button
            class="mic"
            id="micButton"
            onclick="toggleRecording()"
          >
            🎤 Record Request
          </button>

          <button class="stop" onclick="sendCommand('STOP')">
            🛑 EMERGENCY STOP
          </button>

          <button
            class="load"
            id="loadButton"
            onclick="sendCommand('LOAD_COMPLETE')"
            disabled
          >
            📦 Loading Complete
          </button>

          <button
            class="unload"
            id="unloadButton"
            onclick="sendCommand('UNLOAD_COMPLETE')"
            disabled
          >
            ✅ Unloading Complete
          </button>
        </div>
        <div class="log" id="eventLog">UI loaded.</div>
      </div>
      <div class="card" style="margin-top:18px">
        <div class="label">Current Request</div>
        <div class="value" id="currentRequest">-</div>
      </div>
    </div>
  </section>
</main>

<script>
let socket = null;
let requests = [];
let phoneAudioEnabled = false;
let audioContext = null;
let mediaRecorder = null, recordingActive = false, recordedChunks = [];
const reconnectDelayMilliseconds = 500;

function log(text){
  const box=document.getElementById("eventLog");
  box.textContent=`${new Date().toLocaleTimeString()}  ${text}\n`+box.textContent;
}
function updateControlButtons(robotState){
  const loadButton=document.getElementById("loadButton");
  const unloadButton=document.getElementById("unloadButton");

  loadButton.disabled=robotState!=="WAIT_LOADING";
  unloadButton.disabled=robotState!=="WAIT_UNLOAD";
}

function applyStatus(status){
  const robotState=status.robot_state ?? "-";
  document.getElementById("robotState").textContent=robotState;
  document.getElementById("scanDirection").textContent=status.scan_direction ?? "-";
  document.getElementById("routeNode").textContent=status.route_node ?? "-";
  document.getElementById("currentRequest").textContent=status.current_request ?? "-";
  const pi=!!status.pi_connected;
  const piNode=document.getElementById("piStatus");
  piNode.textContent=pi?"Online":"Offline";
  piNode.style.color=pi?"#16a34a":"#dc2626";
  updateControlButtons(robotState);
}
function renderRequests(){
  const body=document.getElementById("requestBody");
  document.getElementById("requestCount").textContent=requests.length;
  if(!requests.length){
    body.innerHTML='<tr><td colspan="5" class="empty">Waiting for requests...</td></tr>';
    return;
  }
  body.innerHTML=[...requests].reverse().map(r=>{
    const time=new Date(r.received_at||Date.now()).toLocaleString("zh-CN",{hour12:false});
    const pos=(r.axis_x==null||r.axis_y==null)?"-":`(${r.axis_x}, ${r.axis_y})`;
    return `<tr><td>${time}</td><td>${r.message_id||"-"}</td><td>${r.request||"-"}</td><td>${r.description||"-"}</td><td>${pos}</td></tr>`;
  }).join("");
}
function reportAudioPlayback(requestId, played){
  if(socket&&socket.readyState===WebSocket.OPEN){
    socket.send(JSON.stringify({
      type:"audio_playback_complete",
      request_id:requestId,
      played
    }));
  }
}
function enablePhoneAudio(){
  const button=document.getElementById("audioEnableButton");
  try{
    const AudioContextClass=window.AudioContext||window.webkitAudioContext;
    if(AudioContextClass){
      audioContext=audioContext||new AudioContextClass();
      audioContext.resume();
    }
    phoneAudioEnabled=true;
    button.disabled=true;
    button.textContent="Phone Audio Enabled";
    log("Phone audio enabled.");
    if(socket&&socket.readyState===WebSocket.OPEN){
      socket.send(JSON.stringify({type:"enable_audio"}));
    }
  }catch(error){
    log("Unable to enable phone audio.");
  }
}
function playBrowserAudio(data){
  let completed=false;
  let audioUrl=null;
  log(`Received audio playback request ${data.request_id}.`);
  const complete=played=>{
    if(completed) return;
    completed=true;
    if(audioUrl) URL.revokeObjectURL(audioUrl);
    reportAudioPlayback(data.request_id,played);
    log(played?"Audio playback completed.":"Audio playback failed.");
  };
  try{
    const binary=atob(data.audio_base64||"");
    const bytes=new Uint8Array(binary.length);
    for(let index=0;index<binary.length;index++) bytes[index]=binary.charCodeAt(index);
    audioUrl=URL.createObjectURL(new Blob([bytes],{type:data.media_type||"audio/mp4"}));
    const audio=new Audio(audioUrl);
    audio.addEventListener("playing",()=>log(`Audio playback started: ${data.request_id}.`),{once:true});
    audio.addEventListener("ended",()=>complete(true),{once:true});
    audio.addEventListener("error",()=>complete(false),{once:true});
    audio.play().catch(error=>{
      log(`Audio playback could not start: ${error.name||"unknown error"}.`);
      complete(false);
    });
  }catch(error){
    log(`Audio payload could not be prepared: ${error.name||"unknown error"}.`);
    complete(false);
  }
}
function handleMessage(message){
  const type=message.type;
  const data=message.data||{};
  if(type==="snapshot"){
    applyStatus(data.status||{});
    requests=data.requests||[];
    renderRequests();
    log("Snapshot received.");
  }else if(type==="robot_state"){
    document.getElementById("robotState").textContent=data.state;
    updateControlButtons(data.state);
    log(`State -> ${data.state}`);
  }else if(type==="scan_update"){
    document.getElementById("scanDirection").textContent=data.direction;
    document.getElementById("routeNode").textContent=data.route_node;
    log(`Scanning ${data.direction}, node ${data.route_node}`);
  }else if(type==="connection_update"){
    if(data.device==="pi"){
      const node=document.getElementById("piStatus");
      node.textContent=data.connected?"Online":"Offline";
      node.style.color=data.connected?"#16a34a":"#dc2626";
    }
    log(`${data.device} connection: ${data.connected}`);
  }else if(type==="new_request"){
    requests.push(data);
    document.getElementById("currentRequest").textContent=data.description||"-";
    renderRequests();
    const confStr=data.confidence!=null?` (confidence: ${(data.confidence*100).toFixed(1)}%)`:"";
    log(`New request: ${data.description||"-"}${confStr}`);
  }else if(type==="audio_playback"){
    log(data.message||"Audio playback update.");
  }else if(type==="play_audio"){
    if(phoneAudioEnabled){
      log("Starting browser audio playback.");
      playBrowserAudio(data);
    }else{
      reportAudioPlayback(data.request_id,false);
      log("Audio playback ignored until phone audio is enabled.");
    }
  }
}
function connect(){
  const protocol=location.protocol==="https:"?"wss":"ws";
  socket=new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onopen=()=>{
    const n=document.getElementById("wsStatus");
    n.textContent="● WebSocket connected";n.className="online";log("WebSocket connected.");
    if(phoneAudioEnabled){
      socket.send(JSON.stringify({type:"enable_audio"}));
    }
  };
  socket.onmessage=e=>{try{handleMessage(JSON.parse(e.data));}catch(err){log("Invalid server message.");}};
  socket.onclose=()=>{
    const n=document.getElementById("wsStatus");
    n.textContent="● WebSocket disconnected";n.className="offline";
    setTimeout(connect,reconnectDelayMilliseconds);
  };
}
function sendCommand(command){
  if(!socket||socket.readyState!==WebSocket.OPEN){
    log("Cannot send command: WebSocket offline.");
    return;
  }

  if(command==="STOP"&&!confirm("确认立即停止小车吗？")) return;

  if(
    command==="LOAD_COMPLETE" &&
    !confirm("确认老师已经完成装载吗？")
  ) return;

  if(
    command==="UNLOAD_COMPLETE" &&
    !confirm("确认学生已经取走物品吗？")
  ) return;

  socket.send(JSON.stringify({type:"control_command",command}));
  log(`Command sent: ${command}`);
}

// ============================================================
//  MediaRecorder Speech Recording (Whisper-based)
// ============================================================

function updateRecordingButton(active){
  const btn=document.getElementById("micButton");
  if(active){
    btn.textContent="⏹ Stop Recording & Send";
    btn.className="mic active";
  }else{
    btn.textContent="🎤 Record Request";
    btn.className="mic";
  }
}

async function toggleRecording(){
  if(recordingActive){
    stopRecording();
  }else{
    await startRecording();
  }
}

async function startRecording(){
  const btn=document.getElementById("micButton");
  btn.disabled=true;
  btn.textContent="⏳ Starting...";

  try{
    if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){
      throw new Error("Browser does not support microphone access. Use HTTPS.");
    }

    const stream=await navigator.mediaDevices.getUserMedia({
      audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}
    });
    log("Microphone access granted. Recording...");

    const mimeType=MediaRecorder.isTypeSupported("audio/webm;codecs=opus")?
      "audio/webm;codecs=opus":"audio/webm";

    mediaRecorder=new MediaRecorder(stream,{mimeType});
    recordedChunks=[];

    mediaRecorder.ondataavailable=(e)=>{
      if(e.data.size>0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop=async()=>{
      stream.getTracks().forEach(t=>t.stop());
      const blob=new Blob(recordedChunks,{type:mimeType});
      log(`Recording stopped. Uploading ${(blob.size/1024).toFixed(1)}KB for transcription...`);

      const formData=new FormData();
      formData.append("audio",blob,"recording.webm");

      try{
        const resp=await fetch("/api/upload_speech",{method:"POST",body:formData});
        const result=await resp.json();
        if(result.status==="success"){
          log(`Transcription: "${result.text}"`);
          log(`Intent: ${result.request}`);
          document.getElementById("currentRequest").textContent=result.request||"-";
        }else{
          log(`Speech recognition failed: ${result.error||"unknown error"}`);
        }
      }catch(err){
        log(`Upload error: ${err.message}`);
      }

      btn.disabled=false;
      updateRecordingButton(false);
    };

    mediaRecorder.start();
    recordingActive=true;
    updateRecordingButton(true);
    btn.disabled=false;

  }catch(err){
    log("Recording error: "+err.message);
    btn.disabled=false;
    updateRecordingButton(false);
  }
}

function stopRecording(){
  if(mediaRecorder&&mediaRecorder.state==="recording"){
    mediaRecorder.stop();
    recordingActive=false;
    log("Stopping recording...");
  }
}
connect();
</script>
</body>
</html>
"""


class UIServer:
    """Run the FastAPI UI and bridge it to a shared UIManager."""

    def __init__(
        self,
        ui_manager: UIManager,
        host: str = "0.0.0.0",
        port: int = 8000,
        open_browser: bool = True,
        voice_transmission_manager: VoiceTransmissionManager | None = None,
        whisper_model: Any = None,
        speech_detector_for_whisper: Any = None,
    ) -> None:
        self.ui_manager = ui_manager
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.voice_manager = voice_transmission_manager
        self.whisper_model = whisper_model
        self.speech_detector_for_whisper = speech_detector_for_whisper

        # Generate self-signed SSL certificate so mobile browsers can
        # access the microphone (getUserMedia requires HTTPS).
        self._ssl_certfile, self._ssl_keyfile = _generate_ssl_cert_paths()
        self._use_https = self._ssl_certfile is not None

        self.app = FastAPI()
        self.active_connections: list[WebSocket] = []
        self.audio_enabled_connections: list[WebSocket] = []
        self._audio_playback_waiters: dict[str, asyncio.Future[bool]] = {}
        self._audio_playback_traces: dict[str, str] = {}
        self._register_routes()

    def _get_audio_connection(self) -> WebSocket | None:
        for connection in reversed(self.audio_enabled_connections):
            if connection in self.active_connections:
                return connection
        return None

    async def _wait_for_audio_connection(self) -> WebSocket | None:
        """Allow a briefly disconnected phone UI to reconnect and re-enable audio."""
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            connection = self._get_audio_connection()
            if connection is not None:
                return connection
            await asyncio.sleep(0.1)
        return None

    def _register_routes(self) -> None:
        @self.app.get("/")
        async def root() -> HTMLResponse:
            return HTMLResponse(HTML_PAGE)

        @self.app.post("/api/audio")
        async def play_audio(audio_payload: dict[str, Any]) -> dict[str, bool]:
            request_started = time.monotonic()
            trace_id = str(audio_payload.get("trace_id") or uuid.uuid4().hex[:8])
            audio_id = audio_payload.get("audio_id", "unknown")
            filename = audio_payload.get("filename", "unknown")
            encoded_size = len(str(audio_payload.get("audio_base64", "")))
            print(
                f"[AUDIO {trace_id}] Received audio {audio_id} ({filename}); "
                f"base64={encoded_size} characters. Waiting for an enabled browser."
            )
            audio_connection = await self._wait_for_audio_connection()

            if audio_connection is None:
                elapsed = time.monotonic() - request_started
                print(
                    f"[AUDIO {trace_id}] No enabled browser after {elapsed:.2f}s; "
                    "rejecting playback request."
                )
                raise HTTPException(
                    status_code=503,
                    detail="No browser has enabled phone audio playback.",
                )

            request_id = str(uuid.uuid4())
            completion = asyncio.get_running_loop().create_future()
            self._audio_playback_waiters[request_id] = completion
            self._audio_playback_traces[request_id] = trace_id
            try:
                await audio_connection.send_json(
                    {
                        "type": "play_audio",
                        "data": {
                            "request_id": request_id,
                            "audio_base64": str(audio_payload.get("audio_base64", "")),
                            "media_type": str(audio_payload.get("media_type", "audio/mp4")),
                        },
                    }
                )
                print(
                  f"[AUDIO {trace_id}] Sent playback request {request_id} "
                  "to the enabled browser; waiting for its completion signal."
                )
            except Exception as error:
                self._audio_playback_waiters.pop(request_id, None)
                self._audio_playback_traces.pop(request_id, None)
                if audio_connection in self.audio_enabled_connections:
                    self.audio_enabled_connections.remove(audio_connection)
                elapsed = time.monotonic() - request_started
                print(
                  f"[AUDIO {trace_id}] Browser send failed after {elapsed:.2f}s: {error}"
                )
                raise HTTPException(
                    status_code=503,
                    detail="Phone audio connection is unavailable.",
                ) from error

            try:
                played = await asyncio.wait_for(completion, timeout=15.0)
            except TimeoutError:
                played = False
                elapsed = time.monotonic() - request_started
                print(
                    f"[AUDIO {trace_id}] Browser did not report playback completion "
                    f"within 15.0s (elapsed {elapsed:.2f}s)."
                )
            finally:
                self._audio_playback_waiters.pop(request_id, None)
                self._audio_playback_traces.pop(request_id, None)

            elapsed = time.monotonic() - request_started
            print(
                f"[AUDIO {trace_id}] Returning playback result played={played} "
                f"after {elapsed:.2f}s."
            )
            return {"played": played}

        @self.app.post("/webrtc/offer")
        async def webrtc_offer(payload: dict[str, Any]) -> dict[str, str]:
            """Handle WebRTC offer from the browser microphone.

            The browser sends its local SDP; the server creates a peer
            connection, sets up audio track handling, and returns an answer.
            """
            if self.voice_manager is None:
                raise HTTPException(
                    status_code=503,
                    detail="Microphone transmission is not available.",
                )

            try:
                answer = await self.voice_manager.handle_offer(
                    str(payload.get("sdp", "")),
                    str(payload.get("type", "offer")),
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"WebRTC error: {exc}"
                ) from exc

            # Broadcast mic status to all connected browsers
            mic_status = self.voice_manager.get_status()
            for conn in list(self.active_connections):
                try:
                    await conn.send_json(
                        {
                            "type": "mic_status",
                            "data": mic_status,
                        }
                    )
                except Exception:
                    pass

            return answer

        @self.app.post("/webrtc/stop")
        async def webrtc_stop() -> dict[str, str]:
            """Stop the current microphone transmission."""
            if self.voice_manager is not None:
                await self.voice_manager.stop()

            # Broadcast stopped status
            for conn in list(self.active_connections):
                try:
                    await conn.send_json(
                        {
                            "type": "mic_status",
                            "data": {"active": False},
                        }
                    )
                except Exception:
                    pass

            return {"status": "stopped"}

        @self.app.post("/api/upload_speech")
        async def upload_speech(audio: UploadFile = File(...)):
            """Receive an audio recording from the browser, transcribe with
            Whisper, and feed the result into the speech request detector."""
            if self.whisper_model is None:
                raise HTTPException(
                    status_code=503,
                    detail="Whisper model is not loaded.",
                )

            suffix = os.path.splitext(
                audio.filename or "recording.webm"
            )[1] or ".webm"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                content = await audio.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                result = await asyncio.to_thread(
                    self.whisper_model.transcribe, tmp_path
                )
                text = result.get("text", "").strip()

                # Feed result into the speech request detector so the
                # state machine can pick it up via poll().
                if self.speech_detector_for_whisper is not None and text:
                    self.speech_detector_for_whisper.push_result(text)

                request_message = parse_intent(text)
                return {
                    "status": "success",
                    "text": text,
                    "request": request_message,
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "error": str(exc),
                }
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self.active_connections.append(websocket)

            await websocket.send_json(
                {
                    "type": "snapshot",
                    "data": self.ui_manager.get_snapshot(),
                }
            )

            try:
                while True:
                    raw = await websocket.receive_text()
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if message.get("type") == "control_command":
                        command = str(message.get("command", "")).upper()
                        if command:
                            self.ui_manager.submit_command(command)
                    elif message.get("type") == "enable_audio":
                        if websocket not in self.audio_enabled_connections:
                            self.audio_enabled_connections.append(websocket)
                        print("[AUDIO] Browser enabled phone audio playback.")
                    elif message.get("type") == "audio_playback_complete":
                        request_id = str(message.get("request_id", ""))
                        completion = self._audio_playback_waiters.get(request_id)
                        if completion is not None and not completion.done():
                          played = bool(message.get("played"))
                          trace_id = self._audio_playback_traces.get(request_id, "unknown")
                          print(
                            f"[AUDIO {trace_id}] Browser reported playback "
                            f"completion for {request_id}; played={played}."
                          )
                          completion.set_result(played)

            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
                if websocket in self.audio_enabled_connections:
                    self.audio_enabled_connections.remove(websocket)

        @self.app.on_event("startup")
        async def startup_event() -> None:
            asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        while True:
            event = await asyncio.to_thread(
                self.ui_manager.get_next_event,
                0.5,
            )
            if event is None:
                continue

            dead_connections: list[WebSocket] = []
            for connection in list(self.active_connections):
                try:
                    await connection.send_json(event)
                except Exception:
                    dead_connections.append(connection)

            for connection in dead_connections:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

    def run(self) -> None:
        if self.host in ("0.0.0.0", ""):
            _configure_firewall(self.port)

        protocol = "https" if self._use_https else "http"

        if self.open_browser:
            threading.Timer(
                1.0,
                lambda: webbrowser.open(
                    f"{protocol}://127.0.0.1:{self.port}"
                ),
            ).start()

        # Print access URLs for mobile / LAN use (mirrors the panns
        # server startup banner so users know the exact address).
        try:
            tailscale_ip, lan_ips = _detect_ips()
        except Exception:
            tailscale_ip, lan_ips = None, []

        print("\n" + "=" * 60)
        print("  Robot Control Center")
        print("=" * 60)
        print(f"\n  Local:   {protocol}://127.0.0.1:{self.port}")
        if tailscale_ip:
            print(f"  Tailscale: {protocol}://{tailscale_ip}:{self.port}"
                  "  <-- phone use this")
        for ip in lan_ips:
            print(f"  LAN:     {protocol}://{ip}:{self.port}")
        if self._use_https:
            print(f"\n  HTTPS enabled (self-signed certificate)")
            print(f"  First visit: accept the certificate warning in browser")
        else:
            print(f"\n  HTTP only — phone microphone requires HTTPS!")
            print(f"  Install: pip install cryptography  (or openssl)")
        print("=" * 60 + "\n")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            ssl_keyfile=self._ssl_keyfile,
            ssl_certfile=self._ssl_certfile,
            log_level="info",
        )

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(
            target=self.run,
            name="ui-server",
            daemon=True,
        )
        thread.start()
        return thread


if __name__ == "__main__":
    # Standalone preview mode.
    preview_manager = UIManager()
    preview_manager.update_robot_state("PREVIEW")
    UIServer(preview_manager).run()
