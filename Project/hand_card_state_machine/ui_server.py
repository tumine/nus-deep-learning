"""
ui_server.py

FastAPI/WebSocket browser interface for the classroom assistant robot.
The server runs in a daemon thread started by main.py.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from aiortc import RTCPeerConnection, RTCSessionDescription

from voice_transmission_server_speech import AudioProcessor
from fastapi.responses import HTMLResponse

from ui_manager import UIManager


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
            onclick="enablePhoneAudioAndMicrophone()"
          >
            🎙 Enable Audio & Microphone
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
let microphoneConnection = null;
let microphoneStream = null;
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
async function waitForIceGatheringComplete(peerConnection){
  if(peerConnection.iceGatheringState==="complete") return;
  await new Promise(resolve=>{
    const checkState=()=>{
      if(peerConnection.iceGatheringState==="complete"){
        peerConnection.removeEventListener("icegatheringstatechange",checkState);
        resolve();
      }
    };
    peerConnection.addEventListener("icegatheringstatechange",checkState);
    setTimeout(resolve,3000);
  });
}

async function enablePhoneAudioAndMicrophone(){
  const button=document.getElementById("audioEnableButton");
  button.disabled=true;
  button.textContent="Enabling...";
  try{
    const AudioContextClass=window.AudioContext||window.webkitAudioContext;
    if(AudioContextClass){
      audioContext=audioContext||new AudioContextClass();
      await audioContext.resume();
    }

    microphoneStream=await navigator.mediaDevices.getUserMedia({
      audio:{
        echoCancellation:true,
        noiseSuppression:true,
        autoGainControl:true
      },
      video:false
    });

    if(microphoneConnection){
      microphoneConnection.close();
    }
    microphoneConnection=new RTCPeerConnection();
    microphoneStream.getTracks().forEach(track=>{
      microphoneConnection.addTrack(track,microphoneStream);
    });
    microphoneConnection.onconnectionstatechange=()=>{
      log(`Microphone connection: ${microphoneConnection.connectionState}`);
    };

    const offer=await microphoneConnection.createOffer();
    await microphoneConnection.setLocalDescription(offer);
    await waitForIceGatheringComplete(microphoneConnection);

    const response=await fetch("/offer",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        sdp:microphoneConnection.localDescription.sdp,
        type:microphoneConnection.localDescription.type
      })
    });
    if(!response.ok){
      throw new Error(`WebRTC offer failed: HTTP ${response.status}`);
    }
    const answer=await response.json();
    await microphoneConnection.setRemoteDescription(answer);

    phoneAudioEnabled=true;
    button.textContent="Audio & Microphone Enabled";
    log("Phone audio and microphone enabled.");
    if(socket&&socket.readyState===WebSocket.OPEN){
      socket.send(JSON.stringify({type:"enable_audio"}));
    }
  }catch(error){
    button.disabled=false;
    button.textContent="🎙 Enable Audio & Microphone";
    log(`Unable to enable audio/microphone: ${error.message||error}.`);
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
    log(`New request: ${data.description||"-"}`);
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
        speech_detector=None,
        host: str = "0.0.0.0",
        port: int = 8000,
        open_browser: bool = True,
    ) -> None:
        self.ui_manager = ui_manager
        self.speech_detector = speech_detector
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.app = FastAPI()
        self.active_connections: list[WebSocket] = []
        self.audio_enabled_connections: list[WebSocket] = []
        self._audio_playback_waiters: dict[str, asyncio.Future[bool]] = {}
        self._audio_playback_traces: dict[str, str] = {}
        self._peer_connection: RTCPeerConnection | None = None
        self._microphone_task: asyncio.Task | None = None
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

        @self.app.post("/offer")
        async def receive_microphone_offer(offer_payload: dict[str, Any]) -> dict[str, str]:
            """Receive the phone microphone through WebRTC on the same UI server."""
            if self.speech_detector is None:
                raise HTTPException(
                    status_code=503,
                    detail="Speech detector is not configured.",
                )

            await self._close_microphone_connection()
            peer_connection = RTCPeerConnection()
            self._peer_connection = peer_connection

            @peer_connection.on("track")
            async def on_track(track) -> None:
                if track.kind != "audio":
                    return
                print("[SPEECH] Phone microphone track connected on UI port 8000.")
                processor = AudioProcessor(
                    detector=None,
                    speech_detector=self.speech_detector,
                )
                self._microphone_task = asyncio.create_task(
                    self._consume_microphone(track, processor)
                )

            @peer_connection.on("connectionstatechange")
            async def on_connection_state_change() -> None:
                state = peer_connection.connectionState
                print(f"[SPEECH] WebRTC microphone state: {state}")
                if state in {"failed", "closed"}:
                    await self._close_microphone_connection()

            try:
                offer = RTCSessionDescription(
                    sdp=str(offer_payload["sdp"]),
                    type=str(offer_payload["type"]),
                )
                await peer_connection.setRemoteDescription(offer)
                answer = await peer_connection.createAnswer()
                await peer_connection.setLocalDescription(answer)
            except Exception as error:
                await self._close_microphone_connection()
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid WebRTC offer: {error}",
                ) from error

            return {
                "sdp": peer_connection.localDescription.sdp,
                "type": peer_connection.localDescription.type,
            }

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

        @self.app.on_event("shutdown")
        async def shutdown_event() -> None:
            await self._close_microphone_connection()

    async def _consume_microphone(self, track, processor: AudioProcessor) -> None:
        """Continuously forward decoded WebRTC audio frames to SpeechRequestDetector."""
        try:
            while True:
                frame = await track.recv()
                processor.add_frame(frame)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            print(f"[SPEECH WARNING] Microphone stream ended: {error}")

    async def _close_microphone_connection(self) -> None:
        task = self._microphone_task
        self._microphone_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        peer_connection = self._peer_connection
        self._peer_connection = None
        if peer_connection is not None:
            try:
                await peer_connection.close()
            except Exception:
                pass

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
        if self.open_browser:
            threading.Timer(
                1.0,
                lambda: webbrowser.open(f"http://127.0.0.1:{self.port}"),
            ).start()

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
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
