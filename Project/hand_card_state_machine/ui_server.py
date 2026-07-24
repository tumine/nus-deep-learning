"""
ui_server.py

FastAPI/WebSocket browser interface for the classroom assistant robot.
The server runs in a daemon thread started by main.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import uuid
import webbrowser

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from ui_manager import UIManager


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
          <button id="audioButton" class="load" onclick="enableAudio()">
            Enable Phone Audio
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
let audioUnlocked = false;
let activeAudio = null;

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
  }else if(type==="play_audio"){
    playRemoteAudio(data);
  }
}
function sendPlaybackResult(type,requestId,error){
  if(socket&&socket.readyState===WebSocket.OPEN){
    socket.send(JSON.stringify({type,request_id:requestId,error:error||""}));
  }
}
async function enableAudio(){
  try{
    const silentAudio=new Audio();
    silentAudio.muted=true;
    await silentAudio.play().catch(()=>{});
    audioUnlocked=true;
    const button=document.getElementById("audioButton");
    button.textContent="Phone Audio Ready";
    button.disabled=true;
    log("Phone audio enabled.");
  }catch(error){
    log("Audio permission was not granted.");
  }
}
function playRemoteAudio(data){
  if(!audioUnlocked){
    sendPlaybackResult("audio_playback_error",data.request_id,"Phone audio is not enabled.");
    log("Audio rejected: enable phone audio first.");
    return;
  }
  try{
    if(activeAudio){
      activeAudio.pause();
      activeAudio=null;
    }
    const bytes=Uint8Array.from(atob(data.audio_base64),character=>character.charCodeAt(0));
    const url=URL.createObjectURL(new Blob([bytes],{type:data.media_type||"audio/mp4"}));
    const audio=new Audio(url);
    activeAudio=audio;
    audio.onended=()=>{
      URL.revokeObjectURL(url);
      activeAudio=null;
      sendPlaybackResult("audio_playback_complete",data.request_id);
      log(`Audio ${data.audio_id} finished.`);
    };
    audio.onerror=()=>{
      URL.revokeObjectURL(url);
      activeAudio=null;
      sendPlaybackResult("audio_playback_error",data.request_id,"Browser could not play the audio.");
    };
    audio.play().catch(error=>{
      URL.revokeObjectURL(url);
      activeAudio=null;
      sendPlaybackResult("audio_playback_error",data.request_id,error.message);
    });
    log(`Playing audio ${data.audio_id}.`);
  }catch(error){
    sendPlaybackResult("audio_playback_error",data.request_id,error.message);
  }
}
function connect(){
  const protocol=location.protocol==="https:"?"wss":"ws";
  socket=new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onopen=()=>{
    const n=document.getElementById("wsStatus");
    n.textContent="● WebSocket connected";n.className="online";log("WebSocket connected.");
  };
  socket.onmessage=e=>{try{handleMessage(JSON.parse(e.data));}catch(err){log("Invalid server message.");}};
  socket.onclose=()=>{
    const n=document.getElementById("wsStatus");
    n.textContent="● WebSocket disconnected";n.className="offline";
    setTimeout(connect,3000);
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
        host: str = "0.0.0.0",
        port: int = 8000,
        open_browser: bool = True,
    ) -> None:
        self.ui_manager = ui_manager
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.app = FastAPI()
        self.active_connections: list[WebSocket] = []
        self._pending_audio: dict[str, asyncio.Future[bool]] = {}
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/")
        async def root() -> HTMLResponse:
            return HTMLResponse(HTML_PAGE)

        @self.app.post("/api/audio")
        async def play_audio(request: Request) -> JSONResponse:
          try:
            payload = await request.json()
            audio_base64 = str(payload["audio_base64"])
            base64.b64decode(audio_base64, validate=True)
          except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise HTTPException(400, "Invalid audio payload.") from error

          if not self.active_connections:
            raise HTTPException(503, "No phone browser is connected.")

          request_id = uuid.uuid4().hex
          completion = asyncio.get_running_loop().create_future()
          self._pending_audio[request_id] = completion
          event = {
            "type": "play_audio",
            "data": {
              "request_id": request_id,
              "audio_id": payload.get("audio_id"),
                    "filename": payload.get("filename", "audio.m4a"),
                    "media_type": payload.get("media_type", "audio/mp4"),
              "audio_base64": audio_base64,
            },
          }

          try:
            delivered = await self._broadcast(event)
            if not delivered:
              raise HTTPException(503, "No phone browser is connected.")
            played = await asyncio.wait_for(completion, timeout=15.0)
          except TimeoutError as error:
            raise HTTPException(504, "Timed out waiting for phone playback.") from error
          finally:
            self._pending_audio.pop(request_id, None)

          if not played:
            raise HTTPException(409, "Phone browser could not play the audio.")

          return JSONResponse({"played": True, "request_id": request_id})

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
                    elif message.get("type") in {
                      "audio_playback_complete",
                      "audio_playback_error",
                    }:
                      request_id = str(message.get("request_id", ""))
                      completion = self._pending_audio.get(request_id)
                      if completion is not None and not completion.done():
                        completion.set_result(
                          message.get("type") == "audio_playback_complete"
                        )

            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

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

            await self._broadcast(event)

    async def _broadcast(self, event: dict) -> bool:
        delivered = False
        dead_connections: list[WebSocket] = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(event)
                delivered = True
            except Exception:
                dead_connections.append(connection)

        for connection in dead_connections:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
        return delivered

    def run(self) -> None:
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
