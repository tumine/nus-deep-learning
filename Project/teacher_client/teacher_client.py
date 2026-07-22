import json
import webbrowser
import threading
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# ---------- 全局数据存储 ----------
# 存储所有接收到的消息（用于页面初始化时加载历史记录）
messages_store: List[Dict[str, Any]] = []
# 存储当前所有活跃的 WebSocket 连接（用于广播）
active_connections: List[WebSocket] = []

# ---------- FastAPI 应用 ----------
app = FastAPI()

# ---------- 前端 HTML 页面（内嵌） ----------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>教师端 - 小车信息监控</title>
    <style>
        * {
            box-sizing: border-box;
            font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
        }
        body {
            margin: 20px;
            background: #f5f7fb;
            color: #1e293b;
        }
        h1 {
            font-weight: 500;
            font-size: 1.8rem;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #64748b;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
        }
        .left-panel {
            flex: 2;
            min-width: 500px;
        }
        .right-panel {
            flex: 1;
            min-width: 300px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .card h2 {
            font-size: 1.2rem;
            font-weight: 500;
            margin-top: 0;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .badge {
            font-size: 0.8rem;
            background: #e2e8f0;
            padding: 2px 10px;
            border-radius: 20px;
            color: #475569;
        }
        /* 表格样式 */
        .table-wrap {
            max-height: 500px;
            overflow-y: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        th {
            text-align: left;
            padding: 8px 6px;
            background: #f1f5f9;
            position: sticky;
            top: 0;
            z-index: 2;
            font-weight: 500;
        }
        td {
            padding: 8px 6px;
            border-bottom: 1px solid #f0f0f0;
            vertical-align: middle;
        }
        tr:hover td {
            background: #fafcff;
        }
        .tag {
            display: inline-block;
            padding: 2px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            color: white;
        }
        .tag-item {
            background: #3b82f6;
        }
        .tag-help {
            background: #ef4444;
        }
        .time-col {
            color: #64748b;
            font-size: 0.8rem;
            white-space: nowrap;
        }
        .desc-col {
            max-width: 180px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .empty-msg {
            color: #94a3b8;
            text-align: center;
            padding: 30px 0;
        }
        /* 散点图 */
        #chartCanvas {
            width: 100%;
            height: auto;
            aspect-ratio: 1/1;
            background: white;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            cursor: crosshair;
            display: block;
        }
        .chart-legend {
            display: flex;
            gap: 20px;
            font-size: 0.85rem;
            margin-top: 8px;
            flex-wrap: wrap;
        }
        .legend-dot {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .coord-info {
            margin-top: 12px;
            font-size: 0.9rem;
            background: #f8fafc;
            padding: 8px 12px;
            border-radius: 8px;
            color: #334155;
        }
        .status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online {
            background: #22c55e;
        }
        .status-offline {
            background: #94a3b8;
        }
        .footer {
            margin-top: 20px;
            color: #94a3b8;
            font-size: 0.85rem;
        }
        /* 滚动条美化 */
        .table-wrap::-webkit-scrollbar {
            width: 6px;
        }
        .table-wrap::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 8px;
        }
        .table-wrap::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 8px;
        }
    </style>
</head>
<body>

    <h1>🚗 小车信息监控</h1>
    <div class="subtitle">
        <span class="status status-online" id="statusDot"></span>
        <span id="statusText">连接中...</span>
        &nbsp;·&nbsp; 共 <span id="msgCount">0</span> 条消息
    </div>

    <div class="container">
        <!-- 左侧：消息列表 -->
        <div class="left-panel">
            <div class="card">
                <h2>
                    📋 实时消息
                    <span class="badge" id="liveBadge">实时</span>
                </h2>
                <div class="table-wrap" id="tableWrap">
                    <table>
                        <thead>
                            <tr>
                                <th>时间</th>
                                <th>消息ID</th>
                                <th>坐标 (X, Y)</th>
                                <th>请求类型</th>
                                <th>描述</th>
                            </tr>
                        </thead>
                        <tbody id="msgBody">
                            <tr><td colspan="5" class="empty-msg">暂无消息，等待小车上报...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- 右侧：坐标图 -->
        <div class="right-panel">
            <div class="card">
                <h2>📍 位置散点图</h2>
                <canvas id="chartCanvas" width="400" height="400"></canvas>
                <div class="chart-legend">
                    <span><span class="legend-dot" style="background:#3b82f6;"></span> 物品请求</span>
                    <span><span class="legend-dot" style="background:#ef4444;"></span> 教师协助</span>
                </div>
                <div class="coord-info" id="coordInfo">点击散点查看详情</div>
            </div>
        </div>
    </div>

    <div class="footer">
        教师端 · 数据通过 WebSocket 实时推送
    </div>

    <script>
        // ---------- DOM 引用 ----------
        const msgBody = document.getElementById('msgBody');
        const msgCount = document.getElementById('msgCount');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const coordInfo = document.getElementById('coordInfo');

        const canvas = document.getElementById('chartCanvas');
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;

        // 存储所有消息（用于绘图）
        let messages = [];

        // ---------- 工具函数 ----------
        function formatTime(iso) {
            const d = new Date(iso);
            return d.toLocaleString('zh-CN', { hour12: false });
        }

        // 获取请求类型标签
        function getTag(request) {
            if (request === '物品') {
                return '<span class="tag tag-item">物品</span>';
            } else if (request === '教师协助') {
                return '<span class="tag tag-help">协助</span>';
            }
            return `<span class="tag" style="background:#94a3b8;">${request}</span>`;
        }

        // ---------- 渲染表格 ----------
        function renderTable() {
            if (messages.length === 0) {
                msgBody.innerHTML = `<tr><td colspan="5" class="empty-msg">暂无消息，等待小车上报...</td></tr>`;
                msgCount.textContent = '0';
                return;
            }
            // 按时间倒序（最新的在上）
            const sorted = [...messages].reverse();
            let html = '';
            sorted.forEach(msg => {
                const time = formatTime(msg.received_at);
                const tagHtml = getTag(msg.request);
                const desc = msg.description || '-';
                html += `<tr>
                    <td class="time-col">${time}</td>
                    <td>${msg.message_id}</td>
                    <td>(${msg.axis_x}, ${msg.axis_y})</td>
                    <td>${tagHtml}</td>
                    <td class="desc-col" title="${desc}">${desc}</td>
                </tr>`;
            });
            msgBody.innerHTML = html;
            msgCount.textContent = messages.length;
        }

        // ---------- 绘制散点图 ----------
        function drawChart() {
            ctx.clearRect(0, 0, W, H);

            // 绘制背景网格
            ctx.strokeStyle = '#e9edf4';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 10; i++) {
                const x = (i / 10) * W;
                const y = (i / 10) * H;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, H);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(W, y);
                ctx.stroke();
            }

            // 坐标轴标注
            ctx.fillStyle = '#94a3b8';
            ctx.font = '11px sans-serif';
            ctx.fillText('X', W - 20, 20);
            ctx.fillText('Y', 10, H - 10);

            if (messages.length === 0) {
                ctx.fillStyle = '#cbd5e1';
                ctx.font = '14px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('等待数据...', W/2, H/2);
                return;
            }

            // 计算坐标范围（假设坐标在 0~100 之间，但为了适应任意值，取所有数据的 min/max，并留边距）
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            messages.forEach(m => {
                if (m.axis_x < minX) minX = m.axis_x;
                if (m.axis_x > maxX) maxX = m.axis_x;
                if (m.axis_y < minY) minY = m.axis_y;
                if (m.axis_y > maxY) maxY = m.axis_y;
            });
            // 防止单点或全相等
            if (minX === maxX) { minX -= 1; maxX += 1; }
            if (minY === maxY) { minY -= 1; maxY += 1; }
            // 扩充边距 10%
            const padX = (maxX - minX) * 0.1;
            const padY = (maxY - minY) * 0.1;
            minX -= padX; maxX += padX;
            minY -= padY; maxY += padY;

            // 映射函数
            const mapX = (val) => ((val - minX) / (maxX - minX)) * (W - 30) + 15;
            const mapY = (val) => H - (((val - minY) / (maxY - minY)) * (H - 30) + 15);

            // 绘制点
            messages.forEach((msg, index) => {
                const x = mapX(msg.axis_x);
                const y = mapY(msg.axis_y);
                const color = msg.request === '物品' ? '#3b82f6' : '#ef4444';
                const radius = 6;

                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.stroke();

                // 保存坐标信息用于点击检测
                msg._cx = x;
                msg._cy = y;
                msg._index = index;
            });

            // 添加坐标轴范围文字
            ctx.fillStyle = '#64748b';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`X: ${minX.toFixed(1)} ~ ${maxX.toFixed(1)}`, 10, H - 4);
            ctx.textAlign = 'right';
            ctx.fillText(`Y: ${minY.toFixed(1)} ~ ${maxY.toFixed(1)}`, W - 10, 14);
        }

        // ---------- 处理新消息 ----------
        function addMessage(msg) {
            // 添加接收时间（服务端已添加，但以防万一）
            if (!msg.received_at) {
                msg.received_at = new Date().toISOString();
            }
            messages.push(msg);
            renderTable();
            drawChart();
        }

        // ---------- 画布点击查看详情 ----------
        canvas.addEventListener('click', function(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const mouseX = (e.clientX - rect.left) * scaleX;
            const mouseY = (e.clientY - rect.top) * scaleY;

            // 逆序查找（后添加的点在上层，优先显示最新的）
            let found = null;
            for (let i = messages.length - 1; i >= 0; i--) {
                const m = messages[i];
                if (m._cx === undefined) continue;
                const dx = mouseX - m._cx;
                const dy = mouseY - m._cy;
                if (dx*dx + dy*dy < 100) { // 半径 10 像素范围内
                    found = m;
                    break;
                }
            }
            if (found) {
                coordInfo.innerHTML = `
                    <strong>消息ID:</strong> ${found.message_id} &nbsp;|&nbsp;
                    <strong>坐标:</strong> (${found.axis_x}, ${found.axis_y}) &nbsp;|&nbsp;
                    <strong>请求:</strong> ${found.request} &nbsp;|&nbsp;
                    <strong>描述:</strong> ${found.description || '-'} &nbsp;|&nbsp;
                    <strong>时间:</strong> ${formatTime(found.received_at)}
                `;
            } else {
                coordInfo.textContent = '点击散点查看详情';
            }
        });

        // ---------- WebSocket 连接 ----------
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        let socket = null;

        function connectWS() {
            socket = new WebSocket(wsUrl);

            socket.onopen = function() {
                statusDot.className = 'status status-online';
                statusText.textContent = '已连接';
                console.log('WebSocket 连接成功');
            };

            socket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'history') {
                        // 服务端发送历史消息
                        data.data.forEach(msg => addMessage(msg));
                    } else if (data.type === 'new_message') {
                        addMessage(data.data);
                    }
                } catch (e) {
                    console.error('解析消息失败:', e);
                }
            };

            socket.onclose = function() {
                statusDot.className = 'status status-offline';
                statusText.textContent = '断开连接，尝试重连...';
                console.log('WebSocket 断开，5秒后重连');
                setTimeout(connectWS, 5000);
            };

            socket.onerror = function(err) {
                console.error('WebSocket 错误:', err);
                socket.close();
            };
        }

        // 启动连接
        connectWS();

        // 页面关闭时主动断开
        window.addEventListener('beforeunload', function() {
            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.close();
            }
        });

        // 初始占位绘制
        drawChart();

        console.log('教师端监控页面已加载');
    </script>
</body>
</html>
"""

# ---------- FastAPI 路由 ----------
@app.get("/")
async def get_root():
    """返回前端 HTML 页面"""
    return HTMLResponse(HTML_PAGE)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点：接收小车消息并广播给所有前端客户端"""
    await websocket.accept()
    active_connections.append(websocket)

    # 发送历史消息给新连接的客户端
    try:
        if messages_store:
            await websocket.send_json({
                "type": "history",
                "data": messages_store
            })
    except Exception:
        pass

    try:
        while True:
            # 接收来自客户端的消息（这里期望小车发送 JSON）
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # 非 JSON 格式，忽略或记录
                continue

            # 校验必要字段
            required = {"message_id", "axis_x", "axis_y", "request", "description"}
            if not required.issubset(data.keys()):
                # 缺少字段，忽略
                continue

            # 添加服务端接收时间
            data["received_at"] = datetime.now().isoformat()

            # 存储到全局列表
            messages_store.append(data)

            # 广播给所有连接的 WebSocket 客户端（包括当前）
            broadcast_msg = {
                "type": "new_message",
                "data": data
            }
            for conn in active_connections:
                try:
                    await conn.send_json(broadcast_msg)
                except Exception:
                    # 连接可能已断开，后续清理
                    pass

    except WebSocketDisconnect:
        # 移除断开的连接
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception as e:
        # 其他异常，断开连接
        if websocket in active_connections:
            active_connections.remove(websocket)
        await websocket.close()

# ---------- 启动入口 ----------
def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    # 延迟0.5秒打开浏览器，确保服务已启动
    threading.Timer(0.5, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)