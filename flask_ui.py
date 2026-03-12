from __future__ import annotations
import threading
from flask import Flask, request, jsonify, render_template_string
from typing import TYPE_CHECKING
from astrbot.api import logger
from werkzeug.serving import make_server

if TYPE_CHECKING:
    from .main import MyPlugin

def start_flask_app(plugin_instance: "MyPlugin", port: int):
    app = Flask(__name__)

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MyEnhance 记忆管理</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.bootcdn.net/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; background-color: #f8f9fa; }
            .memory-card { margin-bottom: 15px; }
            .scope-header { background: #e9ecef; padding: 10px; border-radius: 5px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="mb-4">MyEnhance 记忆管理</h2>
            
            <div id="content">
                <div class="text-center">加载中...</div>
            </div>
        </div>

        <script>
            function escapeHtml(value = "") {
                return String(value)
                    .replace(/&/g, "&amp;")
                    .replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;");
            }

            function escapeAttr(value = "") {
                return escapeHtml(value)
                    .replace(/\"/g, "&quot;")
                    .replace(/'/g, "&#39;");
            }

            async function loadMemories() {
                const resp = await fetch('/api/list');
                const data = await resp.json();
                const content = document.getElementById('content');
                
                if (Object.keys(data.scopes).length === 0) {
                    content.innerHTML = '<div class="alert alert-info">暂无记忆数据</div>';
                    return;
                }

                let html = '';
                for (const [scope, records] of Object.entries(data.scopes)) {
                    const safeScope = escapeHtml(scope);
                    html += `<div class="scope-header"><h5>会话: ${safeScope} <span class="badge bg-secondary">${records.length}</span></h5></div>`;
                    records.forEach(r => {
                        const rawContent = r.content ?? '';
                        const rawKeyword = r.keyword ?? '';
                        const safeContent = escapeHtml(rawContent);
                        const safeKeyword = escapeHtml(rawKeyword);
                        const keywordDisplay = rawKeyword ? safeKeyword : '<span class="text-muted">未设置</span>';
                        const scopeAttr = escapeAttr(scope);
                        const idAttr = escapeAttr(r.id ?? '');
                        const contentData = encodeURIComponent(rawContent);
                        const keywordData = encodeURIComponent(rawKeyword);
                        html += `
                            <div class="card memory-card">
                                <div class="card-body">
                                    <div class="d-flex justify-content-between">
                                        <h6 class="card-subtitle mb-2 text-muted">ID: ${escapeHtml(r.id)} | 更新: ${escapeHtml(r.updated_at)}</h6>
                                        <div>
                                            <button class="btn btn-sm btn-outline-primary" data-scope="${scopeAttr}" data-id="${idAttr}" data-content="${contentData}" data-keyword="${keywordData}" onclick="editMemory(this)">编辑</button>
                                            <button class="btn btn-sm btn-outline-danger" data-scope="${scopeAttr}" data-id="${idAttr}" onclick="deleteMemory(this)">删除</button>
                                        </div>
                                    </div>
                                    <p class="card-text mb-1"><strong>关键词：</strong>${keywordDisplay}</p>
                                    <p class="card-text">${safeContent}</p>
                                </div>
                            </div>`;
                    });
                }
                content.innerHTML = html;
            }

            async function editMemory(button) {
                const scope = button?.dataset?.scope;
                const id = button?.dataset?.id;
                if (!scope || !id) {
                    alert('编辑目标信息缺失');
                    return;
                }
                const oldContent = decodeURIComponent(button.dataset.content || '');
                const oldKeyword = decodeURIComponent(button.dataset.keyword || '');
                const keywordInput = prompt('关键词（用于检索的主语）：', oldKeyword || '');
                if (keywordInput === null) return;
                const trimmedKeyword = keywordInput.trim();
                if (!trimmedKeyword) {
                    alert('关键词不能为空');
                    return;
                }
                const newContent = prompt('修改记忆内容:', oldContent);
                if (newContent === null) return;
                const hasKeywordChange = trimmedKeyword !== (oldKeyword || '').trim();
                const hasContentChange = newContent !== oldContent;
                if (!hasKeywordChange && !hasContentChange) return;
                if (hasContentChange && !newContent.trim()) {
                    alert('内容不能为空');
                    return;
                }

                const payload = {scope, id};
                if (hasContentChange) {
                    payload.content = newContent;
                }
                if (hasKeywordChange) {
                    payload.keyword = trimmedKeyword;
                }

                const resp = await fetch('/api/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload),
                });
                const res = await resp.json();
                alert(res.msg);
                loadMemories();
            }

            async function deleteMemory(button) {
                const scope = button?.dataset?.scope;
                const id = button?.dataset?.id;
                if (!scope || !id) {
                    alert('删除目标信息缺失');
                    return;
                }
                if (!confirm('确定要删除这条记忆吗？')) return;
                
                const resp = await fetch('/api/delete', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({scope, id})
                });
                const res = await resp.json();
                alert(res.msg);
                loadMemories();
            }

            loadMemories();
        </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/api/list')
    def api_list():
        data = {"scopes": {}}
        scopes_items = list(plugin_instance.memory_store.memories_by_scope.items())
        
        for s_id, records in scopes_items:
            data["scopes"][s_id] = [
                {
                    "id": r.id,
                    "content": r.content,
                    "keyword": getattr(r, "keyword", ""),
                    "updated_at": r.updated_at,
                }
                for r in records
            ]
        return jsonify(data)

    @app.route('/api/update', methods=['POST'])
    def api_update():
        data = request.json
        success = plugin_instance.memory_store.update_memory(
            data.get("scope"), data.get("id"), data.get("content"), data.get("keyword")
        )
        return jsonify({"success": bool(success), "msg": "修改成功" if success else "修改失败"})

    @app.route('/api/delete', methods=['POST'])
    def api_delete():
        data = request.json
        success = plugin_instance.memory_store.delete_memory(
            data.get("scope"), data.get("id")
        )
        return jsonify({"success": success, "msg": "删除成功" if success else "删除失败"})

    server = make_server('0.0.0.0', port, app)
    plugin_instance._flask_server = server

    def run():
        try:
            server.serve_forever()
        except Exception as e:
            logger.error(f"[myenhance] Flask failed to start: {e}")
        finally:
            logger.info("[myenhance] Flask server thread exiting")
            try:
                server.server_close()
            except Exception:
                pass
            plugin_instance._flask_server = None
            plugin_instance._flask_thread = None

    thread = threading.Thread(target=run, daemon=True)
    plugin_instance._flask_thread = thread
    thread.start()

    def stop_flask():
        logger.info("[myenhance] stop_flask called")
        try:
            server = getattr(plugin_instance, "_flask_server", None)
            thread = getattr(plugin_instance, "_flask_thread", None)
            logger.info(
                "[myenhance] stop_flask state: server_exists=%s thread_exists=%s thread_alive=%s",
                server is not None,
                thread is not None,
                bool(thread is not None and thread.is_alive()),
            )
            if server is not None:
                logger.info("[myenhance] stopping Flask server")
                server.shutdown()
                logger.info("[myenhance] Flask server shutdown signal sent")
            else:
                logger.warning("[myenhance] stop_flask skipped: _flask_server is None")
        except Exception as exc:
            logger.warning(f"[myenhance] Flask stop failed: {exc}")
        finally:
            if thread is not None and thread.is_alive():
                logger.info("[myenhance] waiting for Flask thread to exit")
                thread.join(timeout=2)
                logger.info(
                    "[myenhance] Flask thread join finished: still_alive=%s",
                    thread.is_alive(),
                )
            elif thread is None:
                logger.warning("[myenhance] stop_flask skipped thread join: _flask_thread is None")
            plugin_instance._flask_server = None
            plugin_instance._flask_thread = None
            logger.info("[myenhance] Flask shutdown state cleared")
            
    logger.info(f"[myenhance] Flask UI started on port {port}")
    return stop_flask
