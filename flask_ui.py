from __future__ import annotations
import threading
from flask import Flask, request, jsonify, render_template_string
from typing import TYPE_CHECKING
from astrbot.api import logger

if TYPE_CHECKING:
    from .main import MyPlugin

def start_flask_app(plugin_instance: "MyPlugin", port: int):
    # 此处也可以添加 print 看看函数是否被调用
    print(f"[myenhance-debug] Starting Flask app on port {port}")
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
                    html += `<div class="scope-header"><h5>会话: ${scope} <span class="badge bg-secondary">${records.length}</span></h5></div>`;
                    records.forEach(r => {
                        html += `
                        <div class="card memory-card">
                            <div class="card-body">
                                <div class="d-flex justify-content-between">
                                    <h6 class="card-subtitle mb-2 text-muted">ID: ${r.id} | 更新: ${r.updated_at}</h6>
                                    <div>
                                        <button class="btn btn-sm btn-outline-primary" onclick="editMemory('${scope}', '${r.id}', \`${r.content}\`)">编辑</button>
                                        <button class="btn btn-sm btn-outline-danger" onclick="deleteMemory('${scope}', '${r.id}')">删除</button>
                                    </div>
                                </div>
                                <p class="card-text">${r.content}</p>
                            </div>
                        </div>`;
                    });
                }
                content.innerHTML = html;
            }

            async function editMemory(scope, id, oldContent) {
                const newContent = prompt('修改记忆内容:', oldContent);
                if (newContent === null || newContent === oldContent) return;
                
                const resp = await fetch('/api/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({scope, id, content: newContent})
                });
                const res = await resp.json();
                alert(res.msg);
                loadMemories();
            }

            async function deleteMemory(scope, id) {
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
        try:
            data = {"scopes": {}}
            scopes_items = list(plugin_instance.memory_store.memories_by_scope.items())
            print(f"[myenhance-debug] Flask /api/list: found {len(scopes_items)} scopes.")
            logger.info(f"[myenhance] Flask /api/list: found {len(scopes_items)} scopes.")
            
            for s_id, records in scopes_items:
                print(f"[myenhance-debug] Flask /api/list: scope {s_id} has {len(records)} records.")
                logger.info(f"[myenhance] Flask /api/list: scope {s_id} has {len(records)} records.")
                data["scopes"][s_id] = [
                    {"id": r.id, "content": r.content, "updated_at": r.updated_at}
                    for r in records
                ]
            return jsonify(data)
        except Exception as e:
            print(f"[myenhance-debug] Error in /api/list: {e}")
            logger.exception(f"[myenhance] Error in /api/list: {e}")
            return jsonify({"scopes": {}, "error": str(e)}), 500

    @app.route('/api/update', methods=['POST'])
    def api_update():
        data = request.json
        success = plugin_instance.memory_store.update_memory(
            data.get("scope"), data.get("id"), data.get("content")
        )
        return jsonify({"success": bool(success), "msg": "修改成功" if success else "修改失败"})

    @app.route('/api/delete', methods=['POST'])
    def api_delete():
        data = request.json
        success = plugin_instance.memory_store.delete_memory(
            data.get("scope"), data.get("id")
        )
        return jsonify({"success": success, "msg": "删除成功" if success else "删除失败"})

    def run():
        try:
            app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"[myenhance] Flask failed to start: {e}")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    logger.info(f"[myenhance] Flask UI started on port {port}")
    return thread
