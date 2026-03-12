from __future__ import annotations
import threading
from flask import Flask, request, jsonify, render_template_string
from typing import TYPE_CHECKING
from astrbot.api import logger
from werkzeug.serving import make_server
from .utils.hybrid_retrieval import hybrid_search

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
            .search-panel { border-radius: 8px; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04); }
            .search-results .card { border-radius: 6px; }
            .search-result-score span { display: inline-flex; gap: 0.25rem; align-items: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="mb-4">MyEnhance 记忆管理</h2>
            
            <div class="card mb-3 search-panel">
                <div class="card-body">
                    <h5 class="card-title mb-3">检索记忆</h5>
                    <div class="row g-2 align-items-end">
                        <div class="col-md-3">
                            <label class="form-label">会话</label>
                            <select class="form-select" id="scopeSelect" disabled>
                                <option value="">暂无会话</option>
                            </select>
                        </div>
                        <div class="col-md-5">
                            <label class="form-label">查询</label>
                            <input type="text" class="form-control" id="searchQuery" placeholder="输入关键词或一句话描述">
                        </div>
                        <div class="col-md-2">
                            <label class="form-label">数量</label>
                            <input type="number" class="form-control" id="searchLimit" value="5" min="1" max="50">
                        </div>
                        <div class="col-md-2">
                            <button class="btn btn-primary w-100" id="search-button" onclick="searchMemories()">检索</button>
                        </div>
                    </div>
                </div>
            </div>
            <div id="search-results" class="mb-4 search-results"></div>

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

            let cachedScopes = {};

            function populateScopeSelect(scopes) {
                const scopeSelect = document.getElementById('scopeSelect');
                if (!scopeSelect) {
                    return;
                }
                cachedScopes = scopes || {};
                scopeSelect.innerHTML = '';
                const entries = Object.keys(cachedScopes);
                if (!entries.length) {
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = '暂无会话';
                    scopeSelect.appendChild(option);
                    scopeSelect.disabled = true;
                    return;
                }
                entries.forEach((scope) => {
                    const option = document.createElement('option');
                    option.value = scope;
                    option.textContent = scope;
                    scopeSelect.appendChild(option);
                });
                scopeSelect.disabled = false;
            }

            function renderSearchResults(results, meta = {}) {
                const container = document.getElementById('search-results');
                if (!container) {
                    return;
                }
                if (!results || !results.length) {
                    const message = meta.message || '暂无检索结果';
                    container.innerHTML = `<div class="alert alert-secondary">${escapeHtml(message)}</div>`;
                    return;
                }
                container.innerHTML = results
                    .map((result) => {
                        const safeContent = escapeHtml(result.content || '');
                        const keywordDisplay = result.keyword
                            ? escapeHtml(result.keyword)
                            : '<span class="text-muted">未设置</span>';
                        const safeScore = Number(result.score ?? 0).toFixed(6);
                        const safeBm25 = Number(result.bm25_score ?? 0).toFixed(6);
                        const safeEmbedding = Number(result.embedding_score ?? 0).toFixed(6);
                        return `
                            <div class="card">
                                <div class="card-body">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <p class="mb-1"><strong>关键词：</strong>${keywordDisplay}</p>
                                            <p class="mb-1 text-truncate">${safeContent}</p>
                                            <small class="text-muted">ID: ${escapeHtml(result.id || '')} | 更新: ${escapeHtml(result.updated_at || '')}</small>
                                        </div>
                                        <div class="text-end search-result-score">
                                            <div><strong>综合</strong> ${safeScore}</div>
                                            <div><small>bm25 ${safeBm25}</small></div>
                                            <div><small>embed ${safeEmbedding}</small></div>
                                        </div>
                                    </div>
                                </div>
                            </div>`;
                    })
                    .join('');
            }

            async function searchMemories() {
                const scopeSelect = document.getElementById('scopeSelect');
                const queryInput = document.getElementById('searchQuery');
                const limitInput = document.getElementById('searchLimit');
                const searchButton = document.getElementById('search-button');
                if (!scopeSelect || !queryInput || !searchButton) {
                    return;
                }
                const scope = scopeSelect.value;
                const query = (queryInput.value || '').trim();
                if (!scope) {
                    alert('请选择会话');
                    return;
                }
                if (!query) {
                    alert('请输入查询内容');
                    return;
                }
                let limit = parseInt(limitInput.value, 10);
                if (Number.isNaN(limit) || limit < 1) {
                    limit = 5;
                }
                limit = Math.min(limit, 50);
                searchButton.disabled = true;
                searchButton.textContent = '检索中...';
                const resultsContainer = document.getElementById('search-results');
                if (resultsContainer) {
                    resultsContainer.innerHTML = '<div class="text-center text-muted">检索中...</div>';
                }
                try {
                    const resp = await fetch('/api/search', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({scope, query, limit}),
                    });
                    const res = await resp.json();
                    if (!res.success) {
                        renderSearchResults([], {message: res.msg || '检索失败'});
                        return;
                    }
                    renderSearchResults(res.results || [], {
                        message: res.results && res.results.length
                            ? `查询「${query}」共 ${res.results.length} 条`
                            : `查询「${query}」未命中记忆`,
                    });
                } catch (err) {
                    if (resultsContainer) {
                        resultsContainer.innerHTML = `<div class="alert alert-danger">检索失败：${escapeHtml((err && err.message) || '网络错误')}</div>`;
                    }
                } finally {
                    searchButton.disabled = false;
                    searchButton.textContent = '检索';
                }
            }

            async function loadMemories() {
                const resp = await fetch('/api/list');
                const data = await resp.json();
                const content = document.getElementById('content');
                const scopeEntries = Object.keys(data.scopes);

                if (scopeEntries.length === 0) {
                    content.innerHTML = '<div class="alert alert-info">暂无记忆数据</div>';
                    populateScopeSelect({});
                    renderSearchResults([], { message: '暂无可检索的会话' });
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
                populateScopeSelect(data.scopes);
                renderSearchResults([], { message: '请选择会话并输入关键词' });
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

    @app.route('/api/search', methods=['POST'])
    def api_search():
        data = request.get_json(silent=True) or {}
        scope = str(data.get("scope") or "").strip()
        query = str(data.get("query") or "").strip()
        if not scope or not query:
            return jsonify({
                "success": False,
                "msg": "scope 和 query 为必填项",
                "results": [],
            })
        try:
            limit = int(data.get("limit", plugin_instance.memory_recall_count or 5))
        except (TypeError, ValueError):
            limit = plugin_instance.memory_recall_count or 5
        limit = max(1, min(limit, 50))
        records = plugin_instance.memory_store.list_memories(scope)
        if not records:
            return jsonify({
                "success": True,
                "msg": "当前会话暂无记忆",
                "results": [],
            })
        try:
            scored = hybrid_search(
                query,
                records,
                limit,
                rrf_k=getattr(plugin_instance, "rrf_k", 60),
            )
        except Exception as exc:
            logger.warning("[myenhance] search failed: %s", exc)
            return jsonify({
                "success": False,
                "msg": "检索失败，请稍后重试",
                "results": [],
            })
        return jsonify({
            "success": True,
            "msg": f"检索到 {len(scored)} 条记忆",
            "results": [
                {
                    "id": item.record.id,
                    "content": item.record.content,
                    "keyword": getattr(item.record, "keyword", ""),
                    "updated_at": item.record.updated_at,
                    "score": round(item.score, 6),
                    "bm25_score": round(item.bm25_score, 6),
                    "embedding_score": round(item.embedding_score, 6),
                }
                for item in scored
            ],
        })

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
