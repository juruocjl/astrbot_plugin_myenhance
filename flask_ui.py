from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from astrbot.api import logger
from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file
from werkzeug.serving import make_server

from .utils.hybrid_retrieval import hybrid_search

if TYPE_CHECKING:
    from .main import MyPlugin


def _load_html_template() -> str:
    template_path = Path(__file__).parent / "assets" / "flask_ui.html"
    if not template_path.exists():
        return "<html><body><h3>Template not found: assets/flask_ui.html</h3></body></html>"
    try:
        return template_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("[myenhance] failed to load flask_ui.html: %s", exc)
        return "<html><body><h3>Failed to load UI template</h3></body></html>"


def start_flask_app(plugin_instance: "MyPlugin", port: int):
    app = Flask(__name__)
    html_template = _load_html_template()

    @app.route("/")
    def index():
        return redirect("/jargon")

    @app.route("/jargon")
    def jargon_page():
        return render_template_string(html_template, ui_mode="jargon")

    @app.route("/memory")
    def memory_page():
        return render_template_string(html_template, ui_mode="memory")

    @app.route("/image")
    def image_page():
        return render_template_string(html_template, ui_mode="image")

    @app.route("/meme")
    def meme_page():
        return render_template_string(html_template, ui_mode="meme")

    @app.route("/api/list")
    def api_list():
        data = {"scopes": {}}
        for scope_id, records in list(plugin_instance.jargon_store.jargons_by_scope.items()):
            data["scopes"][scope_id] = [
                {
                    "id": record.id,
                    "content": record.content,
                    "keyword": getattr(record, "keyword", ""),
                    "updated_at": record.updated_at,
                }
                for record in records
            ]
        return jsonify(data)

    @app.route("/api/update", methods=["POST"])
    def api_update():
        data = request.get_json(silent=True) or {}
        record = plugin_instance.jargon_store.update_jargon(
            data.get("scope"),
            data.get("id"),
            data.get("content"),
            data.get("keyword"),
        )
        return jsonify({"success": bool(record), "msg": "Updated" if record else "Update failed"})

    @app.route("/api/search", methods=["POST"])
    def api_search():
        data = request.get_json(silent=True) or {}
        scope = str(data.get("scope") or "").strip()
        query = str(data.get("query") or "").strip()
        if not scope or not query:
            return jsonify({"success": False, "msg": "scope and query are required", "results": []})

        try:
            limit = int(data.get("limit", plugin_instance.jargon_total_recall_count or 5))
        except (TypeError, ValueError):
            limit = plugin_instance.jargon_total_recall_count or 5
        limit = max(1, min(limit, 50))

        records = plugin_instance.jargon_store.list_jargons(scope)
        if not records:
            return jsonify({"success": True, "msg": "No jargon in this scope", "results": []})

        try:
            embedding_scores = asyncio.run(plugin_instance._build_embedding_scores(query, records))
            scored = hybrid_search(
                query,
                records,
                limit,
                bm25_weight=getattr(plugin_instance, "bm25_weight", 0.55),
                embedding_weight=getattr(plugin_instance, "embedding_weight", 0.45),
                embedding_scores=embedding_scores,
                rrf_k=getattr(plugin_instance, "rrf_k", 60),
            )
        except Exception as exc:
            logger.warning("[myenhance] jargon search failed: %s", exc)
            return jsonify({"success": False, "msg": "Search failed", "results": []})

        return jsonify(
            {
                "success": True,
                "msg": f"Found {len(scored)} jargon items",
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
            }
        )

    @app.route("/api/delete", methods=["POST"])
    def api_delete():
        data = request.get_json(silent=True) or {}
        success = plugin_instance.jargon_store.delete_jargon(data.get("scope"), data.get("id"))
        return jsonify({"success": success, "msg": "Deleted" if success else "Delete failed"})

    @app.route("/api/memory/list")
    def api_memory_list():
        data = {"scopes": {}}
        for scope_id, records in list(plugin_instance.memory_store.memories_by_scope.items()):
            data["scopes"][scope_id] = [
                {
                    "id": record.id,
                    "content": record.content,
                    "keyword": getattr(record, "keyword", ""),
                    "updated_at": record.updated_at,
                }
                for record in records
            ]
        return jsonify(data)

    @app.route("/api/memory/add", methods=["POST"])
    def api_memory_add():
        data = request.get_json(silent=True) or {}
        scope = str(data.get("scope") or "").strip()
        content = str(data.get("content") or "").strip()
        keyword = str(data.get("keyword") or "").strip() or content[:80]
        if not scope or not content:
            return jsonify({"success": False, "msg": "scope and content are required"})
        try:
            plugin_instance.memory_store.add_memory(scope, content, keyword=keyword)
        except ValueError as exc:
            return jsonify({"success": False, "msg": f"Add failed: {exc}"})
        return jsonify({"success": True, "msg": "Added"})

    @app.route("/api/memory/update", methods=["POST"])
    def api_memory_update():
        data = request.get_json(silent=True) or {}
        scope = str(data.get("scope") or "").strip()
        memory_id = str(data.get("id") or "").strip()
        content = data.get("content")
        keyword = data.get("keyword")
        if content is not None:
            content = str(content).strip()
        if keyword is not None:
            keyword = str(keyword).strip()
        record = plugin_instance.memory_store.update_memory(scope, memory_id, content=content, keyword=keyword)
        return jsonify({"success": bool(record), "msg": "Updated" if record else "Update failed"})

    @app.route("/api/memory/delete", methods=["POST"])
    def api_memory_delete():
        data = request.get_json(silent=True) or {}
        scope = str(data.get("scope") or "").strip()
        memory_id = str(data.get("id") or "").strip()
        success = plugin_instance.memory_store.delete_memory(scope, memory_id)
        return jsonify({"success": success, "msg": "Deleted" if success else "Delete failed"})

    @app.route("/api/image/list")
    def api_image_list():
        records = []
        for image_id, raw_entry in reversed(plugin_instance.image_url_lru.items()):
            if not isinstance(raw_entry, dict):
                continue
            records.append(
                {
                    "id": str(image_id or ""),
                    "url": str(raw_entry.get("url") or ""),
                    "keyword": str(raw_entry.get("keyword") or ""),
                    "content": str(raw_entry.get("content") or ""),
                    "has_image": bool(raw_entry.get("local_path")),
                    "has_thumb": bool(raw_entry.get("thumb_path")),
                }
            )
        return jsonify({"count": len(records), "records": records})

    @app.route("/api/image/file/<image_id>")
    def api_image_file(image_id: str):
        key = str(image_id or "").strip()
        if not key:
            abort(404)
        raw_entry = plugin_instance.image_url_lru.get(key)
        if not isinstance(raw_entry, dict):
            abort(404)

        use_thumb = str(request.args.get("thumb") or "").strip() in {"1", "true", "True"}
        path_key = "thumb_path" if use_thumb else "local_path"
        file_path = Path(str(raw_entry.get(path_key) or "").strip())
        if file_path.exists() and file_path.is_file():
            return send_file(file_path)

        # 小图不存在时回退原图。
        if use_thumb:
            fallback = Path(str(raw_entry.get("local_path") or "").strip())
            if fallback.exists() and fallback.is_file():
                return send_file(fallback)

        abort(404)

    @app.route("/api/meme/list")
    def api_meme_list():
        records: dict[str, list[dict]] = {}
        for tag in plugin_instance.meme_manager.list_tags():
            image_ids = list(plugin_instance.meme_manager.memes_by_tag.get(tag) or [])
            tag_items = []
            for image_id in image_ids:
                image_entry = plugin_instance.image_manager.get_entry(image_id, plugin_instance.image_url_lru) or {}
                tag_items.append(
                    {
                        "id": image_id,
                        "keyword": str(image_entry.get("keyword") or ""),
                        "content": str(image_entry.get("content") or ""),
                        "url": str(image_entry.get("url") or ""),
                        "has_image": bool(image_entry.get("local_path")),
                        "has_thumb": bool(image_entry.get("thumb_path")),
                    }
                )
            records[tag] = tag_items
        return jsonify({"tags": records})

    @app.route("/api/meme/add", methods=["POST"])
    def api_meme_add():
        data = request.get_json(silent=True) or {}
        image_id = str(data.get("id") or "").strip()
        tag = str(data.get("tag") or "").strip()
        if not image_id or not tag:
            return jsonify({"success": False, "msg": "id and tag are required"})
        if plugin_instance.image_manager.get_entry(image_id, plugin_instance.image_url_lru) is None:
            return jsonify({"success": False, "msg": f"image_id not found: {image_id}"})
        ok, msg = plugin_instance.meme_manager.add_meme(image_id, tag)
        return jsonify({"success": ok, "msg": msg})

    @app.route("/api/meme/delete", methods=["POST"])
    def api_meme_delete():
        data = request.get_json(silent=True) or {}
        image_id = str(data.get("id") or "").strip()
        tag = str(data.get("tag") or "").strip()
        ok, msg = plugin_instance.meme_manager.delete_meme(image_id, tag)
        return jsonify({"success": ok, "msg": msg})

    server = make_server("0.0.0.0", port, app)
    plugin_instance._flask_server = server

    def run():
        try:
            server.serve_forever()
        except Exception as exc:
            logger.error("[myenhance] Flask failed to start: %s", exc)
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
        server_ref = getattr(plugin_instance, "_flask_server", None)
        thread_ref = getattr(plugin_instance, "_flask_thread", None)
        try:
            if server_ref is not None:
                server_ref.shutdown()
        except Exception as exc:
            logger.warning("[myenhance] Flask stop failed: %s", exc)
        finally:
            if thread_ref is not None and thread_ref.is_alive():
                thread_ref.join(timeout=2)
            plugin_instance._flask_server = None
            plugin_instance._flask_thread = None
            logger.info("[myenhance] Flask shutdown state cleared")

    logger.info("[myenhance] Flask UI started on port %s", port)
    return stop_flask
