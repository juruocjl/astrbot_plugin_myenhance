from __future__ import annotations

from astrbot.api.all import *
from .utils.memory_store import MemoryStore

def register_pages(plugin_instance: "MyPlugin"):
    """
    注册插件的 WebUI 页面
    """
    
    @plugin_instance.context.register_page(
        "memories",
        "记忆管理",
        "manage", 
        "manage-memories-page"
    )
    async def memories_page(context: AstrMessageEvent):
        """
        记忆管理主页面
        """
        # 获取所有会话的 ID
        scopes = list(plugin_instance.memory_store.memories_by_scope.keys())
        
        return {
            "type": "page",
            "body": [
                {
                    "type": "tpl",
                    "tpl": "<h3>记忆管理</h3><p>在此管理已保存的长期记忆。</p>",
                    "className": "mb-4"
                },
                {
                    "type": "tabs",
                    "tabs": [
                        {
                            "title": "已保存记忆",
                            "body": [
                                {
                                    "type": "form",
                                    "title": "搜索与过滤",
                                    "wrapWithPanel": False,
                                    "target": "memory_table",
                                    "body": [
                                        {
                                            "type": "group",
                                            "body": [
                                                {
                                                    "type": "select",
                                                    "label": "选择会话 (Scope ID)",
                                                    "name": "scope_id",
                                                    "options": [{"label": s, "value": s} for s in scopes],
                                                    "description": "如果不选则显示全部"
                                                }
                                            ]
                                        }
                                    ]
                                },
                                {
                                    "type": "service",
                                    "id": "memory_table_service",
                                    "api": "/api/plugins/myenhance/memories/list?scope_id=${scope_id}",
                                    "body": [
                                        {
                                            "type": "table",
                                            "name": "records",
                                            "id": "memory_table",
                                            "source": "${items}",
                                            "columns": [
                                                {"name": "id", "label": "ID", "width": 80},
                                                {"name": "scope", "label": "会话 ID"},
                                                {"name": "content", "label": "记忆内容"},
                                                {"name": "updated_at", "label": "更新时间", "width": 160},
                                                {
                                                    "type": "operation",
                                                    "label": "操作",
                                                    "width": 150,
                                                    "buttons": [
                                                        {
                                                            "type": "button",
                                                            "icon": "fa fa-pencil",
                                                            "actionType": "dialog",
                                                            "dialog": {
                                                                "title": "编辑记忆",
                                                                "body": {
                                                                    "type": "form",
                                                                    "api": "post:/api/plugins/myenhance/memories/update",
                                                                    "body": [
                                                                        {"type": "static", "name": "id", "label": "ID"},
                                                                        {"type": "static", "name": "scope", "label": "会话"},
                                                                        {"type": "textarea", "name": "content", "label": "内容", "required": True}
                                                                    ]
                                                                }
                                                            }
                                                        },
                                                        {
                                                            "type": "button",
                                                            "icon": "fa fa-times text-danger",
                                                            "actionType": "ajax",
                                                            "confirmText": "确定要删除这条记忆吗？",
                                                            "api": "post:/api/plugins/myenhance/memories/delete?id=${id}&scope=${scope}"
                                                        }
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    @plugin_instance.context.register_route("get", "/memories/list")
    async def list_memories(request):
        scope_id = request.query.get("scope_id")
        
        all_items = []
        for s_id, records in plugin_instance.memory_store.memories_by_scope.items():
            if scope_id and s_id != scope_id:
                continue
            for r in records:
                all_items.append({
                    "id": r.id,
                    "scope": s_id,
                    "content": r.content,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at
                })
        
        # 简单按更新时间降序
        all_items.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return {"items": all_items}

    @plugin_instance.context.register_route("post", "/memories/update")
    async def update_memory_route(request):
        data = await request.json()
        memory_id = data.get("id")
        scope = data.get("scope")
        content = data.get("content")
        
        if not all([memory_id, scope, content]):
            return {"status": 1, "msg": "缺少必要参数"}
            
        success = plugin_instance.memory_store.update_memory(scope, memory_id, content)
        if success:
            return {"status": 0, "msg": "修改成功"}
        return {"status": 1, "msg": "修改失败，记忆可能已被删除"}

    @plugin_instance.context.register_route("post", "/memories/delete")
    async def delete_memory_route(request):
        memory_id = request.query.get("id")
        scope = request.query.get("scope")
        
        if not all([memory_id, scope]):
            return {"status": 1, "msg": "缺少必要参数"}
            
        success = plugin_instance.memory_store.delete_memory(scope, memory_id)
        if success:
            return {"status": 0, "msg": "删除成功"}
        return {"status": 1, "msg": "删除失败，未找到对应记录"}
