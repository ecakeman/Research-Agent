from __future__ import annotations

from enum import Enum

from app.config import Settings, settings as default_settings
from app.models.clients import HTTPLLMClient, LLMClient


class ModelRole(str, Enum):
    FAST = "fast"
    PRO = "pro"


class ModelRouter:
    def __init__(
        self,
        routing: str,
        fast: LLMClient,
        pro: LLMClient,
        *,
        fast_name: str,
        pro_name: str,
    ):
        if routing not in {"single", "dual"}:
            raise RuntimeError(f"MODEL_ROUTING 无效: {routing}")
        self.routing = routing
        self._fast = fast
        self._pro = pro
        self._fast_name = fast_name
        self._pro_name = pro_name

    def client(self, role: ModelRole) -> LLMClient:
        if self.routing == "single":
            return self._fast
        return self._pro if role is ModelRole.PRO else self._fast

    def model_name(self, role: ModelRole) -> str:
        if self.routing == "single":
            return self._fast_name
        return self._pro_name if role is ModelRole.PRO else self._fast_name

    @classmethod
    def from_single_client(cls, llm: LLMClient, name: str = "fake") -> ModelRouter:
        return cls("single", llm, llm, fast_name=name, pro_name=name)

    @classmethod
    def from_clients(
        cls,
        *,
        fast: LLMClient,
        pro: LLMClient,
        fast_name: str = "fast",
        pro_name: str = "pro",
        routing: str = "dual",
    ) -> ModelRouter:
        return cls(routing, fast, pro, fast_name=fast_name, pro_name=pro_name)

    @classmethod
    def from_settings(cls, cfg: Settings | None = None) -> ModelRouter:
        cfg = cfg or default_settings
        routing = (cfg.model_routing or "single").strip().lower()
        if routing not in {"single", "dual"}:
            raise RuntimeError(f"MODEL_ROUTING 必须是 single 或 dual，当前: {cfg.model_routing}")
        if routing == "single":
            if not cfg.llm_base_url or not cfg.llm_model:
                raise RuntimeError("Single 模式需要 LLM_BASE_URL 与 LLM_MODEL")
            client = HTTPLLMClient(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
            return cls("single", client, client, fast_name=cfg.llm_model, pro_name=cfg.llm_model)
        if not cfg.llm_fast_base_url or not cfg.llm_fast_model:
            raise RuntimeError("Dual 模式缺少 FAST 配置: LLM_FAST_BASE_URL / LLM_FAST_MODEL")
        if not cfg.llm_pro_base_url or not cfg.llm_pro_model:
            raise RuntimeError("Dual 模式缺少 PRO 配置: LLM_PRO_BASE_URL / LLM_PRO_MODEL")
        fast = HTTPLLMClient(cfg.llm_fast_base_url, cfg.llm_fast_api_key, cfg.llm_fast_model)
        pro = HTTPLLMClient(cfg.llm_pro_base_url, cfg.llm_pro_api_key, cfg.llm_pro_model)
        return cls(
            "dual",
            fast,
            pro,
            fast_name=cfg.llm_fast_model,
            pro_name=cfg.llm_pro_model,
        )


def answer_role(citation_attempts: int) -> ModelRole:
    """0=首次 Pro；1=Fast retry；>=2=Pro retry。"""
    if citation_attempts <= 0:
        return ModelRole.PRO
    if citation_attempts == 1:
        return ModelRole.FAST
    return ModelRole.PRO
