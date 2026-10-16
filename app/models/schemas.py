"""
核心数据模型契约 (Data Models & Schemas)
职责：
1. 定义文档切片与父子分块的实体结构 (Parent & Child Chunks)。
2. 定义多路召回、RRF 融合与重排结果的数据传输对象 (DTO)。
3. 规范证据可溯源引用 (Citations) 与 LLM-as-a-judge 自动化评测指标。
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import uuid
import time


class RetrievalSource(str, Enum):
    """检索召回源枚举"""
    DENSE_VECTOR = "dense_vector"  # 向量稠密语义检索
    SPARSE_BM25 = "sparse_bm25"    # BM25 稀疏关键词检索
    HYBRID_RRF = "hybrid_rrf"      # RRF 融合结果
    RERANKED = "reranked"          # 交叉重排后结果


# =============================================================================
# 一、文档切片与父子层级模型
# =============================================================================

class BaseChunk(BaseModel):
    """基础切片模型"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="切片全局唯一ID")
    doc_id: str = Field(..., description="归属的源文档ID")
    doc_name: str = Field(..., description="源文档文件名 (如: 公司财务规范2024.pdf)")
    content: str = Field(..., description="切片文本内容")
    chunk_index: int = Field(default=0, description="在原文中的顺序索引")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="灵活扩展元数据(如页码、章节、ACL权限)")


class ChildChunk(BaseChunk):
    """子切片：粒度小(200~300字)，专用于高敏度向量命中与BM25索引"""
    parent_id: str = Field(..., description="关联的母切片ID")
    embedding: Optional[List[float]] = Field(default=None, description="稠密语义向量特征 (1024维)")


class ParentChunk(BaseChunk):
    """母切片：粒度大(1000~1500字)，保留完整段落与因果关系，用于生成时回填上下文"""
    child_ids: List[str] = Field(default_factory=list, description="所包含的所有子切片ID列表")


# =============================================================================
# 二、检索召回、融合与精排结果契约
# =============================================================================

class SearchResultItem(BaseModel):
    """单路检索候选返回项"""
    chunk_id: str
    doc_id: str
    doc_name: str
    parent_id: Optional[str] = None
    content: str = Field(..., description="子块或当前块的原文")
    parent_content: Optional[str] = Field(default=None, description="回填后的母块全文")
    
    score: float = Field(..., description="该阶段计算的综合得分")
    source: RetrievalSource = Field(..., description="当前得分所处的检索阶段")
    rank: int = Field(default=1, description="在当前检索结果中的排名次序")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryTransformationResult(BaseModel):
    """查询理解与改写结果"""
    original_query: str = Field(..., description="用户原始提问")
    rewritten_queries: List[str] = Field(default_factory=list, description="多查询扩展列表")
    hyde_passage: Optional[str] = Field(default=None, description="HyDE假设性回答")
    is_complex: bool = Field(default=False, description="是否为需要拆分的复杂复合问题")
    sub_queries: List[str] = Field(default_factory=list, description="分解后的子问题列表")


# =============================================================================
# 三、证据溯源与问答交互响应模型
# =============================================================================

class CitationReference(BaseModel):
    """给前端展示的段落证据卡片"""
    index: int = Field(..., description="引用序号 [1], [2]")
    doc_name: str = Field(..., description="来源文档名称")
    snippet: str = Field(..., description="命中关键证据摘录(高亮展示)")
    relevance_score: float = Field(..., description="重排后的真实置信度得分")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatQueryRequest(BaseModel):
    """用户问答输入请求契约"""
    query: str = Field(..., min_length=1, description="提问内容")
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    top_k: int = Field(default=5, ge=1, le=20, description="期望参考的上下文数量")
    enable_hyde: bool = Field(default=False, description="是否启用假设性文档改写")
    enable_rerank: bool = Field(default=True, description="是否启用 Cross-Encoder 重排")


class ChatQueryResponse(BaseModel):
    """问答最终输出结构 (非流式场景)"""
    session_id: str
    query: str
    answer: str = Field(..., description="模型最终合成回答")
    citations: List[CitationReference] = Field(default_factory=list, description="证据可溯源引用列表")
    latency_ms: float = Field(default=0.0, description="端到端全链路总耗时(毫秒)")
    steps_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="可解释性推理步骤时间线")


# =============================================================================
# 四、LLM-as-a-judge 评测指标模型 (RAG Triad)
# =============================================================================

class RagasEvaluationMetrics(BaseModel):
    """RAG 四维黄金评估指标 (取值均为 0.0 ~ 1.0)"""
    context_recall: float = Field(..., description="召回率：真实参考答案需要的证据是否找全了")
    context_precision: float = Field(..., description="精准度：最关键有用的证据是否排在最前列")
    faithfulness: float = Field(..., description="忠实度：生成的答案是否完全由检索材料支撑(防幻觉)")
    answer_relevancy: float = Field(..., description="相关度：生成的答案是否精准回答了用户的问题")
    overall_score: float = Field(..., description="四项综合调和平均分")
    evaluated_at: float = Field(default_factory=time.time)
