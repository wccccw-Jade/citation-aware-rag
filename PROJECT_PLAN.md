# Citation-Aware RAG 生产级项目计划书

## 1. 项目背景

当前项目已经实现了一个可运行的 Citation-Aware RAG baseline：支持 PDF/文本摄取、页面级元数据保留、递归切块、本地确定性 embedding、FAISS/NumPy 检索、chunk 级引用返回、命令行查询、Streamlit 演示和基础评估。

但当前实现更适合作为实验原型，距离生产级 RAG 系统仍有明显差距：

- embedding 使用本地 hash 方法，语义召回能力有限。
- 回答生成基于证据句抽取，没有真正的 LLM 生成和复杂推理能力。
- 索引构建是一次性离线流程，缺少增量更新、版本管理和数据质量控制。
- 检索、重排、引用校验、上下文压缩等核心 RAG 能力还比较基础。
- 缺少 API 服务、认证、监控、日志、评估闭环和部署方案。
- 缺少系统化测试、安全边界和生产运维设计。

本计划的目标是将该项目升级为一个可部署、可评估、可维护、可扩展的生产级 RAG 平台。

## 2. 项目目标

### 2.1 总体目标

构建一个面向学术文档和企业知识库的生产级 Citation-Aware RAG 系统，使用户能够上传、管理、检索文档，并获得有来源、有页码、有证据链、可审计的问答结果。

### 2.2 核心目标

- 支持稳定的文档摄取、解析、清洗、切块、索引和增量更新。
- 使用高质量 embedding 模型提升语义检索能力。
- 引入 hybrid retrieval、reranking、query rewriting 和上下文压缩。
- 使用 LLM 生成 grounded answer，并强制输出引用。
- 建立自动化评估体系，持续跟踪召回、引用准确性和回答质量。
- 提供生产级 API 服务和基础 Web UI。
- 增加日志、监控、错误追踪、成本统计和权限控制。
- 支持容器化部署和环境配置管理。

## 3. 目标用户与使用场景

### 3.1 目标用户

- 研究人员：对论文集合进行问答和证据追踪。
- 企业知识库用户：对内部文档、报告、规范进行问答。
- 开发人员：将 RAG 能力作为 API 集成到其他系统。
- 管理员：管理文档、索引、模型配置和评估报告。

### 3.2 典型场景

- 用户上传一批 PDF，系统自动解析、切块、生成 embedding 并建立索引。
- 用户提出问题，系统返回简洁答案，并附带引用来源、页码和原文片段。
- 用户查看某个答案的证据链，判断回答是否可信。
- 管理员周期性运行评估集，检查新模型或新切块策略是否提升质量。
- 系统接入企业内部应用，通过 API 提供文档问答能力。

## 4. 生产级能力范围

### 4.1 文档摄取与处理

当前能力：

- 支持 PDF、txt、md。
- PDF 按页解析。
- 保留文件路径、标题、页码等基础元数据。

计划升级：

- 支持批量上传和增量摄取。
- 为每个文档建立稳定 document_id 和版本号。
- 支持重复文档检测、文件 hash、解析状态追踪。
- 增强 PDF 解析，支持标题、章节、表格、图片 OCR 的扩展接口。
- 增加文档清洗流程，包括页眉页脚去除、参考文献识别、乱码过滤。
- 保存原始文档、解析文本、切块结果和索引版本之间的映射关系。

### 4.2 Chunking 策略

当前能力：

- 基于段落或句子切分。
- 支持固定长度和 overlap。

计划升级：

- 增加语义切块和结构化切块。
- 优先按章节、标题、段落、表格边界切分。
- 支持多粒度 chunk：小 chunk 用于检索，大上下文块用于生成。
- 保存 chunk 层级关系，例如 section、page、parent_chunk。
- 对不同文档类型配置不同切块策略。
- 建立切块策略评估机制，比较不同 chunk size、overlap 和结构化策略的效果。

### 4.3 Embedding 与向量存储

当前能力：

- 使用本地 hash embedding。
- 支持 FAISS 或 NumPy fallback。

计划升级：

- 引入生产级 embedding 模型。
- 支持可配置 embedding provider，例如 OpenAI、sentence-transformers、本地部署模型。
- 保存 embedding model、维度、生成时间和索引版本。
- 支持批量 embedding、失败重试、速率限制和成本统计。
- 将向量存储抽象为接口，支持 FAISS、Qdrant、Milvus、Weaviate、pgvector 等后端。
- 支持索引重建、索引迁移、索引回滚和多租户隔离。

### 4.4 Retrieval 检索

当前能力：

- dense retrieval + BM25 风格 lexical retrieval。
- min-max 归一化后加权融合。

计划升级：

- 建立标准 hybrid retrieval pipeline。
- 支持 query rewriting、query expansion 和多查询检索。
- 引入 metadata filter，例如文件、时间、作者、标签、页码范围。
- 引入 cross-encoder 或 LLM reranker。
- 增加 MMR 去重，减少重复 chunk。
- 增加 context packing，把检索结果组织成适合 LLM 的上下文。
- 支持 citation-aware retrieval，优先保留来源完整、页码可靠的证据。

### 4.5 Answer Generation 生成

当前能力：

- 从检索 chunk 中抽取证据句。
- 返回引用列表。

计划升级：

- 接入 LLM 生成答案。
- 使用严格提示词要求答案必须基于检索上下文。
- 输出结构化结果：answer、citations、confidence、supporting_quotes、limitations。
- 对每个引用绑定 chunk_id、page_number、source_path 和原文证据。
- 增加引用校验，确保答案中的引用确实来自对应 chunk。
- 对无足够证据的问题返回“不知道”或“当前文档中没有找到依据”。
- 支持中英文问答和跨语言检索。

### 4.6 Evaluation 评估体系

当前能力：

- 使用 `qa_eval.json` 运行离线评估。
- 统计 Hit@1、Hit@3、Hit@5、MRR、关键词命中。

计划升级：

- 建立多层评估指标：
  - Retrieval：Recall@k、MRR、nDCG、gold page hit、gold chunk hit。
  - Generation：faithfulness、answer relevance、context precision、citation accuracy。
  - System：latency、cost、error rate、timeout rate。
- 建立固定回归评估集。
- 增加人工标注流程和 bad case 分析模板。
- 对模型、检索参数、chunking 策略做 A/B 对比。
- 将评估结果保存为可追踪报告。
- 在 CI 或发布流程中加入最小质量门槛。

### 4.7 API 与应用层

当前能力：

- Streamlit demo。
- CLI 脚本。

计划升级：

- 使用 FastAPI 提供生产级 HTTP API。
- API 能力包括：
  - 上传文档。
  - 查询文档状态。
  - 构建或刷新索引。
  - 发起问答。
  - 获取引用和原文证据。
  - 删除文档或重建索引。
- 保留 Streamlit 作为内部 demo 或迁移到正式前端。
- 增加用户、项目、知识库三个层级的资源模型。
- 支持异步任务队列处理文档摄取和索引构建。

### 4.8 数据库与任务系统

计划新增：

- PostgreSQL 保存文档、chunk、任务、用户、评估结果和审计日志。
- Redis 用于缓存和任务队列。
- Celery、RQ 或 Dramatiq 处理异步任务。
- 对文档解析、embedding、索引构建设置任务状态和失败重试。

### 4.9 安全与权限

计划新增：

- API key 或 OAuth/JWT 鉴权。
- 用户级、知识库级权限控制。
- 上传文件类型和大小限制。
- 文档内容访问控制。
- Prompt injection 基础防护。
- 日志脱敏，避免泄露敏感文档内容。
- 对外部模型调用增加数据合规开关。

### 4.10 可观测性与运维

计划新增：

- 结构化日志。
- 查询链路 trace，包括 retrieval、rerank、generation 每一步耗时。
- 监控指标：
  - 请求量、延迟、错误率。
  - embedding 和 LLM 调用成本。
  - 检索 top-k 分数分布。
  - 无答案率和低置信度率。
- 错误追踪和告警。
- 管理端查看索引状态、任务状态和评估结果。

## 5. 推荐技术架构

### 5.1 后端服务

- Python 3.11+
- FastAPI
- Pydantic v2
- SQLAlchemy 或 SQLModel
- PostgreSQL
- Redis
- Celery/RQ/Dramatiq

### 5.2 RAG 核心

- Embedding provider 抽象层
- Vector store 抽象层
- Retriever pipeline
- Reranker provider
- LLM provider
- Citation validator
- Evaluation runner

### 5.3 向量数据库

短期建议：

- 本地开发继续支持 FAISS。
- 生产环境优先使用 Qdrant 或 pgvector。

选择建议：

- 如果重点是快速落地和运维简单：pgvector。
- 如果重点是向量检索性能和过滤能力：Qdrant。
- 如果已有大规模向量基础设施：Milvus 或 Weaviate。

### 5.4 前端

短期：

- 保留 Streamlit 作为内部调试界面。

中期：

- 建立正式 Web UI，支持文档管理、问答、引用查看和评估报告。

### 5.5 部署

- Dockerfile
- docker-compose 本地部署
- 环境变量配置
- 数据卷持久化
- 后续支持 Kubernetes 部署

## 6. 里程碑计划

### Phase 0：基线整理与工程化准备

目标：

- 梳理当前代码结构。
- 补齐基础测试。
- 明确模块边界和配置方式。

任务：

- 整理 README 和开发文档。
- 为 pipeline、chunking、retrieval、generator 增加单元测试。
- 增加 lint、format 和测试命令。
- 统一配置读取和路径管理。
- 记录当前 baseline 的评估结果。

交付物：

- 工程化后的 baseline。
- 初始评估报告。
- 可重复运行的测试命令。

验收标准：

- 新开发者可以根据 README 在本地完成建索引、查询和评估。
- 核心模块有基础测试覆盖。
- 当前评估结果可复现。

### Phase 1：生产级检索核心升级

目标：

- 替换 hash embedding。
- 抽象 embedding 和 vector store。
- 提升检索质量。

任务：

- 设计 `EmbeddingProvider` 接口。
- 设计 `VectorStore` 接口。
- 接入一个生产级 embedding 模型。
- 支持 FAISS + 至少一个生产向量库。
- 实现 metadata filter。
- 增加 MMR 去重。
- 增加 reranker 接口和一个默认实现。
- 扩展评估指标。

交付物：

- 可配置 embedding provider。
- 可配置 vector store。
- hybrid retrieval + rerank pipeline。
- 检索评估报告。

验收标准：

- Retrieval 指标相比 baseline 明显提升。
- 索引中记录 embedding model 和索引版本。
- 可以通过配置切换本地开发和生产向量库。

### Phase 2：LLM 生成与引用可信度

目标：

- 引入真正的 LLM answer generation。
- 保证回答 grounded，并可追踪引用。

任务：

- 设计 `LLMProvider` 接口。
- 编写 citation-aware prompt。
- 输出结构化 answer schema。
- 实现 citation validator。
- 实现证据不足时的拒答策略。
- 增加 prompt injection 基础防护。
- 增加 generation 评估指标。

交付物：

- LLM 生成模块。
- 引用校验模块。
- Faithfulness 和 citation accuracy 评估结果。

验收标准：

- 回答必须包含引用。
- 引用必须能映射到具体 chunk 和页码。
- 对无证据问题能够拒答。
- 评估集上的引用准确率达到预设门槛。

### Phase 3：API 服务与异步任务

目标：

- 从脚本和 demo 升级为可集成服务。

任务：

- 建立 FastAPI 服务。
- 增加文档上传 API。
- 增加问答 API。
- 增加索引任务 API。
- 引入 PostgreSQL 保存文档和任务状态。
- 引入 Redis + 任务队列处理解析、embedding、索引更新。
- 增加 API 错误模型和请求日志。

交付物：

- FastAPI 后端服务。
- 异步文档处理任务。
- 数据库 schema。
- API 文档。

验收标准：

- 可以通过 API 完成上传、索引、查询闭环。
- 长任务不阻塞请求线程。
- 任务失败可追踪、可重试。

### Phase 4：UI、可观测性与运维

目标：

- 提供更完整的用户操作界面和生产运维能力。

任务：

- 扩展或重建 Web UI。
- 增加文档列表、状态、删除和重新索引。
- 增加问答页面、引用展开和原文预览。
- 增加结构化日志和 trace。
- 增加监控指标。
- 增加 Dockerfile 和 docker-compose。
- 增加基础鉴权。

交付物：

- 可用 Web UI。
- 本地容器化部署方案。
- 监控和日志方案。

验收标准：

- 非开发用户可以通过 UI 完成文档问答。
- 系统运行状态可以被观测。
- 本地或测试环境可通过 docker-compose 启动。

### Phase 5：质量优化与发布准备

目标：

- 做生产发布前的质量收敛。

任务：

- 扩充评估集。
- 做 bad case 分析。
- 优化 chunking、retrieval、rerank、prompt。
- 增加负载测试。
- 增加安全测试。
- 编写部署文档和运维手册。
- 定义发布检查清单。

交付物：

- 发布候选版本。
- 完整评估报告。
- 部署和运维文档。

验收标准：

- 关键质量指标达到预设门槛。
- 常见失败场景有明确处理策略。
- 系统可以稳定部署到目标环境。

## 7. 推荐目录结构

```text
citation-aware-rag/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── security.py
│   ├── ingestion/
│   │   ├── parser.py
│   │   ├── cleaner.py
│   │   └── chunking.py
│   ├── embeddings/
│   │   ├── base.py
│   │   └── providers.py
│   ├── vectorstores/
│   │   ├── base.py
│   │   ├── faiss_store.py
│   │   └── qdrant_store.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── context_builder.py
│   ├── generation/
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   └── citation_validator.py
│   ├── evaluation/
│   │   ├── runner.py
│   │   └── metrics.py
│   ├── db/
│   │   ├── models.py
│   │   └── session.py
│   └── workers/
│       └── tasks.py
├── tests/
├── scripts/
├── docs/
├── data/
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## 8. 数据模型草案

### Document

- id
- tenant_id
- file_name
- file_hash
- source_path
- content_type
- status
- version
- created_at
- updated_at

### Chunk

- id
- document_id
- chunk_index
- text
- page_number
- section_title
- start_char
- end_char
- metadata
- embedding_model
- index_version

### QueryLog

- id
- user_id
- query
- rewritten_query
- retrieved_chunk_ids
- answer
- citations
- latency_ms
- model_name
- cost
- created_at

### IngestionTask

- id
- document_id
- status
- error_message
- started_at
- finished_at

## 9. 关键指标

### Retrieval 指标

- Recall@5
- Recall@10
- MRR
- nDCG@10
- Gold page hit rate
- Duplicate chunk rate

### Generation 指标

- Answer relevance
- Faithfulness
- Citation accuracy
- Unsupported claim rate
- Refusal accuracy

### 系统指标

- P50/P95/P99 latency
- Error rate
- Timeout rate
- Token usage
- Embedding cost
- LLM cost
- Index build duration

## 10. 风险与应对

### 风险 1：PDF 解析质量不稳定

应对：

- 保留解析后的中间文本，方便排查。
- 引入更强 PDF parser 和 OCR 扩展。
- 对低质量页面打标，避免污染索引。

### 风险 2：回答出现幻觉或引用不真实

应对：

- 强制结构化输出。
- 引用必须绑定 chunk_id。
- 对答案做 citation validator。
- 对无证据问题明确拒答。

### 风险 3：检索召回不足

应对：

- hybrid retrieval。
- query rewriting。
- reranking。
- 多粒度 chunk。
- 持续评估和 bad case 分析。

### 风险 4：成本和延迟过高

应对：

- 缓存 embedding 和查询结果。
- 控制 top-k 和上下文长度。
- 对 reranker 和 LLM 设置超时。
- 根据场景选择不同模型档位。

### 风险 5：生产环境数据安全

应对：

- 权限控制。
- 日志脱敏。
- 私有化模型选项。
- 明确外部 API 调用的数据边界。

## 11. 优先级建议

最高优先级：

- 替换 embedding。
- 抽象 vector store。
- 增强 retrieval 和 rerank。
- 接入 LLM 生成。
- 引用校验。
- 评估体系。

中优先级：

- FastAPI 服务。
- 异步任务。
- 数据库。
- 容器化。
- 监控日志。

后续增强：

- 多租户。
- 高级权限。
- OCR。
- 表格问答。
- 正式前端。
- Kubernetes 部署。

## 12. 第一阶段建议任务拆分

建议先从 Phase 0 和 Phase 1 开始，不直接大规模重构。

第一批任务：

1. 增加测试目录和基础测试。
2. 记录当前 baseline 评估结果。
3. 抽象 `EmbeddingProvider`。
4. 抽象 `VectorStore`。
5. 接入生产 embedding 模型。
6. 保留当前 hash embedding 作为本地 fallback。
7. 扩展 retrieval evaluation。
8. 比较 hash embedding 与新 embedding 的召回效果。

这样可以先把质量提升的核心路径打通，同时避免过早引入 API、数据库和任务队列导致复杂度上升。
