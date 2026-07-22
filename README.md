# CodeWeave Agent（码织 Agent）

CodeWeave Agent 是一个面向终端开发场景的 AI 编程助手，提供模型接入、流式对话、工具调用、文件操作和后续任务协作能力。项目以清晰的模块边界和可测试的基础设施为重点，便于在真实开发场景中持续扩展。

## 当前版本

当前主线已经具备以下能力：

- 项目基础：Python 包结构、配置加载、环境校验和命令入口
- 模型接入：OpenAI、Anthropic 与 OpenAI-compatible Chat Completions 适配器
- 工具系统：工具协议、注册表、参数校验，以及文件读取、搜索、编辑和 diff 预览
- 基础运行约束：统一的消息、流式事件、工具调用和 token 用量模型
- Agent 执行循环：流式事件转发、多轮工具调用、错误回传和最大轮次控制
- System Prompt：组合系统指令、工具说明、工作区、用户任务和运行时上下文

## 后续规划

- 权限控制、MCP 集成和上下文管理
- 会话持久化与记忆能力
- Slash Command、Skill 和 Hook 扩展机制
- SubAgent 任务分发与执行状态管理
- Git Worktree 隔离开发
- Agent Teams 协作与结果汇总

## 本地开发

在仓库根目录执行：

```powershell
python -m pytest --basetemp=.pytest-tmp
python -m codeweave --help
```

也可以直接查看命令版本：

```powershell
python -m codeweave --version
```

## 配置变量

项目使用以下环境变量配置运行时行为：

- `CODEWEAVE_PROVIDER_MODE`
- `CODEWEAVE_MODEL`
- `CODEWEAVE_API_KEY`
- `CODEWEAVE_BASE_URL`
- `CODEWEAVE_PERMISSION_MODE`
- `CODEWEAVE_MAX_TURNS`
- `CODEWEAVE_MAX_TOKENS`
- `CODEWEAVE_WORKSPACE`

默认权限模式为确认模式，避免工具调用在未确认的情况下产生文件写入或其他副作用。
