agent/
│
├── main.py
│
├── agent/
│   ├── agent_loop.py
│   ├── reasoning_engine.py
│   ├── context_manager.py
│   ├── termination_checker.py
│   └── action_validator.py
│
├── tools/
│   ├── base_tool.py
│   ├── calculator_tool.py
│   ├── search_tool.py
│   ├── tool_registry.py
│   └── tool_executor.py
│
├── schemas/
│   ├── action_schema.py
│   ├── observation_schema.py
│   └── context_schema.py
│
├── prompts/
│   └── react_prompt.py
│
└── configs/
    └── agent_config.py
└──reflexion/
        ├── evaluation_schema.py
        ├── evaluator.py
        ├── reflection_schema.py
        ├── reflection_engine.py
        └── experience_memory.py