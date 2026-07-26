"""
V3 Planner Agent — Production Entry Point

Initialises every component of the Planner → Reason → Act → Evaluate → Reflect
agent framework and runs an interactive terminal session.

Usage:
    python main.py

Environment Variables:
    OLLAMA_MODEL            Model name           (default: qwen2.5-coder:7b)
    OLLAMA_BASE_URL         Ollama server URL     (default: http://localhost:11434)
    MAX_ITERATIONS          Agent loop ceiling    (default: 10)
    CONFIDENCE_THRESHOLD    Evaluator threshold   (default: 0.6)
    DEBUG                   Verbose output        (1 | true | yes)
    LOG_LEVEL               Python log level      (default: INFO, or DEBUG when DEBUG=1)
    LOG_FILE                Optional log file path
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

# ═══════════════════════════════════════════════════════════════════
# Windows: enable ANSI / VT100 escape-sequence processing
# ═══════════════════════════════════════════════════════════════════
if sys.platform == "win32":
    os.system("")  # lightweight VT100 activation
    try:                                # belt-and-suspenders for older builds
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetConsoleMode(k32.GetStdHandle(-11), 7)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════
# Project Imports
# ═══════════════════════════════════════════════════════════════════
from agent.llm_client import LLMClient
from agent.reasoning_engine import ReasoningEngine
import agent.action_validator as agent_action_validator
from agent.termination_checker import TerminationChecker
from agent.agent_loop import AgentLoop

from planner.linear.planner import Planner
from planner.linear.planner_prompt import PlannerPrompt
from planner.linear.planner_parser import PlannerParser

import tools.tool_registry
import tools.tool_executor
from tools.calculator_tool import CalculatorTool
from tools.search_tool import SearchTool
from tools.finish_tool import FinishTool

from reflexion.evaluator import Evaluator
from reflexion.reflection_engine import ReflectionEngine


# ═══════════════════════════════════════════════════════════════════
# Environment-Variable Helpers
# ═══════════════════════════════════════════════════════════════════

def _env_bool(key: str, default: bool = False) -> bool:
    """Read a boolean from the environment."""
    val = os.environ.get(key, "").strip().lower()
    return val in ("1", "true", "yes", "on") if val else default


def _env_int(key: str, default: int) -> int:
    """Read an integer from the environment."""
    val = os.environ.get(key, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _env_float(key: str, default: float) -> float:
    """Read a float from the environment."""
    val = os.environ.get(key, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# ═══════════════════════════════════════════════════════════════════
# Configuration  (resolved once at module load)
# ═══════════════════════════════════════════════════════════════════

OLLAMA_MODEL:          str   = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL:       str   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_ITERATIONS:        int   = _env_int("MAX_ITERATIONS", 10)
CONFIDENCE_THRESHOLD:  float = _env_float("CONFIDENCE_THRESHOLD", 0.6)
DEBUG:                 bool  = _env_bool("DEBUG")
LOG_LEVEL:             str   = os.environ.get("LOG_LEVEL", "DEBUG" if DEBUG else "INFO").upper()
LOG_FILE:              str   = os.environ.get("LOG_FILE", "")


# ═══════════════════════════════════════════════════════════════════
# ANSI Escape Codes
# ═══════════════════════════════════════════════════════════════════

class _C:
    """Terminal color constants."""
    RESET     = "\033[0m"
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    UNDERLINE = "\033[4m"

    RED       = "\033[91m"
    GREEN     = "\033[92m"
    YELLOW    = "\033[93m"
    BLUE      = "\033[94m"
    MAGENTA   = "\033[95m"
    CYAN      = "\033[96m"
    WHITE     = "\033[97m"
    GRAY      = "\033[90m"


# ═══════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════

class _ColorFormatter(logging.Formatter):
    """Adds ANSI color to log-level names."""

    _LEVEL_COLOR = {
        logging.DEBUG:    _C.GRAY,
        logging.INFO:     _C.CYAN,
        logging.WARNING:  _C.YELLOW,
        logging.ERROR:    _C.RED,
        logging.CRITICAL: _C.RED + _C.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLOR.get(record.levelno, _C.RESET)
        record.levelname = f"{color}{record.levelname:<8}{_C.RESET}"
        record.msg = f"{_C.DIM}{record.msg}{_C.RESET}"
        return super().format(record)


def _setup_logging() -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("v3_agent")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        _ColorFormatter(
            fmt="  %(asctime)s │ %(levelname)s │ %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console)

    # Optional file handler
    if LOG_FILE:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger


log = _setup_logging()


# ═══════════════════════════════════════════════════════════════════
# Experience Memory  (lightweight in-process store)
# ═══════════════════════════════════════════════════════════════════

class ExperienceMemory:
    """
    Stores reflections produced during agent runs and retrieves
    relevant lessons for future planning / reasoning steps.

    This is an intentionally simple keyword-overlap memory.
    Swap in a vector store for semantic retrieval later.
    """

    def __init__(self, max_entries: int = 50):
        self._entries: list[dict] = []
        self._max_entries = max_entries

    # ── write ────────────────────────────────────────────────
    def store(self, query: str, reflection) -> None:
        """Persist a single reflection."""
        self._entries.append(
            {
                "query": query,
                "mistake": reflection.mistake,
                "lesson": reflection.lesson,
                "recommendation": reflection.recommendation,
            }
        )
        # Evict oldest when over capacity
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    # ── read ─────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Return the most relevant past lessons as a formatted string."""
        if not self._entries:
            return ""

        query_tokens = set(query.lower().split())
        scored = []
        for entry in self._entries:
            overlap = len(query_tokens & set(entry["query"].lower().split()))
            scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        if not top or top[0][0] == 0:
            return ""

        lines: list[str] = []
        for _, entry in top:
            lines.append(
                f"- Query: {entry['query']}\n"
                f"  Lesson: {entry['lesson']}\n"
                f"  Recommendation: {entry['recommendation']}"
            )
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._entries)

    def __str__(self) -> str:
        return f"ExperienceMemory(entries={self.size})"

    def __repr__(self) -> str:
        return self.__str__()


# ═══════════════════════════════════════════════════════════════════
# Display Helpers
# ═══════════════════════════════════════════════════════════════════

def _banner() -> None:
    """Print the startup banner."""
    print(
        f"\n{_C.CYAN}{_C.BOLD}"
        f"╔══════════════════════════════════════════════════════════════╗\n"
        f"║                                                            ║\n"
        f"║          V3 Planner Agent  ·  ReAct + Reflexion            ║\n"
        f"║                                                            ║\n"
        f"╚══════════════════════════════════════════════════════════════╝"
        f"{_C.RESET}\n"
    )


def _print_config() -> None:
    """Print the active configuration block."""
    print(f"  {_C.DIM}Model:{_C.RESET}           {_C.WHITE}{OLLAMA_MODEL}{_C.RESET}")
    print(f"  {_C.DIM}Endpoint:{_C.RESET}        {_C.WHITE}{OLLAMA_BASE_URL}{_C.RESET}")
    print(f"  {_C.DIM}Max Iterations:{_C.RESET}  {_C.WHITE}{MAX_ITERATIONS}{_C.RESET}")
    print(f"  {_C.DIM}Confidence:{_C.RESET}      {_C.WHITE}{CONFIDENCE_THRESHOLD}{_C.RESET}")
    print(f"  {_C.DIM}Debug:{_C.RESET}           {_C.WHITE}{DEBUG}{_C.RESET}")
    if LOG_FILE:
        print(f"  {_C.DIM}Log File:{_C.RESET}        {_C.WHITE}{LOG_FILE}{_C.RESET}")
    print()


def _section(title: str, color: str = _C.CYAN) -> None:
    """Print a coloured section divider."""
    ruler = "─" * max(1, 50 - len(title))
    print(f"\n  {color}{_C.BOLD}{'─' * 3} {title} {ruler}{_C.RESET}")


def _kv(key: str, value, indent: int = 4) -> None:
    """Print a key: value line with consistent indentation."""
    print(f"{' ' * indent}{_C.DIM}{key}:{_C.RESET} {value}")


# ── Plan ─────────────────────────────────────────────────────

def _display_plan(agent: AgentLoop) -> None:
    """Show the generated plan (DEBUG mode only)."""
    if not DEBUG or agent.plan is None:
        return

    plan = agent.plan
    _section("PLAN", _C.MAGENTA)
    _kv("Goal", plan.goal)
    _kv("Status", plan.status)
    _kv("Success Criteria", plan.overall_success_criteria)

    if plan.planner_notes:
        _kv("Planner Notes", plan.planner_notes)

    for task in plan.tasks:
        print(
            f"\n    {_C.YELLOW}Task {task.task_id}:{_C.RESET} "
            f"{task.description}"
        )
        _kv("Capability", task.capability_required, indent=6)
        _kv("Expected Output", task.expected_output, indent=6)
        _kv("Completion", task.completion_criteria, indent=6)
        _kv("Priority", task.priority, indent=6)


# ── Full Trajectory ──────────────────────────────────────────

def _display_trajectory(context, agent: AgentLoop) -> None:
    """Render every Think → Act → Evaluate → Reflect step."""
    if not DEBUG:
        return

    _section("TRAJECTORY", _C.BLUE)
    _kv("Iterations", context.iteration_count)
    _kv("Actions Taken", len(context.action_history))
    _kv("Done", context.done)

    for i, action in enumerate(context.action_history):
        obs = (
            context.observation_history[i]
            if i < len(context.observation_history)
            else None
        )
        eval_result = (
            context.evaluation_history[i]
            if i < len(context.evaluation_history)
            else None
        )
        reflection = (
            context.reflection_histroy[i]        # NOTE: typo preserved from schema
            if i < len(context.reflection_histroy)
            else None
        )

        print(
            f"\n    {_C.BOLD}"
            f"┌─ Step {i + 1} "
            f"{'─' * 40}{_C.RESET}"
        )

        # ACTION
        print(f"    │ {_C.YELLOW}ACTION{_C.RESET}")
        print(f"    │   Tool:   {action.tool_name}")
        print(f"    │   Params: {action.parameters}")
        print(f"    │   Reason: {action.reason}")

        # OBSERVATION
        if obs is not None:
            marker = (
                f"{_C.GREEN}✓{_C.RESET}"
                if obs.success
                else f"{_C.RED}✗{_C.RESET}"
            )
            print(f"    │ {_C.CYAN}OBSERVATION{_C.RESET} {marker}")
            raw = str(obs.result or obs.error or "—")
            if len(raw) > 300:
                raw = raw[:300] + " …"
            for line in raw.splitlines():
                print(f"    │   {line}")

        # EVALUATION
        if eval_result is not None:
            print(f"    │ {_C.MAGENTA}EVALUATION{_C.RESET}")
            print(f"    │   Score:    {eval_result.progress_score:.2f}")
            print(f"    │   Feedback: {eval_result.feedback}")
            print(f"    │   Reflect?: {eval_result.should_reflect}")

        # REFLECTION
        if reflection is not None:
            print(f"    │ {_C.RED}REFLECTION{_C.RESET}")
            print(f"    │   Mistake:        {reflection.mistake}")
            print(f"    │   Lesson:         {reflection.lesson}")
            print(f"    │   Recommendation: {reflection.recommendation}")

        print(
            f"    {_C.BOLD}"
            f"└{'─' * 48}{_C.RESET}"
        )


# ── Final Answer ─────────────────────────────────────────────

def _extract_answer(context) -> str | None:
    """Pull the final answer from the trajectory."""
    # Prefer the observation produced by the "finish" tool
    for i, action in enumerate(context.action_history):
        if action.tool_name.lower() == "finish":
            if i < len(context.observation_history):
                obs = context.observation_history[i]
                if obs.success and obs.result:
                    return str(obs.result)

    # Fallback: last successful observation
    if context.observation_history:
        last = context.observation_history[-1]
        if last.success and last.result:
            return str(last.result)

    return None


def _display_answer(context) -> None:
    """Print the agent's answer in a highlighted block."""
    answer = _extract_answer(context)
    if answer:
        print(f"\n  {_C.GREEN}{_C.BOLD}Agent:{_C.RESET} {answer}")
    else:
        print(
            f"\n  {_C.YELLOW}Agent:{_C.RESET} "
            f"{_C.DIM}(No conclusive answer produced){_C.RESET}"
        )


# ═══════════════════════════════════════════════════════════════════
# Component Initialisation
# ═══════════════════════════════════════════════════════════════════

def _init_tools() -> tools.tool_registry.ToolRegistry:
    """Create the tool registry and register all available tools."""
    registry = tools.tool_registry.ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(SearchTool())
    registry.register_tool(FinishTool())
    log.info("Tools registered: %s", registry.list_tools())
    return registry


def _init_llm() -> LLMClient:
    """Create, health-check, and return the LLM client."""
    llm = LLMClient(
        model_name=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    log.info("Connecting to LLM: %s @ %s …", OLLAMA_MODEL, OLLAMA_BASE_URL)

    if not llm.health_check():
        log.critical(
            "Ollama is not reachable at %s.  Start it with:  ollama serve",
            OLLAMA_BASE_URL,
        )
        sys.exit(1)

    log.info("LLM connection verified ✓")
    return llm


def _build_agent(llm: LLMClient, registry: tools.tool_registry.ToolRegistry) -> AgentLoop:
    """Wire every component via constructor injection and return AgentLoop."""

    # ── Reasoning ────────────────────────────────────────────
    reasoning_engine = ReasoningEngine(
        llm_client=llm,
        tool_registry=registry,
    )
    log.debug("ReasoningEngine  → ready")

    # ── Validation & Execution ───────────────────────────────
    validator = agent_action_validator.ActionValidator(registry)
    executor = tools.tool_executor.ToolExecutor(registry)
    log.debug("ActionValidator   → ready")
    log.debug("ToolExecutor      → ready")

    # ── Termination ──────────────────────────────────────────
    termination_checker = TerminationChecker(
        max_iterations=MAX_ITERATIONS,
    )
    log.debug(
        "TerminationChecker → ready  (max_iterations=%d)",
        MAX_ITERATIONS,
    )

    # ── Planner ──────────────────────────────────────────────
    planner = Planner(
        llm_client=llm,
        planner_prompt=PlannerPrompt(),
        planner_parser=PlannerParser(),
    )
    log.debug("Planner           → ready")

    # ── Evaluator (rule-based + LLM fallback) ────────────────
    evaluator = Evaluator(
        llm_client=llm,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    log.debug(
        "Evaluator         → ready  (threshold=%.2f)",
        CONFIDENCE_THRESHOLD,
    )

    # ── Reflection Engine ────────────────────────────────────
    reflection_engine = ReflectionEngine(llm_client=llm)
    log.debug("ReflectionEngine  → ready")

    # ── Agent Loop (top-level orchestrator) ───────────────────
    agent_loop = AgentLoop(
        reasoning_engine=reasoning_engine,
        action_validator=validator,
        tool_executor=executor,
        termination_checker=termination_checker,
        evaluator=evaluator,
        reflection_engine=reflection_engine,
        planner=planner,
    )
    log.info("AgentLoop assembled ✓")

    return agent_loop


# ═══════════════════════════════════════════════════════════════════
# Graceful Shutdown via Signal Handling
# ═══════════════════════════════════════════════════════════════════

_shutdown_requested: bool = False


def _handle_interrupt(signum, frame):                     # noqa: ARG001
    """First Ctrl-C sets a flag; second Ctrl-C force-quits."""
    global _shutdown_requested
    if _shutdown_requested:
        print(f"\n{_C.RED}{_C.BOLD}  Force quit.{_C.RESET}")
        sys.exit(1)
    _shutdown_requested = True
    print(
        f"\n{_C.YELLOW}"
        f"  Interrupt received — type 'exit' or press Ctrl+C again to quit."
        f"{_C.RESET}"
    )


# ═══════════════════════════════════════════════════════════════════
# Interactive REPL
# ═══════════════════════════════════════════════════════════════════

_HELP_TEXT = f"""
  {_C.BOLD}Commands:{_C.RESET}
    exit / quit     Shut down the agent
    debug           Toggle verbose debug output
    clear           Clear the screen
    memory          Show experience memory stats
    help            Show this message
"""


def main() -> None:
    """Entry point: initialise components, then run the interactive loop."""
    global DEBUG, _shutdown_requested

    signal.signal(signal.SIGINT, _handle_interrupt)

    # ── Startup ──────────────────────────────────────────────
    _banner()
    _print_config()

    registry = _init_tools()
    llm      = _init_llm()
    agent    = _build_agent(llm, registry)
    memory   = ExperienceMemory()

    log.info("All components initialised — agent ready")

    print(
        f"  {_C.GREEN}{_C.BOLD}Agent Ready{_C.RESET}\n"
        f"  {_C.DIM}Type a query and press Enter.  "
        f"Type 'help' for commands.{_C.RESET}\n"
    )

    # ── Loop ─────────────────────────────────────────────────
    while True:
        if _shutdown_requested:
            break

        # ── Read input ───────────────────────────────────────
        try:
            query = input(f"  {_C.BOLD}{_C.WHITE}> {_C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue

        cmd = query.lower()

        # ── Meta commands ────────────────────────────────────
        if cmd in ("exit", "quit", "q"):
            break

        if cmd == "help":
            print(_HELP_TEXT)
            continue

        if cmd == "clear":
            os.system("cls" if sys.platform == "win32" else "clear")
            _banner()
            continue

        if cmd == "memory":
            if memory.size == 0:
                print(f"\n  {_C.DIM}Memory is empty.{_C.RESET}\n")
            else:
                print(f"\n  {_C.DIM}{memory}{_C.RESET}")
                for idx, entry in enumerate(memory._entries, 1):
                    print(
                        f"    {_C.GRAY}{idx}.{_C.RESET} "
                        f"{entry['lesson']}"
                    )
                print()
            continue

        if cmd == "debug":
            DEBUG = not DEBUG
            status = f"{_C.GREEN}ON{_C.RESET}" if DEBUG else f"{_C.RED}OFF{_C.RESET}"
            print(f"\n  Debug mode: {status}\n")
            continue

        # ── Process query ────────────────────────────────────
        log.info("Query: \"%s\"", query)
        t0 = time.perf_counter()

        try:
            # Reset plan so the planner generates a fresh one
            agent.plan = None

            # Retrieve relevant past lessons (if any)
            relevant_memory = memory.retrieve(query) or None

            # ── Run the full agent loop ──────────────────────
            context = agent.run(query)

            elapsed = time.perf_counter() - t0

            # ── Display results ──────────────────────────────
            _display_plan(agent)
            _display_trajectory(context, agent)
            _display_answer(context)

            # ── Persist reflections into memory ──────────────
            for reflection in context.reflection_histroy:
                memory.store(query, reflection)
                log.debug(
                    "Stored reflection: %s",
                    reflection.lesson,
                )

            if DEBUG:
                print(
                    f"\n  {_C.DIM}Completed in "
                    f"{elapsed:.2f}s  ·  "
                    f"{context.iteration_count} iteration(s)"
                    f"{_C.RESET}"
                )

            print()                                   # breathing room

        except KeyboardInterrupt:
            print(f"\n  {_C.YELLOW}Query interrupted.{_C.RESET}\n")
            _shutdown_requested = False               # allow another query
            continue

        except Exception:
            log.exception("Unhandled error during agent execution")
            print(
                f"\n  {_C.RED}An error occurred. "
                f"Check logs for details.{_C.RESET}\n"
            )

    # ── Shutdown ─────────────────────────────────────────────
    print(f"\n  {_C.CYAN}Goodbye.{_C.RESET}\n")
    log.info("Agent shutdown complete.")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

