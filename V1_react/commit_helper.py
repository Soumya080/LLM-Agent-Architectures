import os
import subprocess
import sys
import argparse

# Define the commits per day mapping
# Paths are relative to the repository root (d:\LLM-RESEARCH-LAB\AGENTS\V1_react)
COMMITS_BY_DAY = {
    1: [
        {
            "files": ["readme.md"],
            "message": "docs(react-agent-v1): add initial comprehensive project readme"
        },
        {
            "files": ["tools/__init__.py", "tools/base_tool.py"],
            "message": "feat(react-agent-v1): define base tool interface class"
        }
    ],
    2: [
        {
            "files": ["schemas/__init__.py", "schemas/context_schema.py"],
            "message": "feat(react-agent-v1): add context schema for state representation"
        },
        {
            "files": ["schemas/action_schema.py"],
            "message": "feat(react-agent-v1): add action schema and definition"
        },
        {
            "files": ["schemas/observation_schema.py"],
            "message": "feat(react-agent-v1): add observation schema for tool results"
        }
    ],
    3: [
        {
            "files": ["agent/__init__.py", "agent/llm_client.py"],
            "message": "feat(react-agent-v1): introduce llm client connector"
        },
        {
            "files": ["agent/action_validator.py"],
            "message": "feat(react-agent-v1): add action validator tool processor"
        },
        {
            "files": ["agent/termination_checker.py"],
            "message": "feat(react-agent-v1): add termination checker utility"
        }
    ],
    4: [
        {
            "files": ["tools/tool_registry.py"],
            "message": "feat(react-agent-v1): add dynamic tool registry"
        },
        {
            "files": ["tools/tool_executor.py"],
            "message": "feat(react-agent-v1): add tool execution manager"
        },
        {
            "files": ["tools/calculator_tool.py"],
            "message": "feat(react-agent-v1): implement calculator tool"
        },
        {
            "files": ["tools/search_tool.py", "tools/finish_tool.py"],
            "message": "feat(react-agent-v1): implement web search and finish tools"
        }
    ],
    5: [
        {
            "files": ["agent/reasoning_engine.py"],
            "message": "feat(react-agent-v1): implement main agent reasoning engine"
        },
        {
            "files": ["agent/agent_loop.py"],
            "message": "feat(react-agent-v1): implement agent loop controller"
        },
        {
            "files": ["main.py", "test_run.py"],
            "message": "feat(react-agent-v1): add interactive main runner and test execution run scripts"
        }
    ],
    6: [
        {
            "files": ["tests/__init__.py", "tests/test_reasoning.py", "tests/test_parser_fallback.py"],
            "message": "test(react-agent-v1): add unit tests for agent reasoning and parser fallback"
        },
        {
            "files": ["tests/test_ollama.py", "tests/test_agent.py"],
            "message": "test(react-agent-v1): add client connection tests and general agent test"
        }
    ],
    7: [
        {
            "files": [
                "evaluation/benchmarks/ambiguous_queries.json",
                "evaluation/benchmarks/arithmetic_tasks.json",
                "evaluation/benchmarks/failure_cases.json",
                "evaluation/benchmarks/multistep_tasks.json",
                "evaluation/benchmarks/search_tasks.json",
                "evaluation/academic_loader.py",
                "evaluation/answer_evaluator.py"
            ],
            "message": "feat(react-agent-v1): add evaluation benchmarks and loader tools"
        },
        {
            "files": [
                "evaluation/run_evaluation.py",
                "evaluation/run_academic_eval.py",
                "evaluation/results/academic_report_20260702_031734.txt",
                "evaluation/results/academic_results_20260702_031734.json",
                "evaluation/results/eval_report_20260702_002750.txt",
                "evaluation/results/eval_results_20260702_002750.json"
            ],
            "message": "feat(react-agent-v1): integrate evaluation run scripts and compile baseline report results"
        }
    ]
}

def run_cmd(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(args)}")
        print(result.stderr)
        return False
    return True

def check_git_status():
    # Check if inside a git repository
    if not os.path.exists(".git"):
        print("Git repository not found. Initializing repository...")
        if not run_cmd(["git", "init"]):
            print("Failed to initialize git repository.")
            sys.exit(1)
        print("Initialized empty Git repository.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Git Progressive Committer for React Agent V1")
    parser.add_argument("--day", type=int, choices=range(1, 8), required=True,
                        help="Specify which day's commits to run (1-7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what git commands would be executed without running them")
    args = parser.parse_args()

    # Move current working directory to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Working Directory set to: {os.getcwd()}")
    check_git_status()

    day = args.day
    commits = COMMITS_BY_DAY[day]

    print(f"\n*** Running commits scheduled for Day {day}:")
    print("=" * 60)

    for i, commit in enumerate(commits):
        files_to_stage = commit["files"]
        commit_msg = commit["message"]
        
        # Verify files exist
        valid_files = []
        missing_files = []
        for f in files_to_stage:
            if os.path.exists(f):
                valid_files.append(f)
            else:
                missing_files.append(f)

        if missing_files:
            print(f"\n[Warning] The following files for Commit {i+1} do not exist in the workspace path:")
            for f in missing_files:
                print(f"  - {f}")

        if not valid_files:
            print(f"Skipping Commit {i+1} as no target files exist.")
            continue

        print(f"\n[Commit {i+1} / {len(commits)}]: {commit_msg}")
        print(f"Staging files: {', '.join(valid_files)}")

        if args.dry_run:
            print("  (Dry-Run) git add " + " ".join(valid_files))
            print(f"  (Dry-Run) git commit -m \"{commit_msg}\"")
        else:
            # Stage only these specific files
            success = run_cmd(["git", "add"] + valid_files)
            if not success:
                print(f"Failed to stage files for commit {i+1}.")
                continue
            
            # Commit with message
            success = run_cmd(["git", "commit", "-m", commit_msg])
            if success:
                print(f"SUCCESS: Successfully committed changes.")
            else:
                print(f"FAILED: Failed to commit changes (or no changes to commit).")

    print("\n" + "=" * 60)
    print(f"Day {day} operations completed.")

if __name__ == "__main__":
    main()
