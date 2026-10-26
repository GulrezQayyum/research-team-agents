import os
import time
from system_a_baseline import run_agent_with_metrics

MODELS_TO_TEST = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

TEST_TOPICS = [
    "Recent breakthroughs in solid-state battery commercialization",
    "Developments in AI hardware inference accelerators",
]

def run_benchmark():
    print("=== RUNNING TASK 2: 20-RUN GROQ BENCHMARK ===")

for model in MODELS_TO_TEST:
    os.environ["GROQ_MODEL"] = model
    print(f"\n==========================================")
    print(f"TESTING MODEL: {model}")
    print(f"==========================================")

    for topic in TEST_TOPICS:
      print(f"\nTopic: '{topic}'")
      for run_idx in range(1, 6):
        res = run_agent_with_metrics(topic, run_idx)
        print(
            f"Run {run_idx}/5 | Time: {res['wall_time_sec']}s | Failed:"
            f" {res['status_failed']} ({res['failure_reason'] or 'None'}) |"
            f" Tokens: {res['input_tokens']} in / {res['output_tokens']} out"
        )
        time.sleep(3.0)

if __name__ == "__main__":
   run_benchmark()