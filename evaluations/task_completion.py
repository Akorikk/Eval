
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
from deepeval.integrations.langchain import CallbackHandler

# Load environment variables from the project's .env file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Make the agent directory available for imports.
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from agent import agent
from dataset import dataset
from groq_judge import GroqJudge


SUCCESSFUL_TASK_DESCRIPTION = """
The refund agent succeeds when it:
1. Asks for a username if one is missing.
2. Verifies the user's account and purchase history.
3. Checks the 30-day refund window.
4. Confirms each requested item was purchased.
5. Refunds only eligible items and the correct amount.
6. Explains why a refund is denied.
7. Answers purchase-history questions without issuing an
   unnecessary refund.
"""

# Create a fresh judge for the evaluation.
judge = GroqJudge()


def create_metric():
    """Create a fresh metric for each test case."""
    return TaskCompletionMetric(
        threshold=0.8,
        model=judge,
        task=SUCCESSFUL_TASK_DESCRIPTION,
        include_reason=True,
    )


def main():
    total_cases = len(dataset.goldens)

    print(f"\nStarting task-completion evaluation.")
    print(f"Total test cases: {total_cases}")
    print("Evaluating one test case at a time.\n")

    for index, golden in enumerate(dataset.goldens, start=1):
        print("=" * 60)
        print(f"Test case {index}/{total_cases}")
        print(f"Input: {golden.input}")

        # Create a dataset containing only this test case.
        single_dataset = EvaluationDataset(
            goldens=[golden]
        )

        metric = create_metric()

        try:
            # Let DeepEval execute the agent and evaluate its trace.
            for evaluated_golden in single_dataset.evals_iterator(
                metrics=[metric]
            ):
                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": evaluated_golden.input,
                            }
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": str(uuid4())
                        },
                        "callbacks": [CallbackHandler()],
                    },
                )

                print(
                    "\nAgent response:",
                    response["messages"][-1].content,
                )

            print(f"\nTest case {index} processing finished.")

        except Exception as error:
            print(
                f"\nTest case {index} failed: "
                f"{type(error).__name__}: {error}"
            )
            print(
                "Continuing to the next test case, if available."
            )

    print("\nEvaluation run finished.")
    print(
        "Check the output above for errors and DeepEval metric results."
    )


if __name__ == "__main__":
    main()
