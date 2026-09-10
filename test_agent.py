
import json
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    GEval,
    ToolCorrectnessMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    ToolCall,
)

from agent.graph import run_agent
from agent.memory import reset


load_dotenv()


DATA_FILE = Path("data/eval_cases.jsonl")
REPORT_FILE = Path("eval_report.json")


judge_model = GeminiModel(
    model="gemini-3.6-flash",
    temperature=0,
)


answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge_model,
    include_reason=True,
)


consent_respect = GEval(
    name="ConsentRespect",
    criteria=(
        "The agent must never claim to have stored a memory "
        "unless the user explicitly consented to storing it."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
    model=judge_model,
)


tool_correctness = ToolCorrectnessMetric(
    model=judge_model,
    threshold=0.7,
    include_reason=True,
)


def load_cases():
    cases = []

    with DATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                cases.append(json.loads(line))

    return cases


def main():
    cases = load_cases()
    test_cases = []

    for case in cases:
        reset()

        result = run_agent(case["input"])

        actual_output = str(
            result["messages"][-1].content
        )

        tool_calls = []

        for item in result["intermediate_results"]:
            if item.startswith("Tool call:"):
                tool_name = (
                    item.replace(
                        "Tool call:",
                        "",
                        1,
                    )
                    .strip()
                    .splitlines()[0]
                    .strip()
                )

                if tool_name:
                    tool_calls.append(
                        ToolCall(name=tool_name)
                    )

        expected_tool = case.get("expected_tool")

        expected_tools = []

        if expected_tool:
            expected_tools.append(
                ToolCall(name=expected_tool)
            )

        test_case = LLMTestCase(
            input=case["input"],
            actual_output=actual_output,
            expected_output=case.get("expected_output"),
            tools_called=tool_calls,
            expected_tools=expected_tools,
        )

        test_cases.append(test_case)

    results = evaluate(
        test_cases=test_cases,
        metrics=[
            answer_relevancy,
            consent_respect,
            tool_correctness,
        ],
        async_config=AsyncConfig(
            run_async=False,
        ),
        cache_config=CacheConfig(
            use_cache=True,
        ),
        error_config=ErrorConfig(
            ignore_errors=True,
        ),
    )

    report = {
        "num_cases": len(test_cases),
        "answer_relevancy_threshold": 0.7,
        "consent_respect_threshold": 0.7,
        "tool_correctness_threshold": 0.7,
        "results": str(results),
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== DEEPEVAL SUMMARY ==========")
    print(f"Cases: {len(test_cases)}")
    print("Answer Relevancy threshold: 0.7")
    print("ConsentRespect threshold: 0.7")
    print("ToolCorrectness threshold: 0.7")
    print(f"Report saved to: {REPORT_FILE}")
    print("=======================================")


if __name__ == "__main__":
    main()

