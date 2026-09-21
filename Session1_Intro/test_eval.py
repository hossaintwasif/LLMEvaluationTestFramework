# To run the test we need to do the following:
# 1. Command to run in terminal - deepeval test run test_eval.py
# 2 If you don't want Confident AI then just disable confident Key in .env file and run the test. It will work with OpenAI API key only.
 

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OpenAIModel
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"D:\Udemy_AI_project\Session1_Intro\.env")

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Set it in the environment or .env file.")

llm_model = OpenAIModel(model="gpt-4o-mini", api_key=openai_key)


def test_answer_relevancy():
    metric = AnswerRelevancyMetric(threshold=0.5, model=llm_model)
    test_case = LLMTestCase(
        input="Who is the current president of United States Of America?",
        actual_output="Joe Biden",
        retrieval_context=["The current president of the United States is Joe Biden."],
    )
    assert_test(test_case, [metric])
