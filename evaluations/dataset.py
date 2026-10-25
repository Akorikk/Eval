from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import Toolcall 

dataqset = EvaluationDataset(
    goldens=[
        Golden(
            input="I am user123",
            expected_output="Paris",
            expected_tools=[]
        )
    ]
)