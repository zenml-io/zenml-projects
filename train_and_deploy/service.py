

import bentoml
from bentoml.io import NumpyNdarray

e2e_use_case_runner = bentoml.sklearn.get("e2e_use_case").to_runner()

svc = bentoml.Service(name="e2e_use_case_service", runners=[e2e_use_case_runner])

input_spec = NumpyNdarray(dtype="float", shape=(-1, 30))

@svc.api(input=input_spec, output=NumpyNdarray())
async def predict(input_arr):
    return await e2e_use_case_runner.predict.async_run(input_arr)
