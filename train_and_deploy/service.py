

import bentoml
from bentoml.io import NumpyNdarray

gitguarden_runner = bentoml.sklearn.get("gitguarden").to_runner()

svc = bentoml.Service(name="gitguarden_service", runners=[gitguarden_runner])

input_spec = NumpyNdarray(dtype="float", shape=(-1, 30))

@svc.api(input=input_spec, output=NumpyNdarray())
async def predict(input_arr):
    return await gitguarden_runner.predict.async_run(input_arr)
