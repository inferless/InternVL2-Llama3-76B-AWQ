from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image


class InferlessPythonModel:

    def initialize(self):
        model = 'OpenGVLab/InternVL2-Llama3-76B-AWQ'
        backend_config = TurbomindEngineConfig(model_format='awq')
        self.pipe = pipeline(model, backend_config=backend_config, log_level='INFO')

    def infer(self, inputs):
        image_url = inputs["image_url"]
        prompt = inputs["prompt"]
        image = load_image(image_url)
        response = self.pipe((prompt, image))
        return {'response': response.text }

    def finalize(self):
        pass
