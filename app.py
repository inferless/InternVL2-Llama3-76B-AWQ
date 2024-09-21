from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image


class InferlessPythonModel:

    def initialize(self):
        backend_config = TurbomindEngineConfig(model_format='awq')
        self.pipe = pipeline(model, backend_config=backend_config, log_level='INFO')

    def infer(self, inputs):
        image_url = inputs["image_url"]
        image = load_image(image_url)
        response = self.pipe(('describe this image', image))
        return {'response': response.text }

    def finalize(self):
        pass
