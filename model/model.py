"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._models = {}
        self._tokenizers = {}

    def load(self):
        models = {
            "BSJCode-1-Stable": "BSAtlas/BSJCode-1-Stable",
            "CodeLlama": "codellama/CodeLlama-7b-Instruct-hf",
            "Terjman": "atlasia/Terjman-Ultra"
        }

        for name, model_path in models.items():
            self._models[name] = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto"
            )
            self._tokenizers[name] = AutoTokenizer.from_pretrained(model_path)

    def generate_response(self, model_name, input_text):
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not loaded.")
        tokenizer = self._tokenizers[model_name]
        model = self._models[model_name]
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs["input_ids"], max_length=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def predict(self, request):
        input_text = request.get("input", "")
        service_type = request.get("service", "BS-friendly")

        if not input_text:
            return {"error": "Input text is required"}

        if service_type == "BS-friendly":
            return {"output": self.generate_response("BSJCode-1-Stable", input_text)}
        elif service_type == "Pro":
            return {"output": self.generate_response("CodeLlama", input_text)}
        elif service_type == "Premium":
            intermediate_bs = self.generate_response("BSJCode-1-Stable", input_text)
            intermediate_cl = self.generate_response("CodeLlama", intermediate_bs)
            response = self.generate_response("Terjman", intermediate_cl)
            return {"output": response}
        else:
            return {"error": "Invalid service type"}