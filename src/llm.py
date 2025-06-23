from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_community.vectorstores import Chroma


class LLM:

    def __init__(self, model_name, token='TOKEN'):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit = True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, device_map = "auto")

    def generates(self, prompt):
        inputs = self.tokenizer.encode(
            prompt, return_tensors="pt"
        ).cuda()
        kwargs = {
            #"max_new_tokens": 1048,
            "max_new_tokens": 200,
            "min_new_tokens": 1,
            "temperature": 0.2,
            "do_sample": True
        }
        outputs = self.model.generate(inputs, **kwargs)
        result = self.tokenizer.decode(outputs[0][len(inputs[0]) :])
        return result

    def predict_RAG(self, inst, input_data, context):

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Você é uma assistente de IA que responde dúvidas de usuários sobre medicamentos.
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        Siga as instuções abaixo:
        {inst} /
        Use o contexto a seguir, se necessário: {context}
        Dados de entrada: {input_data}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        output = self.generates(prompt)
        return output
