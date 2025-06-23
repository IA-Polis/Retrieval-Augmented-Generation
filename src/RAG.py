import sys
import pandas as pd
import chroma, extract_input
from llm import LLM
from tqdm import tqdm

# TESTE


def main():
    # Reading args
    model_name = sys.argv[1]
    data_path = sys.argv[2]
    input_data_path = sys.argv[3]

    df = pd.DataFrame(columns=['Saída LLama3'])

    context = pd.read_csv('Itens 6&7-1750-800_condicional.csv')
    context['input'] = context['item 6'] + ' ' + context['item7']
    inst, input_data, references = extract_input.main(input_data_path)

    llm = LLM(model_name=model_name)

    for i in tqdm(range(0, len(input_data)), desc="Gerando Saídas"):
        if references[i] == "1324261.pdf": 
            result = "Arquivo corrompido"
            df.loc[len(df)] = result
            continue
        result = llm.predict_RAG(inst, input_data[i], context['input'][i])
        df.loc[len(df)] = result[:-10]

    df.to_csv("RAG_llama3_final.csv", index=False)

if __name__ == "__main__":
    main()
