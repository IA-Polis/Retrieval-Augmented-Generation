import pandas as pd
import argparse


def main(path):

    print(path)
    df = pd.read_csv(path, usecols=["Entrada", "Documento Referência", "PROMPT"])

    prompt_format = df['PROMPT'][0]
    input_receita = df['Entrada']
    documentos = df["Documento Referência"]

    return prompt_format, input_receita, documentos




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to call module's main function"
    )
    parser.add_argument("path", type=str, help="The path to pass to module")
    args = parser.parse_args()
    main(args.path)
