# app.py
import streamlit as st
import pandas as pd
import boto3
from langchain.llms.bedrock import Bedrock
from langchain import PromptTemplate, LLMChain
import matplotlib.pyplot as plt

st.title("🤖 InsightBot - Analiza tus datos con IA")

pregunta = st.text_input("Haz una pregunta sobre las ventas:")

if pregunta:
    # Leer dataset desde S3
    s3 = boto3.client('s3')
    bucket = "agustin-lab-bucket01"
    key = "data/ventas-2.csv"
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])

    # Conectar con Bedrock
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    llm = Bedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock)

    # Prompt para generar análisis
    template = """
    Eres un experto en análisis de datos. Usa Python (Pandas) para responder la siguiente pregunta sobre df.
    Devuelve solo el resultado o una breve interpretación:
    {¿Quien tuvo mas ventas?,muestra las primeras filas del dataframe}
    """
    prompt = PromptTemplate(template=template, input_variables=["pregunta"])
    chain = LLMChain(llm=llm, prompt=prompt)

    respuesta = chain.run(pregunta)
    st.write("💬 Respuesta de InsightBot:")
    st.write(respuesta)
