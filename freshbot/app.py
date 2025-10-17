import streamlit as st
import pandas as pd
import boto3
import json
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIGURACIÓN DE STREAMLIT
# ---------------------------------------------------------
st.set_page_config(page_title="InsightBot", page_icon="🤖")
st.title("🤖 InsightBot – Análisis Inteligente con Amazon Bedrock (Titan)")

# ---------------------------------------------------------
# PARÁMETROS DEL BUCKET S3
# ---------------------------------------------------------
bucket = "agustin-lab-bucket01"      # <-- tu bucket
key = "data/ventas-2.csv"            # <-- ruta del dataset en S3

# ---------------------------------------------------------
# CARGAR DATASET DESDE S3
# ---------------------------------------------------------
try:
    s3 = boto3.client("s3", region_name="us-east-1")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj["Body"])
    st.subheader("📊 Dataset cargado desde S3")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Error al leer el dataset desde S3: {e}")
    st.stop()

# ---------------------------------------------------------
# CONFIGURAR CLIENTE DE AMAZON BEDROCK
# ---------------------------------------------------------
bedrock_region = "us-east-1"
client = boto3.client("bedrock-runtime", region_name=bedrock_region)
model_id = "amazon.titan-text-express-v1"  # Modelo disponible en tu cuenta

# ---------------------------------------------------------
# INTERFAZ DE PREGUNTAS
# ---------------------------------------------------------
st.subheader("💬 Haz una pregunta sobre los datos")
user_question = st.text_input("Ejemplo: ¿Qué producto tuvo más ventas totales?")

if st.button("Analizar con IA"):
    if not user_question:
        st.warning("Por favor escribe una pregunta primero.")
    else:
        # -----------------------------------------------------
        # PROMPT PARA TITAN
        # -----------------------------------------------------
        prompt = f"""
        Eres un analista de datos experto. Analiza el siguiente dataset de ventas:
        {df.head(10).to_string(index=False)}

        Responde en español con claridad y precisión a la siguiente pregunta:
        {user_question}
        """

        # -----------------------------------------------------
        # INVOCAR MODELO TITAN TEXT
        # -----------------------------------------------------
        try:
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 500,
                    "temperature": 0.7,
                },
            })

            response = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            result = json.loads(response["body"].read())
            full_response = result.get("results", [{}])[0].get("outputText", "")

            st.subheader("🧠 Respuesta de InsightBot")
            st.write(full_response)

        except Exception as e:
            st.error(f"❌ Error al comunicarse con Bedrock: {e}")
            st.stop()

        # -----------------------------------------------------
        # VISUALIZACIONES AUTOMÁTICAS SEGÚN LA PREGUNTA
        # -----------------------------------------------------
        try:
            if "mes" in user_question.lower():
                st.subheader("📈 Gráfico: Ventas por mes")
                resumen = df.groupby("mes")["ventas"].sum().reset_index()
                plt.figure(figsize=(6, 4))
                plt.bar(resumen["mes"], resumen["ventas"])
                plt.xlabel("Mes")
                plt.ylabel("Ventas totales")
                st.pyplot(plt)

            elif "producto" in user_question.lower():
                st.subheader("📈 Gráfico: Ventas por producto")
                resumen = df.groupby("producto")["ventas"].sum().reset_index()
                plt.figure(figsize=(6, 4))
                plt.bar(resumen["producto"], resumen["ventas"])
                plt.xlabel("Producto")
                plt.ylabel("Ventas totales")
                st.pyplot(plt)

            elif "region" in user_question.lower():
                st.subheader("📈 Gráfico: Ventas por región")
                resumen = df.groupby("region")["ventas"].sum().reset_index()
                plt.figure(figsize=(6, 4))
                plt.bar(resumen["region"], resumen["ventas"])
                plt.xlabel("Región")
                plt.ylabel("Ventas totales")
                st.pyplot(plt)
        except Exception as e:
            st.warning(f"No se pudo generar gráfico automático: {e}")

# ---------------------------------------------------------
# FIN DE LA APP
# ---------------------------------------------------------
