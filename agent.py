# agent.py
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

class AutoMLAgent:
    def __init__(self, model="models/gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def ask(self, question: str) -> str:
        response = self.llm.invoke(question)
        return response.content.strip()

    def get_task_type(self, df: pd.DataFrame) -> str:
        prompt = f"""
        You are a data scientist. Given the dataset with columns:
        {list(df.columns)},
        determine the type of ML problem (classification or regression).
        Return one word only: "classification" or "regression".
        """
        return self.ask(prompt).lower()

    def get_cleaning_suggestion(self, df: pd.DataFrame) -> str:
        prompt = f"""
        You are a data science assistant. Given this sample of the dataset:
        {df.head(10).to_string(index=False)},
        suggest:
        - Handling missing values
        - Feature engineering or dropping unnecessary columns
        - Data type conversions

        Respond in bullet points.
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()
    
    


    def get_cleaning_code(self, df: pd.DataFrame) -> str:
        import re

        def clean_response_code(code: str) -> str:
            # Remove triple backticks and python language markers
            code = re.sub(r"```(?:python)?", "", code)
            code = re.sub(r"```", "", code)
            return code.strip()
        
        prompt = f"""
        You're a data preprocessing assistant. The following is a sample dataset:
        {df.head(10).to_markdown(index=False)}

        Based on this data, generate Python Pandas code that:
        1. Handles missing values (drop or fill),
        2. Fixes obvious type issues,
        3. Removes duplicates or outliers if needed.

        Only return executable Python code inside a function named clean_data(df), 
        which accepts a DataFrame and returns the cleaned DataFrame.
        Only return pure Python code. No markdown. No explanation. No formatting characters.
        """
        response = self.llm.invoke(prompt)
        raw_code = response.content.strip()
        return clean_response_code(raw_code)
    


