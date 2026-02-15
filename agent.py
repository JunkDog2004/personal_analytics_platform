# agent.py
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import re
from langchain_google_genai import ChatGoogleGenerativeAI

class AutoMLAgent:
    def __init__(self, model="gemini-2.0-flash"):
        # Initializing with a temperature of 0 for consistent code generation
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0 
        )

    def ask(self, question: str) -> str:
        response = self.llm.invoke(question)
        return response.content.strip()

    def get_task_type(self, df: pd.DataFrame) -> str:
        prompt = f"""
        Analyze these columns: {list(df.columns)}.
        Based on standard data science practices, is this a 'classification' or 'regression' problem?
        Return ONLY the word 'classification' or 'regression'.
        """
        return self.ask(prompt).lower()

    def get_cleaning_suggestion(self, df: pd.DataFrame) -> str:
        # Using markdown for better LLM readability
        prompt = f"""
        Review this data sample:
        {df.head(10).to_markdown(index=False)}

        Provide brief bullet points for:
        - Missing value strategy
        - Recommended feature drops
        - Type conversions
        Use a professional, minimalist tone.
        """
        return self.ask(prompt)

    def get_cleaning_code(self, df: pd.DataFrame) -> str:
        prompt = f"""
        Generate a Python function `clean_data(df)` for this dataset:
        {df.head(5).to_markdown(index=False)}

        Requirements:
        1. Handle nulls.
        2. Encode strings if necessary.
        3. Return the cleaned dataframe.
        
        CRITICAL: Include 'import pandas as pd' and 'import numpy as np' inside the function.
        Return ONLY the raw python code. No backticks, no markdown, no explanations.
        """
        raw_response = self.ask(prompt)
        return self._extract_pure_code(raw_response)

    def _extract_pure_code(self, text: str) -> str:
        # Robustly remove markdown code blocks if the LLM ignores instructions
        text = re.sub(r"```python\n?", "", text)
        text = re.sub(r"```\n?", "", text)
        return text.strip()
