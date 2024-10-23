from setuptools import setup, find_packages

setup(
    name="computer_use_demo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.38.0",
        "anthropic[bedrock,vertex]>=0.37.1",
        "pyautogui>=0.9.54",
        "Pillow>=10.0.0",
        "jsonschema==4.22.0",
        "boto3>=1.28.57",
        "google-auth<3,>=2",
        "python-dotenv>=1.0.0",
    ],
)
