# RAG_AI_Chatbot

This is a simple chatbot that does q&a with the data feed that is provided in pdf in folder. It can be any pdf, that you want it to be trained and be used.
In terms of architecture, it uses LLAMA 3.2 as based model for LLM. This is installed in local via Ollama framework. The texts within PDF's is extracted and stored as vector embeddings in chromaDB(Open Source vector DB).
These embeddings can be regularly queried with LLM based on similarities.
The output and input for queries are displayed in simple FLask Web application framework.
There is no cloud involved in betweeen and everything is running on local.

![image](https://github.com/user-attachments/assets/8e2902d6-d098-471d-98a0-834dd72f5ce1)

