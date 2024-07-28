# Digital-GDM_Screen
This repo has RAG Application imlementation using Sentence transformers for Retrieval and ChatGPT-4 as a Generator.

## Introduction
Gestational diabetes mellitus (GDM) is a growing health concern affecting many pregnant women worldwide. Access to accurate and comprehensive information is crucial for managing and mitigating the risks associated with GDM. Traditional methods of information retrieval can be cumbersome and may not provide tailored responses. This project leverages the Retrieval-Augmented Generation (RAG) framework to develop a system that delivers relevant, contextually rich answers to queries related to gestational diabetes.

## Objectives
The primary objectives of this project are to:
- Improve the accessibility and accuracy of information on gestational diabetes.
- Enhance the user experience by delivering tailored responses.
- Integrate state-of-the-art retrieval and generation techniques to ensure coherence and relevance.

## Features
- Accurate Information: Provides reliable and accurate information on gestational diabetes.
- Personalized Responses: Delivers contextually rich and tailored responses to user queries.
- Efficient Retrieval: Combines retrieval and generation techniques to ensure relevant information is presented.
- User-Friendly Interface: Easy-to-use interface for querying and receiving information.


## Components
- Corpus of Gestational Diabetes Texts: The knowledge base for the system.
- Embeddings Model: Converts texts into numerical embeddings.
- Vectorstore: Database that stores embeddings and facilitates retrieval.
- User Interface: Where users submit queries.
- Gestational Diabetes Query Prompt: Combines user query with retrieved information.
- OpenAI GPT-4 (LLM): Generates the final response.
- Output: The response delivered to the user.
