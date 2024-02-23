import os

from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_community.embeddings.openai import OpenAIEmbeddings

import json
import openai
import re

class _langchainApi:
    def __init__(self):
        curdir = os.path.dirname(__file__)
        config_path = os.path.join(curdir, "config.json")
        conf = None
        if not os.path.exists(config_path):
            raise Exception("config.json not found")
        else:
            with open(config_path, "r",encoding='utf-8') as f:
                conf = json.load(f)

        self.pinecone_api_key = conf["pinecone_api_key"]
        self.pinecone_environment = conf["pinecone_environment"]
        self.pinecone_index_name = conf["pinecone_index_name"]
        self.pinecone_name_space = conf["pinecone_name_space"]

        self.openai_api_key = conf["openai_api_key"]
        self.openai_model_name = conf["openai_model_name"]
        self.openai_api_base = conf["openai_api_base"]


        self.openai_query_key = conf["openai_query_key"]
        self.openai_query_base = conf["openai_query_base"]
        self.openai_query_prompt = conf["openai_query_prompt"]
        self.openai_query_model = conf["openai_query_model"]
        
        openai.api_key = self.openai_query_key 
        openai.api_base = self.openai_query_base

        self.llm_threshold = conf.get("llm_threshold", 0.8)
        self.plugin_trigger_prefix = conf.get("plugin_trigger_prefix", "$")

    
    def get_docs(self, content):
        try:
            # configure client
            pc = Pinecone(api_key=self.pinecone_api_key)
            index_name = self.pinecone_index_name
            index = pc.Index(index_name)

            embed = OpenAIEmbeddings(
                model=self.openai_model_name,
                deployment=self.openai_model_name,
                openai_api_key=self.openai_api_key,
                openai_api_base=self.openai_api_base,
            )
            vectorstore = PineconeStore(
                index, embed, 'text',namespace=self.pinecone_name_space
            )
            docs = vectorstore.similarity_search_with_score(
                content,  # our search query
                k=1  # return 3 most relevant docs
            )
            score = docs[0][1]
            print("search docs with score : %s " % score )

            query = content + "（请尝试在以下知识库中整理出答案，找不到再自己尝试回答:" + docs[0][0].page_content + ")"
            response = openai.ChatCompletion.create(
                model=self.openai_query_model, # model = "deployment_name".
               
                messages=[
                    {"role": "system", "content": self.openai_query_prompt },
                    {"role": "user", "content": query}
                ]
            )
            res_content = response.choices[0].message.content.strip().replace("<|endoftext|>", "")
            print(res_content)

            # print(docs[0][0].page_content)
        except Exception as e:
            raise e
    
def main():

    
    api = _langchainApi()
    content = 'Mega怎么用'
    content = re.sub('\[.*?\]', '', content)
    print(content)
    api.get_docs(content)

if __name__ == '__main__':
    main()