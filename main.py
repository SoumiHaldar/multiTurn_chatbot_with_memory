from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models  import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS



def new():
    # Extract the text from a pdf
    with open('DP1Merrill_Manual_en.pdf','rb') as f:
        pdf_reader=PdfReader(f)
        text=''
        for each_page in pdf_reader.pages:
            text=text+each_page.extract_text()

        # get chunks
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)

        # embedding
        embeddings = OpenAIEmbeddings()
        vectorscore=FAISS.from_texts(chunks,embedding=embeddings)
        # retrieve the conversation
        retriever=vectorscore.as_retriever(search_type="similarity",search_kwargs={'k':3})
        llm=ChatOpenAI(temperature=0.7,model_name='gpt-3.5-turbo')
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

        chat_history=[]   
        while True:
            query=input("Ask questions about your uploaded pdf file:")
            if query=='exit':
                break
            # print(query)
            chat_history.append(query)
            response=conversation_chain({"question":query,'chat_history':chat_history})
            # print(response)
            result=response['answer']
            print(result)
          


if __name__=='__main__':
    new()


