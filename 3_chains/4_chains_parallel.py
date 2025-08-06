from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4")

# Define prompt template for movie summary
# Define prompt templates
prompt_template_location = ChatPromptTemplate.from_messages(
    [
        ("system", "Your job is to come up with a list of 5 dishes from the area that the users suggests."),
        ("human", "Provide a list of dishes from {user_location}."),
    ]
)

# Generate Recipe step
def generate_receipe(dish):
    dish_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Given a list of dishes, select any one dish and give a short and simple recipe with ingedients list of selected dish on how to make that dish at home"),
            ("human", "How to prepare {dish}."),
        ]
    )
    return dish_template.format_prompt(dish=dish)

# Generate Recipe step
def benefits_list(ingredients):
    benefits_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Using the ingredients list of the dish, provide analysis of few of its ingredients along with their health benefits"),
            ("human", "What are the health benefits of {ingredients}?"),
        ]
    )
    return benefits_template.format_prompt(ingredients=ingredients)

# Combine analyses into a final verdict
def combine_dish(generate_receipe, benefits_list):
    return f"Recipe:\n{generate_receipe}\n\nBenefits:\n{benefits_list}"

# Simplify branches with LCEL
recipe_branch_chain = (
    RunnableLambda(lambda x: generate_receipe(x)) | model | StrOutputParser()
)

benefits_branch_chain = (
    RunnableLambda(lambda x: benefits_list(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template_location
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"recipe": recipe_branch_chain, "benefits": benefits_branch_chain})
    | RunnableLambda(lambda x: combine_dish(x["branches"]["recipe"], x["branches"]["benefits"]))
)

# Run the chain
result = chain.invoke({"user_location": "Amritsar"})

print(result)