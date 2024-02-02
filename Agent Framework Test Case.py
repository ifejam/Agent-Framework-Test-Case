import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
from langchain_community.llms import Ollama

ollama_llm = Ollama(model="mistral")
ollama_llm_orca2 = Ollama(model="orca2")

# Define your agents with roles and goals
researcher = Agent(
    role="Senior Research Analyst",
    goal="Generate ideas and key points that are crucial for teaching someone new to the subject.",
    backstory="""You are an expert at a technology research group, 
  skilled in identifying new subject topics.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=ollama_llm,
)
writer = Agent(
    role="educational Content Strategist",
    goal="educate someone new to the topic.",
    backstory="""You are a content strategist known for 
    making complex tech topics interesting and easy to understand.""",
    verbose=True,
    allow_delegation=True,
    llm=ollama_llm_orca2,
)
examiner = Agent(
    role="Top Exams strategist",
    goal="Craft compelling content on AI related subject",
    backstory="""You are a content strategist known for 
    making complex tech topics interesting and easy to understand.""",
    verbose=True,
    allow_delegation=True,
    llm=ollama_llm_orca2,
)


# Create tasks for your agents
task1 = Task(
    description="""Develop ideas for teaching someone new to AI related subject.""",
    agent=researcher,
)

task2 = Task(
    description="""Use the Researcher’s ideas to write a piece of text to explain the topic.""",
    agent=writer,
)
task3 = Task(
    description="""Use the Researcher’s ideas to Craft 2-3 test questions 
    to evaluate understanding of the created text, along with the correct answers.""",
    agent=examiner,
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer, examiner], tasks=[task1, task2, task3], verbose=2
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
