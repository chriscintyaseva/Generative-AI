
from crewai import Agent, LLM

# Configure local Ollama model
llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0
)

# Agent 1: Research specialist
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate and up-to-date information on AI topics.",
    backstory="""You are a senior AI research analyst with 10 years
    of experience reading and synthesising academic papers,
    technical blogs, and release notes.

    You are known for being precise, citing sources,
    and never making things up.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

# Agent 2: Writing specialist
writer = Agent(
    role="Technical Content Writer",
    goal="Write clear, accurate, and engaging technical content for developers.",
    backstory="""You are a technical writer with a background
    in software engineering.

    You excel at turning complex AI concepts into practical,
    well-structured guides.

    You always write in active voice, use concrete examples,
    and avoid jargon.""",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agent 3: Quality reviewer
reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate, complete, and publication-ready.",
    backstory="""You are a meticulous editor who reviews
    technical content for accuracy, clarity, and completeness.

    You check facts, flag unsupported claims,
    and ensure the writing matches the target audience's level.""",
    llm=llm,
    verbose=True,
    allow_delegation=True
)

# Display agent information
print("========== CREWAI AGENTS ==========")

print("1. Researcher:", researcher.role)
print("2. Writer:", writer.role)
print("3. Reviewer:", reviewer.role)

print("\nAll three agents have been created successfully!")
