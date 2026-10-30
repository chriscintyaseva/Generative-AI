
from crewai import Crew, Process, Agent, Task, LLM


# ==========================================
# 1. CONFIGURATION
# ==========================================

print("Initializing Ollama LLM...")

llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0.2
)


# ==========================================
# 2. CREATE MANAGER AGENT
# ==========================================

manager = Agent(
    role="AI Content Production Manager",
    goal=(
        "Coordinate a team to produce high-quality AI content "
        "efficiently. Assign research and writing work to the "
        "appropriate specialists and ensure the final result "
        "meets quality standards."
    ),
    backstory=(
        "You are an experienced content production manager "
        "who excels at breaking down projects, assigning work "
        "to specialists, and ensuring quality standards are "
        "met before publication."
    ),
    llm=llm,
    allow_delegation=True,
    verbose=False,
    max_iter=3
)


# ==========================================
# 3. CREATE WORKER AGENTS
# ==========================================

researcher = Agent(
    role="AI Research Analyst",
    goal=(
        "Research AI topics thoroughly and provide "
        "accurate technical information."
    ),
    backstory=(
        "You are a senior researcher specializing in "
        "Large Language Models, vector databases, "
        "information retrieval, and RAG systems."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=False,
    max_iter=2
)


writer = Agent(
    role="Technical Writer",
    goal=(
        "Write clear, structured, and developer-focused "
        "technical content based on research findings."
    ),
    backstory=(
        "You are a technical writer experienced in Python, "
        "machine learning, and AI engineering documentation."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=False,
    max_iter=2
)


# ==========================================
# 4. CREATE MAIN TASK
# ==========================================

main_task = Task(
    description="""
    Produce a complete technical brief titled:

    "Vector Databases for RAG in Production"

    The technical brief must cover:

    1. What vector databases are and how they work.
    2. Their role in Retrieval-Augmented Generation (RAG).
    3. How to choose a vector database based on:
       - Performance and scalability.
       - Search capabilities.
       - Cost and infrastructure.
       - Integration with Python and AI frameworks.
    4. Three implementation best practices:
       - Choosing appropriate embedding models.
       - Designing effective document chunking and indexing.
       - Monitoring retrieval quality and performance.

    The manager should coordinate the research and writing
    specialists to complete this assignment.

    IMPORTANT:
    - Focus only on vector databases and RAG.
    - Do not invent technical specifications or benchmarks.
    - Use clear and professional technical English.
    - Include practical examples where appropriate.
    - Do not ask the user for additional information.
    """,
    expected_output=(
        "A structured technical brief in Markdown, "
        "approximately 400-500 words, covering vector databases, "
        "selection criteria, and three implementation best practices."
    )
)


# ==========================================
# 5. CREATE HIERARCHICAL CREW
# ==========================================

crew = Crew(
    agents=[
        researcher,
        writer
    ],
    tasks=[
        main_task
    ],
    process=Process.hierarchical,
    manager_agent=manager,
    verbose=False,
    memory=False
)


# ==========================================
# 6. EXECUTE CREW
# ==========================================

print("\n========== STARTING HIERARCHICAL CREW ==========\n")

result = crew.kickoff()


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("\n========== FINAL CREW OUTPUT ==========\n")
print(result)

print("\n========== EXECUTION COMPLETED ==========")
