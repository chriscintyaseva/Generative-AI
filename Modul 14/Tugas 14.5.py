
from crewai import Agent, Task, Crew, Process, LLM


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
# 2. CREATE AGENTS
# ==========================================

researcher = Agent(
    role="AI Research Analyst",
    goal=(
        "Research Retrieval-Augmented Generation (RAG) "
        "and produce accurate, structured research findings."
    ),
    backstory=(
        "You are an AI researcher specializing in "
        "Large Language Models, information retrieval, "
        "and Retrieval-Augmented Generation."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=False,
    max_iter=2
)


writer = Agent(
    role="Technical Content Writer",
    goal=(
        "Write a clear and technically accurate developer guide "
        "about building production RAG systems."
    ),
    backstory=(
        "You are a technical writer experienced in explaining "
        "AI engineering concepts to Python developers."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=False,
    max_iter=2
)


reviewer = Agent(
    role="Content Quality Reviewer",
    goal=(
        "Evaluate the technical accuracy, completeness, "
        "and clarity of the RAG developer guide."
    ),
    backstory=(
        "You are a senior technical editor specializing "
        "in AI and software engineering documentation."
    ),
    llm=llm,
    verbose=False,
    allow_delegation=False,
    max_iter=2
)


# ==========================================
# 3. CREATE TASKS
# ==========================================

research_task = Task(
    description="""
    Research the topic: Retrieval-Augmented Generation (RAG).

    Cover these five points:

    1. Definition and purpose of RAG.
    2. Differences between Naive RAG, Advanced RAG,
       and Modular RAG.
    3. Main components of a RAG pipeline:
       document loading, chunking, embedding,
       vector database, retrieval, and generation.
    4. Common challenges when implementing RAG in production.
    5. Practical use cases of RAG.

    IMPORTANT:
    - Focus ONLY on Retrieval-Augmented Generation.
    - Do not discuss unrelated GPT model context windows.
    - Do not invent model names, statistics, or references.
    - If a detail is uncertain, clearly state that it is uncertain.
    - Organize findings using headings and bullet points.
    """,
    expected_output=(
        "A structured research report about RAG containing "
        "its definition, approaches, architecture, challenges, "
        "and practical use cases."
    ),
    agent=researcher
)


writing_task = Task(
    description="""
    Using the research findings from the previous task,
    write a developer guide titled:

    "Building Production RAG Systems"

    Target audience:
    Senior Python developers.

    Required sections:

    1. Introduction to RAG.
    2. Understanding RAG architecture.
    3. Naive RAG, Advanced RAG, and Modular RAG.
    4. Building a basic RAG pipeline.
    5. Production challenges and best practices.
    6. Practical applications.
    7. Conclusion.

    Writing requirements:
    - Write approximately 600-800 words.
    - Use clear and professional technical English.
    - Explain technical terms briefly.
    - Include practical examples where appropriate.
    - Stay focused on RAG technology.
    - Do not introduce unrelated GPT model names.
    - Do not invent technical specifications.
    """,
    expected_output=(
        "A complete developer guide titled "
        "'Building Production RAG Systems', "
        "organized into the seven requested sections."
    ),
    agent=writer,
    context=[research_task]
)


review_task = Task(
    description="""
    Review the developer guide produced by the previous task.

    Evaluate these four criteria:

    1. Technical accuracy:
       Are the RAG concepts and explanations correct?

    2. Completeness:
       Are all seven required sections included?

    3. Clarity:
       Is the writing understandable and well organized?

    4. Audience suitability:
       Is the content appropriate for senior Python developers?

    IMPORTANT:
    - Review ONLY the provided RAG article.
    - Do not introduce unrelated model names.
    - Do not ask the user for additional information.
    - Identify specific weaknesses if they exist.
    - Suggest practical improvements.

    End your review with one verdict:
    APPROVED or NEEDS REVISION.
    """,
    expected_output=(
        "A quality review report containing an overall assessment, "
        "technical accuracy evaluation, completeness evaluation, "
        "clarity evaluation, improvement suggestions, "
        "and a final verdict."
    ),
    agent=reviewer,
    context=[writing_task]
)


# ==========================================
# 4. CREATE CREW
# ==========================================

crew = Crew(
    agents=[
        researcher,
        writer,
        reviewer
    ],
    tasks=[
        research_task,
        writing_task,
        review_task
    ],
    process=Process.sequential,
    verbose=False,
    memory=False
)


# ==========================================
# 5. EXECUTE CREW
# ==========================================

print("\n========== STARTING CREW ==========\n")

result = crew.kickoff()


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n========== FINAL CREW OUTPUT ==========\n")
print(result)


print("\n========== INDIVIDUAL TASK OUTPUTS ==========")

task_names = [
    "AI Research Analyst",
    "Technical Content Writer",
    "Content Quality Reviewer"
]

for name, task in zip(task_names, crew.tasks):
    print(f"\n--- {name} ---\n")
    print(task.output.raw if task.output else "No output generated.")


print("\n========== CREW EXECUTION COMPLETED ==========")
