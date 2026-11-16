"""Specialist tutor agents that can be attached to a Strands agent."""

from strands import Agent
from strands.models import Model
from strands.types.tools import AgentTool


def create_tutor_agents(model: Model) -> tuple[Agent, Agent, Agent]:
    """Create the math, writing, and science specialists with one shared model."""
    math_tutor = Agent(
        name="math_tutor",
        description="Solves mathematical equations and explains math concepts step by step.",
        model=model,
        system_prompt=(
            "You are a math tutor. Solve math problems and explain your "
            "solutions clearly, step by step."
        ),
        callback_handler=None,
    )
    writing_coach = Agent(
        name="writing_coach",
        description="Helps with writing, grammar, and essay structure.",
        model=model,
        system_prompt=(
            "You are a writing coach. Help with writing, grammar, and essay "
            "structure. Provide constructive feedback."
        ),
        callback_handler=None,
    )
    science_helper = Agent(
        name="science_helper",
        description="Explains scientific concepts and principles in simple terms.",
        model=model,
        system_prompt=(
            "You are a science tutor. Explain scientific concepts clearly "
            "using simple language and real-world examples."
        ),
        callback_handler=None,
    )
    return math_tutor, writing_coach, science_helper


def create_math_tools(model: Model) -> list[AgentTool]:
    """Create specialist tutor tools using the caller's configured model."""
    return [tutor.as_tool() for tutor in create_tutor_agents(model)]


def create_teacher(model: Model) -> Agent:
    """Create a teacher agent that delegates to specialist tutors."""
    return Agent(
        name="teacher",
        model=model,
        system_prompt="""You are a teacher who helps students by delegating to specialist tutors.
Never solve problems yourself. Always delegate to the appropriate specialist:
- math_tutor: for math problems and equations
- writing_coach: for writing, grammar, and essays
- science_helper: for science questions

After receiving the specialist's response, present it to the student in a friendly way.""",
        tools=create_math_tools(model),
        callback_handler=None,
    )
