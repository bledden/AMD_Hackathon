"""
Dataset Configuration for Q&A Agent
Define themes, sources, and formatting templates
"""

# Theme Options - Choose one for your competition
THEMES = {
    "science": {
        "name": "Science & Technology",
        "description": "Physics, chemistry, biology, computer science, engineering",
        "example_topics": [
            "quantum mechanics", "molecular biology", "artificial intelligence",
            "thermodynamics", "genetics", "computer networks", "chemistry"
        ],
        "difficulty_levels": ["high_school", "undergraduate", "graduate"],
    },
    "history": {
        "name": "World History",
        "description": "Ancient to modern history, civilizations, wars, cultural movements",
        "example_topics": [
            "ancient civilizations", "world wars", "industrial revolution",
            "renaissance", "cold war", "colonialism", "medieval period"
        ],
        "difficulty_levels": ["basic", "intermediate", "advanced"],
    },
    "space": {
        "name": "Space & Astronomy",
        "description": "Planets, stars, galaxies, space exploration, astrophysics",
        "example_topics": [
            "solar system", "black holes", "cosmology", "space missions",
            "exoplanets", "stellar evolution", "dark matter"
        ],
        "difficulty_levels": ["beginner", "intermediate", "expert"],
    },
    "technology": {
        "name": "Modern Technology",
        "description": "Computing, internet, AI, software, hardware, innovations",
        "example_topics": [
            "machine learning", "blockchain", "quantum computing",
            "cybersecurity", "cloud computing", "robotics", "5G"
        ],
        "difficulty_levels": ["basic", "intermediate", "advanced"],
    },
}

# Default theme - change this to your chosen theme
SELECTED_THEME = "science"

# Dataset sources
DATASET_SOURCES = {
    "squad": {
        "name": "SQuAD 2.0",
        "huggingface_id": "squad_v2",
        "type": "reading_comprehension",
        "size": "~150k examples",
    },
    "natural_questions": {
        "name": "Natural Questions",
        "huggingface_id": "natural_questions",
        "type": "open_domain_qa",
        "size": "~300k examples",
    },
    "trivia_qa": {
        "name": "TriviaQA",
        "huggingface_id": "trivia_qa",
        "type": "trivia",
        "size": "~95k examples",
    },
}

# Instruction templates for fine-tuning
INSTRUCTION_TEMPLATES = {
    "question_generation": [
        "Generate a challenging question about {topic}.",
        "Create a {difficulty} question on {topic}.",
        "Write a question that tests understanding of {topic}.",
        "Formulate a question about {topic} that requires {skill}.",
    ],
    "question_answering": [
        "Answer this question: {question}",
        "Provide a concise answer to: {question}",
        "Answer the following question accurately: {question}",
    ],
}

# Skills to test
QUESTION_SKILLS = [
    "factual recall",
    "deep understanding",
    "critical thinking",
    "application of concepts",
    "analysis and synthesis",
]

# Dataset split ratios
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1

# Target dataset sizes
MIN_EXAMPLES = 500
TARGET_EXAMPLES = 1000
MAX_EXAMPLES = 2000

# Quality filters
MIN_QUESTION_LENGTH = 10  # characters
MAX_QUESTION_LENGTH = 200
MIN_ANSWER_LENGTH = 10
MAX_ANSWER_LENGTH = 500

# Format configuration
DATASET_FORMAT = "alpaca"  # or "sharegpt"

ALPACA_FORMAT = {
    "instruction": "str",
    "input": "str",  # usually empty
    "output": "str",
}

def get_theme_config(theme_key=None):
    """Get configuration for selected theme"""
    theme_key = theme_key or SELECTED_THEME
    if theme_key not in THEMES:
        raise ValueError(f"Unknown theme: {theme_key}. Choose from: {list(THEMES.keys())}")
    return THEMES[theme_key]


def format_question_generation_prompt(topic, difficulty=None, skill=None):
    """Format a question generation instruction"""
    template = INSTRUCTION_TEMPLATES["question_generation"][0]

    if difficulty and skill:
        template = "Create a {difficulty} question about {topic} that tests {skill}."
    elif difficulty:
        template = "Create a {difficulty} question about {topic}."

    return template.format(
        topic=topic,
        difficulty=difficulty or "challenging",
        skill=skill or "understanding"
    )


def format_question_answering_prompt(question):
    """Format a question answering instruction"""
    template = INSTRUCTION_TEMPLATES["question_answering"][0]
    return template.format(question=question)


def create_alpaca_format_example(instruction, output, input_text=""):
    """Create an example in Alpaca format"""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def validate_example(example):
    """Validate a Q&A example meets quality criteria"""
    question = example.get("question", "")
    answer = example.get("answer", "")

    if not question or not answer:
        return False

    if len(question) < MIN_QUESTION_LENGTH or len(question) > MAX_QUESTION_LENGTH:
        return False

    if len(answer) < MIN_ANSWER_LENGTH or len(answer) > MAX_ANSWER_LENGTH:
        return False

    return True


if __name__ == "__main__":
    # Print configuration
    print("Dataset Configuration")
    print("=" * 50)
    print(f"Selected Theme: {SELECTED_THEME}")
    theme = get_theme_config()
    print(f"Description: {theme['description']}")
    print(f"Example Topics: {', '.join(theme['example_topics'][:3])}")
    print(f"Target Examples: {TARGET_EXAMPLES}")
    print("=" * 50)
