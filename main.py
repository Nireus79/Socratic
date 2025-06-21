import os
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

claude_api_key = os.getenv("API_KEY_CLAUDE")
print(claude_api_key)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


@dataclass
class ProjectContext:
    """Class for storing project context"""
    goals: List[str] = None
    requirements: List[str] = None
    constraints: List[str] = None
    tech_stack: List[str] = None
    experience_level: str = "beginner"
    domain: str = ""
    timeline: str = ""
    current_challenges: List[str] = None
    # New fields for project specifications
    target_audience: str = ""
    language_preference: str = "english"  # english, multilingual, etc.
    code_style: str = "clean"  # clean, documented, minimal, etc.
    deployment_target: str = ""  # cloud, local, mobile, etc.
    team_size: str = "individual"  # individual, small_team, large_team
    maintenance_requirements: str = ""
    internationalization: bool = False

    def __post_init__(self):
        if self.goals is None:
            self.goals = []
        if self.requirements is None:
            self.requirements = []
        if self.constraints is None:
            self.constraints = []
        if self.tech_stack is None:
            self.tech_stack = []
        if self.current_challenges is None:
            self.current_challenges = []


@dataclass
class KnowledgeEntry:
    """Entry for the knowledge base"""
    content: str
    category: str
    tags: List[str]
    embedding: Optional[np.ndarray] = None


class SocraticRAG:
    def __init__(self, claude_api_key: str, knowledge_base_path: str = "knowledge_base.pkl"):
        """
        Initialize the Socratic RAG system

        Args:
            claude_api_key: API key for Claude
            knowledge_base_path: Path for storing the knowledge base
        """
        self.client = anthropic.Anthropic(api_key=claude_api_key)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base: List[KnowledgeEntry] = []
        self.knowledge_base_path = knowledge_base_path
        self.project_context = ProjectContext()
        self.conversation_history = []
        self.current_phase = "discovery"  # discovery, analysis, design, implementation
        self.missing_context_items = []  # Track what context we still need

        # Enhanced Socratic questions with context-gathering focus
        self.socratic_questions = {
            "discovery": [
                "What exactly do you want to achieve with this project?",
                "What is the main problem you're trying to solve?",
                "Who will be the end users of this system?",
                "What would success look like for this project?",
                "What are the core features it must have?",
                # Context-gathering questions
                "Will this be used by people who speak different languages?",
                "Do you prefer code comments and documentation in English or another language?",
                "Is this for personal use, a team, or public release?",
                "What's your target deployment environment (cloud, local, mobile)?",
                "Do you have any preferences for code style or documentation standards?"
            ],
            "analysis": [
                "Have you considered what happens if the system is used by many users simultaneously?",
                "How will you handle error cases and edge conditions?",
                "What data will you need and where will you store it?",
                "Are there similar systems you could study for reference?",
                "What are the biggest challenges you anticipate?",
                # Context-gathering questions
                "What's your experience level with the technologies involved?",
                "Will you be maintaining this long-term or is it a one-time project?",
                "Do you need the system to be internationalized for different regions?",
                "Are there specific coding standards your team follows?"
            ],
            "design": [
                "How will you organize your code into modules and components?",
                "What's the data flow through your system?",
                "How will you ensure security and data protection?",
                "What APIs or external services will you need?",
                "How will you approach testing and quality assurance?",
                # Context-gathering questions
                "Should variable names and functions be in English or your native language?",
                "Do you need comprehensive documentation for other developers?",
                "What level of code comments would be appropriate for your team?"
            ],
            "implementation": [
                "In what order will you implement the features?",
                "How will you track progress and milestones?",
                "What will you do if you get stuck on a particular issue?",
                "How will you handle deployment and release?",
                "How will you maintain the project after completion?",
                # Context-gathering questions
                "Should error messages be in English or localized?",
                "Do you need the code to be easily readable by international developers?",
                "What documentation language would serve your team best?"
            ]
        }

        # Context validation rules
        self.context_requirements = {
            "language_preference": ["english", "native", "multilingual"],
            "code_style": ["minimal", "documented", "enterprise", "academic"],
            "deployment_target": ["cloud", "local", "mobile", "embedded", "web"],
            "team_size": ["individual", "small_team", "large_team", "open_source"],
            "experience_level": ["beginner", "intermediate", "advanced", "expert"]
        }

        # Load existing knowledge base
        self.load_knowledge_base()

    def add_knowledge(self, content: str, category: str, tags: List[str]):
        """Add new knowledge to the base"""
        embedding = self.encoder.encode([content])[0]
        entry = KnowledgeEntry(
            content=content,
            category=category,
            tags=tags,
            embedding=embedding
        )
        self.knowledge_base.append(entry)
        self.save_knowledge_base()

    def load_knowledge_base(self):
        """Load knowledge base from file"""
        try:
            with open(self.knowledge_base_path, 'rb') as f:
                self.knowledge_base = pickle.load(f)
        except FileNotFoundError:
            self.initialize_default_knowledge()

    def save_knowledge_base(self):
        """Save knowledge base to file"""
        with open(self.knowledge_base_path, 'wb') as f:
            pickle.dump(self.knowledge_base, f)

    def initialize_default_knowledge(self):
        """Initialize with basic development knowledge"""
        default_knowledge = [
            (
                "The Model-View-Controller (MVC) pattern separates the application into three components: Model (data), View (UI), Controller (logic). This helps with code maintainability.",
                "architecture", ["mvc", "design-patterns", "architecture"]),

            (
                "RESTful API design follows principles: stateless communication, uniform interface, cacheable responses. Uses HTTP methods (GET, POST, PUT, DELETE) and status codes.",
                "api-design", ["rest", "api", "web-development"]),

            ("Database normalization reduces redundancy and improves data integrity. Main levels are 1NF, 2NF, 3NF.",
             "database", ["database", "normalization", "sql"]),

            (
                "Test-Driven Development (TDD): Write tests first, then code to pass tests, then refactor. Red-Green-Refactor cycle.",
                "testing", ["tdd", "testing", "methodology"]),

            (
                "Microservices architecture splits the application into small, independent services. Advantages: scalability, technology diversity. Disadvantages: complexity, network overhead.",
                "architecture", ["microservices", "architecture", "scalability"]),

            (
                "For international projects, use English for variable names, function names, and comments. This ensures code readability across global teams.",
                "best-practices", ["internationalization", "code-style", "collaboration"]),

            (
                "Clean code principles: meaningful names, small functions, consistent formatting, comprehensive comments for complex logic.",
                "best-practices", ["clean-code", "maintainability", "documentation"])
        ]

        for content, category, tags in default_knowledge:
            self.add_knowledge(content, category, tags)

    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[KnowledgeEntry]:
        """Retrieve relevant knowledge from the base"""
        if not self.knowledge_base:
            return []

        query_embedding = self.encoder.encode([query])[0]

        similarities = []
        for entry in self.knowledge_base:
            if entry.embedding is not None:
                similarity = cosine_similarity([query_embedding], [entry.embedding])[0][0]
                similarities.append((similarity, entry))

        # Sort and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:top_k]]

    def identify_missing_context(self) -> List[str]:
        """Identify what context information is still missing"""
        missing = []

        # Check for critical missing context
        if not self.project_context.language_preference or self.project_context.language_preference == "english":
            if not any("language" in conv.get("user_input", "").lower() for conv in self.conversation_history):
                missing.append("language_preference")

        if not self.project_context.deployment_target:
            missing.append("deployment_target")

        if not self.project_context.team_size or self.project_context.team_size == "individual":
            if not any("team" in conv.get("user_input", "").lower() for conv in self.conversation_history):
                missing.append("team_size")

        if not self.project_context.code_style or self.project_context.code_style == "clean":
            if not any("style" in conv.get("user_input", "").lower() for conv in self.conversation_history):
                missing.append("code_style")

        return missing

    def get_context_gathering_question(self, missing_item: str) -> str:
        """Get a specific question to gather missing context"""
        context_questions = {
            "language_preference": "Should the code (variable names, comments, documentation) be in English or your native language? This affects maintainability and collaboration.",
            "deployment_target": "Where do you plan to deploy this - cloud platforms, local servers, or mobile devices?",
            "team_size": "Will you be working on this alone, with a small team, or as part of a larger organization?",
            "code_style": "Do you prefer minimal code with fewer comments, or comprehensive documentation and detailed comments?",
            "internationalization": "Do you need this system to work in multiple languages/regions?",
            "maintenance_requirements": "How long do you expect to maintain this project - short-term prototype or long-term production system?"
        }

        return context_questions.get(missing_item, "Can you tell me more about your project requirements?")

    def get_next_question(self) -> str:
        """Select the next Socratic question, prioritizing missing context"""
        # First, check if we're missing critical context
        missing_context = self.identify_missing_context()
        if missing_context:
            return self.get_context_gathering_question(missing_context[0])

        # Then proceed with phase-appropriate questions
        phase_questions = self.socratic_questions.get(self.current_phase, [])

        if not phase_questions:
            return "What else would you like to explore about your project?"

        # Simple strategy: random selection from phase questions
        import random
        return random.choice(phase_questions)

    def update_context_from_response(self, user_response: str):
        """Update project context from user response"""
        response_lower = user_response.lower()

        # Detect goals
        goal_indicators = ["want to", "goal", "purpose", "build", "create", "develop"]
        if any(indicator in response_lower for indicator in goal_indicators):
            self.project_context.goals.append(user_response)

        # Detect technologies
        tech_keywords = ["python", "javascript", "react", "django", "flask", "sql", "mongodb", "api"]
        mentioned_tech = [tech for tech in tech_keywords if tech in response_lower]
        self.project_context.tech_stack.extend(mentioned_tech)

        # Detect constraints
        constraint_indicators = ["can't", "cannot", "limitation", "budget", "time", "deadline"]
        if any(indicator in response_lower for indicator in constraint_indicators):
            self.project_context.constraints.append(user_response)

        # Detect language preferences
        if any(word in response_lower for word in ["english", "international", "global", "multilingual"]):
            self.project_context.language_preference = "english"
        elif any(word in response_lower for word in ["native", "local", "greek", "german", "french"]):
            self.project_context.language_preference = "native"

        # Detect team size
        if any(word in response_lower for word in ["team", "colleagues", "developers", "group"]):
            self.project_context.team_size = "team"
        elif any(word in response_lower for word in ["alone", "myself", "solo", "individual"]):
            self.project_context.team_size = "individual"

        # Detect deployment preferences
        if any(word in response_lower for word in ["cloud", "aws", "azure", "gcp", "heroku"]):
            self.project_context.deployment_target = "cloud"
        elif any(word in response_lower for word in ["local", "on-premise", "server"]):
            self.project_context.deployment_target = "local"
        elif any(word in response_lower for word in ["mobile", "app", "android", "ios"]):
            self.project_context.deployment_target = "mobile"

        # Detect code style preferences
        if any(word in response_lower for word in ["documented", "comments", "documentation", "detailed"]):
            self.project_context.code_style = "documented"
        elif any(word in response_lower for word in ["minimal", "clean", "simple", "lightweight"]):
            self.project_context.code_style = "minimal"
        elif any(word in response_lower for word in ["enterprise", "corporate", "standards"]):
            self.project_context.code_style = "enterprise"

    def determine_next_phase(self) -> str:
        """Determine next phase based on context completeness"""
        if len(self.project_context.goals) < 2 and self.current_phase == "discovery":
            return "discovery"
        elif len(self.project_context.requirements) < 3 and self.current_phase in ["discovery", "analysis"]:
            return "analysis"
        elif not self.project_context.tech_stack and self.current_phase in ["discovery", "analysis", "design"]:
            return "design"
        else:
            return "implementation"

    def generate_claude_prompt(self, user_input: str) -> str:
        """Generate prompt for Claude with enriched context"""
        # Retrieve relevant knowledge
        relevant_knowledge = self.retrieve_relevant_knowledge(user_input)

        # Build context string
        context_parts = []

        if self.project_context.goals:
            context_parts.append(f"Goals: {', '.join(self.project_context.goals)}")

        if self.project_context.tech_stack:
            context_parts.append(f"Technologies: {', '.join(self.project_context.tech_stack)}")

        if self.project_context.constraints:
            context_parts.append(f"Constraints: {', '.join(self.project_context.constraints)}")

        # Add project specifications
        context_parts.append(f"Language preference: {self.project_context.language_preference}")
        context_parts.append(f"Code style: {self.project_context.code_style}")
        context_parts.append(f"Team size: {self.project_context.team_size}")
        context_parts.append(f"Deployment: {self.project_context.deployment_target}")

        context_str = " | ".join(context_parts) if context_parts else "New project"

        # Relevant knowledge
        knowledge_str = ""
        if relevant_knowledge:
            knowledge_str = "\n\nRelevant knowledge from base:\n"
            for i, entry in enumerate(relevant_knowledge, 1):
                knowledge_str += f"{i}. {entry.content}\n"

        # Check for missing context
        missing_context = self.identify_missing_context()
        missing_str = ""
        if missing_context:
            missing_str = f"\n\nMissing context: {', '.join(missing_context)}"

        prompt = f"""You are an experienced software architect using Socratic maieutics to help developers.

Project context: {context_str}
Current phase: {self.current_phase}
{missing_str}

{knowledge_str}

User response: {user_input}

Based on the user's response:
1. If there are gaps or contradictions in their thinking, ask a targeted Socratic question
2. If critical project context is missing (especially language preferences, team structure, deployment), prioritize gathering that information
3. If the response is clear, provide brief guidance and suggest the next step
4. Use relevant knowledge from the base where helpful
5. Ensure all code suggestions align with their stated preferences (language, style, team size)

Keep responses concise and focused. Don't give ready-made solutions - help the user discover them.

IMPORTANT: Based on their language preference ({self.project_context.language_preference}) and team size ({self.project_context.team_size}), adjust code suggestions accordingly:
- If English preference or team work: use English variable names and comments
- If individual + native preference: ask if they prefer native language in code
- Always consider maintainability and collaboration needs"""

        return prompt

    def chat(self, user_input: str) -> str:
        """Main chat method"""
        # Update context
        self.update_context_from_response(user_input)

        # Generate Claude prompt
        prompt = self.generate_claude_prompt(user_input)

        try:
            # Call Claude API
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text

            # Update phase
            self.current_phase = self.determine_next_phase()

            # Store in history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "claude_response": response,
                "phase": self.current_phase,
                "context": asdict(self.project_context)
            })

            return response

        except Exception as e:
            return f"Communication error with Claude: {str(e)}"

    def start_conversation(self) -> str:
        """Initialize the dialogue"""
        return self.get_next_question()

    def get_project_summary(self) -> Dict[str, Any]:
        """Get project context summary"""
        return {
            "context": asdict(self.project_context),
            "current_phase": self.current_phase,
            "conversation_length": len(self.conversation_history)
        }


# Example usage
def main():
    """Example usage"""
    # Initialize (needs Claude API key)
    api_key = os.getenv("CLAUDE_API_KEY", claude_api_key)
    rag = SocraticRAG(api_key)

    print("🤖 Socratic Counselor for Project Development")
    print("=" * 50)

    # Start conversation
    first_question = rag.start_conversation()
    print(f"Assistant: {first_question}")

    # Interactive loop
    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit', 'end']:
            break

        if user_input.lower() == 'summary':
            summary = rag.get_project_summary()
            print(f"\n📊 Project summary:")
            print(f"Phase: {summary['current_phase']}")
            print(f"Goals: {summary['context']['goals']}")
            print(f"Technologies: {summary['context']['tech_stack']}")
            continue

        if not user_input:
            continue

        # Get response from the system
        response = rag.chat(user_input)
        print(f"\nAssistant: {response}")

    print("\n👋 Good luck with your project!")


if __name__ == "__main__":
    main()
