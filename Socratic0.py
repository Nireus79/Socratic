import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import anthropic
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sqlite3
from pathlib import Path
import threading

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
api_key = os.environ["API_KEY_CLAUDE"]


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
    # Project specifications (removed language_preference)
    target_audience: str = ""
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


@dataclass
class ConversationMessage:
    """Individual conversation message"""
    user_id: str
    message_id: str
    timestamp: datetime
    user_input: str
    assistant_response: str
    phase: str
    context_snapshot: Dict[str, Any]


class DatabaseManager:
    """Handles all database operations for persistent storage"""

    def __init__(self, db_path: str = "socratic_rag.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    current_phase TEXT DEFAULT 'discovery'
                )
            ''')

            # Project contexts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_contexts (
                    user_id TEXT PRIMARY KEY,
                    context_data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')

            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    message_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    timestamp TIMESTAMP,
                    user_input TEXT,
                    assistant_response TEXT,
                    phase TEXT,
                    context_snapshot TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')

            conn.commit()

    def create_or_get_user(self, user_id: str) -> Dict[str, Any]:
        """Create new user or get existing user info"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if user exists
                cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
                user = cursor.fetchone()

                if not user:
                    # Create new user
                    cursor.execute('''
                        INSERT INTO users (user_id, current_phase)
                        VALUES (?, 'discovery')
                    ''', (user_id,))
                    conn.commit()
                    return {"user_id": user_id, "current_phase": "discovery", "is_new": True}
                else:
                    # Update last active
                    cursor.execute('''
                        UPDATE users SET last_active = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                    ''', (user_id,))
                    conn.commit()
                    return {"user_id": user_id, "current_phase": user[3], "is_new": False}

    def save_project_context(self, user_id: str, context: ProjectContext):
        """Save project context for user"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                context_json = json.dumps(asdict(context))

                cursor.execute('''
                    INSERT OR REPLACE INTO project_contexts
                    (user_id, context_data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (user_id, context_json))
                conn.commit()

    def load_project_context(self, user_id: str) -> Optional[ProjectContext]:
        """Load project context for user"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT context_data FROM project_contexts
                    WHERE user_id = ?
                ''', (user_id,))

                result = cursor.fetchone()
                if result:
                    context_dict = json.loads(result[0])
                    return ProjectContext(**context_dict)
                return None

    def save_conversation_message(self, message: ConversationMessage):
        """Save conversation message"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations
                    (message_id, user_id, timestamp, user_input, assistant_response,
                     phase, context_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message.message_id,
                    message.user_id,
                    message.timestamp.isoformat(),
                    message.user_input,
                    message.assistant_response,
                    message.phase,
                    json.dumps(message.context_snapshot)
                ))
                conn.commit()

    def load_conversation_history(self, user_id: str, limit: int = 50) -> List[ConversationMessage]:
        """Load conversation history for user"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_id, user_id, timestamp, user_input,
                           assistant_response, phase, context_snapshot
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_id, limit))

                messages = []
                for row in cursor.fetchall():
                    messages.append(ConversationMessage(
                        message_id=row[0],
                        user_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        user_input=row[3],
                        assistant_response=row[4],
                        phase=row[5],
                        context_snapshot=json.loads(row[6])
                    ))

                return list(reversed(messages))  # Return in chronological order

    def update_user_phase(self, user_id: str, phase: str):
        """Update user's current phase"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users SET current_phase = ?, last_active = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (phase, user_id))
                conn.commit()

    def get_active_users(self, hours: int = 24) -> List[str]:
        """Get list of users active in the last N hours"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id FROM users
                    WHERE last_active > datetime('now', '-{} hours')
                '''.format(hours))

                return [row[0] for row in cursor.fetchall()]


class MultiUserSocraticRAG:
    def __init__(self, claude_api_key: str, knowledge_base_path: str = "knowledge_base.pkl",
                 db_path: str = "socratic_rag.db"):
        """
        Initialize the Multi-User Socratic RAG system

        Args:
            claude_api_key: API key for Claude
            knowledge_base_path: Path for storing the knowledge base
            db_path: Path for SQLite database
        """
        self.client = anthropic.Anthropic(api_key=claude_api_key)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base: List[KnowledgeEntry] = []
        self.knowledge_base_path = knowledge_base_path
        self.db = DatabaseManager(db_path)

        # User sessions cache (for performance)
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()

        # Enhanced Socratic questions (removed language preference questions)
        self.socratic_questions = {
            "discovery": [
                "What exactly do you want to achieve with this project?",
                "What is the main problem you're trying to solve?",
                "Who will be the end users of this system?",
                "What would success look like for this project?",
                "What are the core features it must have?",
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
                "Do you need comprehensive documentation for other developers?",
                "What level of code comments would be appropriate for your team?"
            ],
            "implementation": [
                "In what order will you implement the features?",
                "How will you track progress and milestones?",
                "What will you do if you get stuck on a particular issue?",
                "How will you handle deployment and release?",
                "How will you maintain the project after completion?",
                "Do you need the code to be easily readable by international developers?",
                "What documentation language would serve your team best?"
            ]
        }

        # Context validation rules (removed language_preference)
        self.context_requirements = {
            "code_style": ["minimal", "documented", "enterprise", "academic"],
            "deployment_target": ["cloud", "local", "mobile", "embedded", "web"],
            "team_size": ["individual", "small_team", "large_team", "open_source"],
            "experience_level": ["beginner", "intermediate", "advanced", "expert"]
        }

        # Load existing knowledge base
        self.load_knowledge_base()

    def get_or_create_user_session(self, user_id: str) -> Dict[str, Any]:
        """Get or create user session"""
        with self.session_lock:
            if user_id not in self.user_sessions:
                # Get user info from database
                user_info = self.db.create_or_get_user(user_id)

                # Load project context
                project_context = self.db.load_project_context(user_id)
                if not project_context:
                    project_context = ProjectContext()

                # Load conversation history
                conversation_history = self.db.load_conversation_history(user_id)

                self.user_sessions[user_id] = {
                    "project_context": project_context,
                    "current_phase": user_info["current_phase"],
                    "conversation_history": conversation_history,
                    "is_new_user": user_info["is_new"]
                }

            return self.user_sessions[user_id]

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
                "The Model-View-Controller (MVC) pattern separates the application into three components: Model (data),"
                "View (UI), Controller (logic). This helps with code maintainability.",
                "architecture", ["mvc", "design-patterns", "architecture"]),

            (
                "RESTful API design follows principles: stateless communication, uniform interface, cacheable responses."
                "Uses HTTP methods (GET, POST, PUT, DELETE) and status codes.",
                "api-design", ["rest", "api", "web-development"]),

            ("Database normalization reduces redundancy and improves data integrity. Main levels are 1NF, 2NF, 3NF.",
             "database", ["database", "normalization", "sql"]),

            (
                "Test-Driven Development (TDD): Write tests first, then code to pass tests, then refactor. "
                "Red-Green-Refactor cycle.",
                "testing", ["tdd", "testing", "methodology"]),

            (
                "Microservices architecture splits the application into small, independent services. Advantages: "
                "scalability, technology diversity. Disadvantages: complexity, network overhead.",
                "architecture", ["microservices", "architecture", "scalability"]),

            (
                "Clean code principles: meaningful names, small functions, consistent formatting, comprehensive "
                "comments for complex logic.",
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

    def identify_missing_context(self, project_context: ProjectContext) -> List[str]:
        """Identify what context information is still missing"""
        missing = []

        if not project_context.deployment_target:
            missing.append("deployment_target")

        if not project_context.team_size or project_context.team_size == "individual":
            missing.append("team_size")

        if not project_context.code_style or project_context.code_style == "clean":
            missing.append("code_style")

        return missing

    def get_context_gathering_question(self, missing_item: str) -> str:
        """Get a specific question to gather missing context"""
        context_questions = {
            "deployment_target": "Where do you plan to deploy this - cloud platforms, local servers, or mobile devices?",
            "team_size": "Will you be working on this alone, with a small team, or as part of a larger organization?",
            "code_style": "Do you prefer minimal code with fewer comments, or comprehensive documentation and "
                          "detailed comments?",
            "internationalization": "Do you need this system to work in multiple languages/regions?",
            "maintenance_requirements": "How long do you expect to maintain this project - short-term prototype or "
                                        "long-term production system?"
        }

        return context_questions.get(missing_item, "Can you tell me more about your project requirements?")

    def get_next_question(self, user_id: str) -> str:
        """Select the next Socratic question, prioritizing missing context"""
        session = self.get_or_create_user_session(user_id)
        project_context = session["project_context"]
        current_phase = session["current_phase"]

        # First, check if we're missing critical context
        missing_context = self.identify_missing_context(project_context)
        if missing_context:
            return self.get_context_gathering_question(missing_context[0])

        # Then proceed with phase-appropriate questions
        phase_questions = self.socratic_questions.get(current_phase, [])

        if not phase_questions:
            return "What else would you like to explore about your project?"

        # Simple strategy: random selection from phase questions
        import random
        return random.choice(phase_questions)

    def update_context_from_response(self, user_id: str, user_response: str):
        """Update project context from user response"""
        session = self.get_or_create_user_session(user_id)
        project_context = session["project_context"]
        response_lower = user_response.lower()

        # Detect goals
        goal_indicators = ["want to", "goal", "purpose", "build", "create", "develop"]
        if any(indicator in response_lower for indicator in goal_indicators):
            project_context.goals.append(user_response)

        # Detect technologies
        tech_keywords = ["python", "javascript", "react", "django", "flask", "sql", "mongodb", "api"]
        mentioned_tech = [tech for tech in tech_keywords if tech in response_lower]
        project_context.tech_stack.extend(mentioned_tech)

        # Detect constraints
        constraint_indicators = ["can't", "cannot", "limitation", "budget", "time", "deadline"]
        if any(indicator in response_lower for indicator in constraint_indicators):
            project_context.constraints.append(user_response)

        # Detect team size
        if any(word in response_lower for word in ["team", "colleagues", "developers", "group"]):
            project_context.team_size = "team"
        elif any(word in response_lower for word in ["alone", "myself", "solo", "individual"]):
            project_context.team_size = "individual"

        # Detect deployment preferences
        if any(word in response_lower for word in ["cloud", "aws", "azure", "gcp", "heroku"]):
            project_context.deployment_target = "cloud"
        elif any(word in response_lower for word in ["local", "on-premise", "server"]):
            project_context.deployment_target = "local"
        elif any(word in response_lower for word in ["mobile", "app", "android", "ios"]):
            project_context.deployment_target = "mobile"

        # Detect code style preferences
        if any(word in response_lower for word in ["documented", "comments", "documentation", "detailed"]):
            project_context.code_style = "documented"
        elif any(word in response_lower for word in ["minimal", "clean", "simple", "lightweight"]):
            project_context.code_style = "minimal"
        elif any(word in response_lower for word in ["enterprise", "corporate", "standards"]):
            project_context.code_style = "enterprise"

        # Save updated context to database
        self.db.save_project_context(user_id, project_context)

    def determine_next_phase(self, user_id: str) -> str:
        """Determine next phase based on context completeness"""
        session = self.get_or_create_user_session(user_id)
        project_context = session["project_context"]
        current_phase = session["current_phase"]

        if len(project_context.goals) < 2 and current_phase == "discovery":
            return "discovery"
        elif len(project_context.requirements) < 3 and current_phase in ["discovery", "analysis"]:
            return "analysis"
        elif not project_context.tech_stack and current_phase in ["discovery", "analysis", "design"]:
            return "design"
        else:
            return "implementation"

    def generate_claude_prompt(self, user_id: str, user_input: str) -> str:
        """Generate prompt for Claude with enriched context"""
        session = self.get_or_create_user_session(user_id)
        project_context = session["project_context"]
        current_phase = session["current_phase"]

        # Retrieve relevant knowledge
        relevant_knowledge = self.retrieve_relevant_knowledge(user_input)

        # Build context string
        context_parts = []

        if project_context.goals:
            context_parts.append(f"Goals: {', '.join(project_context.goals)}")

        if project_context.tech_stack:
            context_parts.append(f"Technologies: {', '.join(project_context.tech_stack)}")

        if project_context.constraints:
            context_parts.append(f"Constraints: {', '.join(project_context.constraints)}")

        # Add project specifications
        context_parts.append(f"Code style: {project_context.code_style}")
        context_parts.append(f"Team size: {project_context.team_size}")
        context_parts.append(f"Deployment: {project_context.deployment_target}")

        context_str = " | ".join(context_parts) if context_parts else "New project"

        # Relevant knowledge
        knowledge_str = ""
        if relevant_knowledge:
            knowledge_str = "\n\nRelevant knowledge from base:\n"
            for i, entry in enumerate(relevant_knowledge, 1):
                knowledge_str += f"{i}. {entry.content}\n"

        # Check for missing context
        missing_context = self.identify_missing_context(project_context)
        missing_str = ""
        if missing_context:
            missing_str = f"\n\nMissing context: {', '.join(missing_context)}"

        # Recent conversation context
        recent_messages = session["conversation_history"][-3:] if session["conversation_history"] else []
        history_str = ""
        if recent_messages:
            history_str = "\n\nRecent conversation:\n"
            for msg in recent_messages:
                history_str += f"User: {msg.user_input[:100]}...\n"
                history_str += f"Assistant: {msg.assistant_response[:100]}...\n\n"

        prompt = f"""You are an experienced software architect using Socratic maieutics to help developers.

Project context: {context_str}
Current phase: {current_phase}
{missing_str}

{knowledge_str}

{history_str}

Current user response: {user_input}

Based on the user's response:
1. If there are gaps or contradictions in their thinking, ask a targeted Socratic question
2. If critical project context is missing, prioritize gathering that information
3. If the response is clear, provide brief guidance and suggest the next step
4. Use relevant knowledge from the base where helpful
5. Ensure all code suggestions align with their stated preferences (style, team size)

Keep responses concise and focused. Don't give ready-made solutions - help the user discover them.

Consider their code style ({project_context.code_style}) and team size ({project_context.team_size}) preferences when
making suggestions."""

        return prompt

    def chat(self, user_id: str, user_input: str) -> str:
        """Main chat method for specific user"""
        # Update context
        self.update_context_from_response(user_id, user_input)

        # Generate Claude prompt
        prompt = self.generate_claude_prompt(user_id, user_input)

        try:
            # Call Claude API
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text

            # Update phase
            new_phase = self.determine_next_phase(user_id)
            session = self.get_or_create_user_session(user_id)
            session["current_phase"] = new_phase

            # Update phase in database
            self.db.update_user_phase(user_id, new_phase)

            # Create conversation message
            conversation_message = ConversationMessage(
                user_id=user_id,
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                user_input=user_input,
                assistant_response=response,
                phase=new_phase,
                context_snapshot=asdict(session["project_context"])
            )

            # Save to database
            self.db.save_conversation_message(conversation_message)

            # Add to session history
            session["conversation_history"].append(conversation_message)

            return response

        except Exception as e:
            return f"Communication error with Claude: {str(e)}"

    def start_conversation(self, user_id: str) -> str:
        """Initialize the dialogue for a user"""
        session = self.get_or_create_user_session(user_id)

        if session["is_new_user"]:
            return ("Hello! I'm here to help you develop your software project using the Socratic method. Let's start "
                    "by understanding what you want to build. What exactly do you want to achieve with this project?")
        else:
            # Returning user
            return (f"Welcome back! I can see we were in the {session['current_phase']} phase. "
                    + self.get_next_question(user_id))

    def get_project_summary(self, user_id: str) -> Dict[str, Any]:
        """Get project context summary for specific user"""
        session = self.get_or_create_user_session(user_id)
        return {
            "user_id": user_id,
            "context": asdict(session["project_context"]),
            "current_phase": session["current_phase"],
            "conversation_length": len(session["conversation_history"])
        }

    def get_active_users(self, hours: int = 24) -> List[str]:
        """Get list of recently active users"""
        return self.db.get_active_users(hours)

    def cleanup_old_sessions(self, hours: int = 72):
        """Clean up inactive user sessions from memory"""
        with self.session_lock:
            active_users = set(self.get_active_users(hours))
            inactive_users = set(self.user_sessions.keys()) - active_users

            for user_id in inactive_users:
                del self.user_sessions[user_id]

            print(f"Cleaned up {len(inactive_users)} inactive sessions")


# Example usage with multiple users
def main():
    """Example usage with multiple users"""
    # Initialize (needs Claude API key)
    rag = MultiUserSocraticRAG(api_key)

    print("🤖 Multi-User Socratic Counselor for Project Development")
    print("=" * 60)

    # Simulate multiple users or single user session
    while True:
        print("Ουδέν οίδα ούτε διδάσκω τι, αλλά διαπορώ μόνον.")
        print("\nCommands:")
        print("1. Start/continue conversation (enter user ID)")
        print("2. 'users' - Show active users")
        print("3. 'summary <user_id>' - Get project summary")
        print("4. 'quit' - Exit")

        command = input("\nEnter command or user ID: ").strip()

        if command.lower() == 'quit':
            break
        elif command.lower() == 'users':
            active_users = rag.get_active_users()
            print(f"Active users: {active_users}")
            continue
        elif command.lower().startswith('summary '):
            user_id = command.split(' ', 1)[1]
            summary = rag.get_project_summary(user_id)
            print(f"\n📊 Project summary for {user_id}:")
            print(f"Phase: {summary['current_phase']}")
            print(f"Goals: {summary['context']['goals']}")
            print(f"Technologies: {summary['context']['tech_stack']}")
            continue

        # Treat as user ID
        user_id = command
        if not user_id:
            continue

        print(f"\n--- Conversation with user: {user_id} ---")

        # Start conversation for this user
        first_response = rag.start_conversation(user_id)
        print(f"Assistant: {first_response}")

        # Interactive loop for this user
        while True:
            user_input = input(f"\n{user_id}: ").strip()

            if user_input.lower() in ['quit', 'exit', 'switch']:
                break

            if not user_input:
                continue

            # Get response from the system
            response = rag.chat(user_id, user_input)
            print(f"\nAssistant: {response}")

    print("\n👋 Good luck with your projects!")


if __name__ == "__main__":
    main()


# Assistant: I understand you need to leave. When you return, I'd like to explore this key question:
#
# What specific metrics or characteristics would indicate that your synthetic data is sufficiently representative of
# real market behavior for your validation purposes?
#
# This will help us evaluate your current validation approach and potentially identify additional validation methods.
#
# Let me know when you're ready to continue the discussion.
#
# Themis: