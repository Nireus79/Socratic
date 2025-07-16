#!/usr/bin/env python3
"""
Enhanced Socratic System v5.2
Multi-Project & Multi-User Support with Persistent Progress Saving

Features:
- Multi-project management
- Multi-user support per project
- Persistent progress saving with SQLite database
- Enhanced context tracking
- User authentication system
- Project collaboration features
"""

import json
import sqlite3
import hashlib
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

# Configuration
DATABASE_NAME = "socratic_projects.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SESSION_TIMEOUT = 24 * 60 * 60  # 24 hours in seconds


@dataclass
class User:
    """User information"""
    user_id: str
    username: str
    email: str
    created_at: datetime
    last_active: datetime


@dataclass
class Project:
    """Project information"""
    project_id: str
    name: str
    description: str
    owner_id: str
    created_at: datetime
    updated_at: datetime
    phase: str = "discovery"
    status: str = "active"


@dataclass
class ProjectContext:
    """Enhanced project context with multi-user support"""
    project_id: str
    goals: List[str]
    requirements: List[str]
    tech_stack: List[str]
    constraints: List[str]
    team_structure: str
    language_preferences: List[str]
    deployment_target: str
    code_style_preferences: Dict[str, Any]
    phase: str
    progress_markers: List[str]
    user_contributions: Dict[str, List[str]]  # user_id -> contributions


@dataclass
class ConversationEntry:
    """Individual conversation entry"""
    entry_id: str
    project_id: str
    user_id: str
    message: str
    response: str
    timestamp: datetime
    phase: str
    context_updates: Dict[str, Any]


@dataclass
class KnowledgeEntry:
    """Knowledge base entry with embeddings"""
    entry_id: str
    content: str
    category: str
    embedding: np.ndarray
    tags: List[str]
    created_at: datetime


class DatabaseManager:
    """Manages SQLite database operations"""

    def __init__(self, db_name: str = DATABASE_NAME):
        self.db_name = db_name
        self.init_database()

    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                phase TEXT DEFAULT 'discovery',
                status TEXT DEFAULT 'active',
                FOREIGN KEY (owner_id) REFERENCES users (user_id)
            )
        ''')

        # Project collaborators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_collaborators (
                project_id TEXT,
                user_id TEXT,
                role TEXT DEFAULT 'collaborator',
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (project_id, user_id),
                FOREIGN KEY (project_id) REFERENCES projects (project_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # Project contexts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_contexts (
                project_id TEXT PRIMARY KEY,
                context_data TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        ''')

        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                entry_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                phase TEXT NOT NULL,
                context_updates TEXT,
                FOREIGN KEY (project_id) REFERENCES projects (project_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # Knowledge base table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                entry_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                embedding BLOB NOT NULL,
                tags TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_user(self, username: str, email: str, password: str) -> str:
        """Create a new user"""
        user_id = str(uuid.uuid4())
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (user_id, username, email, password_hash))
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            raise ValueError("Username or email already exists")
        finally:
            conn.close()

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return user_id"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT user_id FROM users 
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def create_session(self, user_id: str) -> str:
        """Create a new session for user"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(seconds=SESSION_TIMEOUT)

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO user_sessions (session_id, user_id, expires_at)
            VALUES (?, ?, ?)
        ''', (session_id, user_id, expires_at))

        conn.commit()
        conn.close()

        return session_id

    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return user_id"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT user_id FROM user_sessions 
            WHERE session_id = ? AND expires_at > CURRENT_TIMESTAMP
        ''', (session_id,))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def save_project(self, project: Project) -> None:
        """Save or update a project"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO projects 
            (project_id, name, description, owner_id, created_at, updated_at, phase, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (project.project_id, project.name, project.description, project.owner_id,
              project.created_at, project.updated_at, project.phase, project.status))

        conn.commit()
        conn.close()

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT project_id, name, description, owner_id, created_at, updated_at, phase, status
            FROM projects WHERE project_id = ?
        ''', (project_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return Project(
                project_id=result[0],
                name=result[1],
                description=result[2],
                owner_id=result[3],
                created_at=datetime.fromisoformat(result[4]),
                updated_at=datetime.fromisoformat(result[5]),
                phase=result[6],
                status=result[7]
            )
        return None

    def get_user_projects(self, user_id: str) -> List[Project]:
        """Get all projects for a user (owned or collaborated)"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT p.project_id, p.name, p.description, p.owner_id, 
                   p.created_at, p.updated_at, p.phase, p.status
            FROM projects p
            LEFT JOIN project_collaborators pc ON p.project_id = pc.project_id
            WHERE p.owner_id = ? OR pc.user_id = ?
            ORDER BY p.updated_at DESC
        ''', (user_id, user_id))

        results = cursor.fetchall()
        conn.close()

        projects = []
        for result in results:
            projects.append(Project(
                project_id=result[0],
                name=result[1],
                description=result[2],
                owner_id=result[3],
                created_at=datetime.fromisoformat(result[4]),
                updated_at=datetime.fromisoformat(result[5]),
                phase=result[6],
                status=result[7]
            ))

        return projects

    def save_project_context(self, project_id: str, context: ProjectContext) -> None:
        """Save project context"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        context_data = json.dumps(asdict(context))

        cursor.execute('''
            INSERT OR REPLACE INTO project_contexts (project_id, context_data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (project_id, context_data))

        conn.commit()
        conn.close()

    def get_project_context(self, project_id: str) -> Optional[ProjectContext]:
        """Get project context"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT context_data FROM project_contexts WHERE project_id = ?
        ''', (project_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            context_dict = json.loads(result[0])
            return ProjectContext(**context_dict)
        return None

    def save_conversation(self, entry: ConversationEntry) -> None:
        """Save conversation entry"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        context_updates = json.dumps(entry.context_updates)

        cursor.execute('''
            INSERT INTO conversations 
            (entry_id, project_id, user_id, message, response, timestamp, phase, context_updates)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (entry.entry_id, entry.project_id, entry.user_id, entry.message,
              entry.response, entry.timestamp, entry.phase, context_updates))

        conn.commit()
        conn.close()

    def get_conversation_history(self, project_id: str, limit: int = 50) -> List[ConversationEntry]:
        """Get conversation history for a project"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT entry_id, project_id, user_id, message, response, timestamp, phase, context_updates
            FROM conversations 
            WHERE project_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (project_id, limit))

        results = cursor.fetchall()
        conn.close()

        conversations = []
        for result in results:
            conversations.append(ConversationEntry(
                entry_id=result[0],
                project_id=result[1],
                user_id=result[2],
                message=result[3],
                response=result[4],
                timestamp=datetime.fromisoformat(result[5]),
                phase=result[6],
                context_updates=json.loads(result[7]) if result[7] else {}
            ))

        return conversations

    def add_collaborator(self, project_id: str, user_id: str, role: str = "collaborator") -> None:
        """Add collaborator to project"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO project_collaborators (project_id, user_id, role)
            VALUES (?, ?, ?)
        ''', (project_id, user_id, role))

        conn.commit()
        conn.close()


class EnhancedSocraticRAG:
    """Enhanced Socratic RAG system with multi-project and multi-user support"""

    def __init__(self, api_key: str = None):
        self.db = DatabaseManager()
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("CLAUDE_API_KEY"))
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.current_user_id = None
        self.current_project_id = None
        self.knowledge_base = self._load_knowledge_base()

        # Development phases
        self.phases = {
            "discovery": "Understanding project goals and requirements",
            "analysis": "Examining challenges and technical considerations",
            "design": "Planning architecture and implementation approach",
            "implementation": "Guidance on execution and deployment"
        }

    def _load_knowledge_base(self) -> Dict[str, KnowledgeEntry]:
        """Load knowledge base from database"""
        conn = sqlite3.connect(self.db.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT entry_id, content, category, embedding, tags, created_at
            FROM knowledge_base
        ''')

        results = cursor.fetchall()
        conn.close()

        knowledge_base = {}
        for result in results:
            embedding = np.frombuffer(result[3], dtype=np.float32)
            tags = json.loads(result[4])

            knowledge_base[result[0]] = KnowledgeEntry(
                entry_id=result[0],
                content=result[1],
                category=result[2],
                embedding=embedding,
                tags=tags,
                created_at=datetime.fromisoformat(result[5])
            )

        return knowledge_base

    def register_user(self, username: str, email: str, password: str) -> str:
        """Register a new user"""
        return self.db.create_user(username, email, password)

    def login(self, username: str, password: str) -> str:
        """Login user and return session ID"""
        user_id = self.db.authenticate_user(username, password)
        if user_id:
            self.current_user_id = user_id
            return self.db.create_session(user_id)
        return None

    def validate_session(self, session_id: str) -> bool:
        """Validate session and set current user"""
        user_id = self.db.validate_session(session_id)
        if user_id:
            self.current_user_id = user_id
            return True
        return False

    def create_project(self, name: str, description: str = "") -> str:
        """Create a new project"""
        if not self.current_user_id:
            raise ValueError("User must be logged in to create projects")

        project_id = str(uuid.uuid4())
        now = datetime.now()

        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            owner_id=self.current_user_id,
            created_at=now,
            updated_at=now
        )

        self.db.save_project(project)

        # Initialize project context
        context = ProjectContext(
            project_id=project_id,
            goals=[],
            requirements=[],
            tech_stack=[],
            constraints=[],
            team_structure="individual",
            language_preferences=[],
            deployment_target="",
            code_style_preferences={},
            phase="discovery",
            progress_markers=[],
            user_contributions={}
        )

        self.db.save_project_context(project_id, context)

        return project_id

    def select_project(self, project_id: str) -> bool:
        """Select a project to work on"""
        if not self.current_user_id:
            return False

        project = self.db.get_project(project_id)
        if project:
            self.current_project_id = project_id
            return True
        return False

    def get_user_projects(self) -> List[Project]:
        """Get all projects for current user"""
        if not self.current_user_id:
            return []

        return self.db.get_user_projects(self.current_user_id)

    def add_collaborator(self, project_id: str, username: str) -> bool:
        """Add a collaborator to a project"""
        if not self.current_user_id:
            return False

        # Get user_id from username
        conn = sqlite3.connect(self.db.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT user_id FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()

        if result:
            user_id = result[0]
            self.db.add_collaborator(project_id, user_id)
            return True
        return False

    def get_relevant_context(self, query: str, limit: int = 5) -> List[str]:
        """Get relevant context from knowledge base"""
        if not self.knowledge_base:
            return []

        query_embedding = self.embedding_model.encode([query])

        similarities = []
        for entry_id, entry in self.knowledge_base.items():
            similarity = cosine_similarity(query_embedding, [entry.embedding])[0][0]
            similarities.append((similarity, entry.content))

        similarities.sort(reverse=True)
        return [content for _, content in similarities[:limit]]

    def analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze user message to extract intent and context"""
        context = self.db.get_project_context(self.current_project_id)
        if not context:
            return {"intent": "setup", "extracted_info": {}}

        # Simple keyword-based analysis (can be enhanced with NLP)
        analysis = {
            "intent": "discuss",
            "extracted_info": {},
            "phase_indicators": []
        }

        # Check for phase transitions
        if any(word in message.lower() for word in ["implement", "code", "build", "develop"]):
            analysis["phase_indicators"].append("implementation")
        elif any(word in message.lower() for word in ["design", "architecture", "plan"]):
            analysis["phase_indicators"].append("design")
        elif any(word in message.lower() for word in ["analyze", "consider", "challenge"]):
            analysis["phase_indicators"].append("analysis")

        return analysis

    def generate_socratic_response(self, message: str) -> str:
        """Generate Socratic response using Claude"""
        context = self.db.get_project_context(self.current_project_id)
        if not context:
            return "Let's start by understanding your project. What exactly do you want to achieve?"

        # Get recent conversation history
        history = self.db.get_conversation_history(self.current_project_id, 10)

        # Get relevant knowledge
        relevant_context = self.get_relevant_context(message)

        # Build prompt
        prompt = f"""
You are a Socratic counselor helping with software development projects. Use the Socratic method to guide the user through discovery rather than providing direct answers.

Current Project Context:
- Phase: {context.phase}
- Goals: {', '.join(context.goals) if context.goals else 'Not defined'}
- Requirements: {', '.join(context.requirements) if context.requirements else 'Not defined'}
- Tech Stack: {', '.join(context.tech_stack) if context.tech_stack else 'Not defined'}
- Constraints: {', '.join(context.constraints) if context.constraints else 'None specified'}

Recent Conversation:
{chr(10).join([f"User: {entry.message}" for entry in history[:3]])}

Relevant Knowledge:
{chr(10).join(relevant_context[:3])}

User's current message: {message}

Respond with a thoughtful Socratic question that helps the user think deeper about their project. Focus on the current phase ({context.phase}) and guide them toward the next insight or decision they need to make.
"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"I'm having trouble processing your message right now. Can you tell me more about what you're trying to achieve with this project?"

    def update_context(self, message: str, response: str) -> None:
        """Update project context based on conversation"""
        context = self.db.get_project_context(self.current_project_id)
        if not context:
            return

        analysis = self.analyze_message(message)
        context_updates = {}

        # Extract goals
        if "goal" in message.lower() or "want to" in message.lower():
            if message not in context.goals:
                context.goals.append(message)
                context_updates["goals"] = context.goals

        # Extract requirements
        if "need" in message.lower() or "require" in message.lower():
            if message not in context.requirements:
                context.requirements.append(message)
                context_updates["requirements"] = context.requirements

        # Track user contributions
        if self.current_user_id not in context.user_contributions:
            context.user_contributions[self.current_user_id] = []
        context.user_contributions[self.current_user_id].append(message)

        # Update phase if needed
        if analysis["phase_indicators"]:
            new_phase = analysis["phase_indicators"][0]
            if new_phase != context.phase:
                context.phase = new_phase
                context_updates["phase"] = new_phase

        # Save updated context
        self.db.save_project_context(self.current_project_id, context)

        # Save conversation entry
        entry = ConversationEntry(
            entry_id=str(uuid.uuid4()),
            project_id=self.current_project_id,
            user_id=self.current_user_id,
            message=message,
            response=response,
            timestamp=datetime.now(),
            phase=context.phase,
            context_updates=context_updates
        )

        self.db.save_conversation(entry)

    def get_project_summary(self) -> str:
        """Get project summary"""
        context = self.db.get_project_context(self.current_project_id)
        if not context:
            return "No project context available."

        project = self.db.get_project(self.current_project_id)

        summary = f"""
Project: {project.name}
Phase: {context.phase}
Status: {project.status}

Goals:
{chr(10).join(f"• {goal}" for goal in context.goals) if context.goals else "• Not defined yet"}

Requirements:
{chr(10).join(f"• {req}" for req in context.requirements) if context.requirements else "• Not defined yet"}

Tech Stack:
{chr(10).join(f"• {tech}" for tech in context.tech_stack) if context.tech_stack else "• Not defined yet"}

Constraints:
{chr(10).join(f"• {constraint}" for constraint in context.constraints) if context.constraints else "• None specified"}

Team Structure: {context.team_structure}

Progress Markers:
{chr(10).join(f"• {marker}" for marker in context.progress_markers) if context.progress_markers else "• None yet"}

Contributors:
{chr(10).join(f"• User {user_id}: {len(contributions)} contributions" for user_id, contributions in context.user_contributions.items())}
"""
        return summary

    def process_message(self, message: str) -> str:
        """Process user message and return response"""
        if not self.current_user_id:
            return "Please log in first to use the Socratic counselor."

        if not self.current_project_id:
            return "Please select a project first. Use 'list projects' to see available projects."

        # Handle special commands
        if message.lower() in ["summary", "project summary"]:
            return self.get_project_summary()

        if message.lower().startswith("add collaborator"):
            username = message.split()[-1]
            if self.add_collaborator(self.current_project_id, username):
                return f"Successfully added {username} as a collaborator."
            return f"Could not add {username}. User may not exist."

        # Generate Socratic response
        response = self.generate_socratic_response(message)

        # Update context
        self.update_context(message, response)

        return response


def main():
    """Main application loop"""
    print("🤖 Enhanced Socratic Counselor v5.2")
    print("Multi-Project & Multi-User Support")
    print("=" * 50)

    # Initialize system
    try:
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            api_key = input("Enter your Claude API key: ")

        socratic = EnhancedSocraticRAG(api_key)

        # User authentication
        while True:
            print("\n1. Login")
            print("2. Register")
            print("3. Exit")

            choice = input("Choose option (1-3): ")

            if choice == "1":
                username = input("Username: ")
                password = input("Password: ")
                session_id = socratic.login(username, password)
                if session_id:
                    print(f"✅ Login successful! Session: {session_id[:8]}...")
                    break
                else:
                    print("❌ Login failed. Please try again.")

            elif choice == "2":
                username = input("Username: ")
                email = input("Email: ")
                password = input("Password: ")
                try:
                    user_id = socratic.register_user(username, email, password)
                    session_id = socratic.login(username, password)
                    print(f"✅ Registration successful! Auto-logged in.")
                    break
                except ValueError as e:
                    print(f"❌ Registration failed: {e}")

            elif choice == "3":
                return

        # Project selection
        while True:
            projects = socratic.get_user_projects()

            if projects:
                print("\nYour Projects:")
                for i, project in enumerate(projects, 1):
                    print(f"{i}. {project.name} ({project.phase}) - {project.status}")

                print(f"{len(projects) + 1}. Create new project")
                print(f"{len(projects) + 2}. Exit")

                choice = input(f"Choose project (1-{len(projects) + 2}): ")

                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(projects):
                        selected_project = projects