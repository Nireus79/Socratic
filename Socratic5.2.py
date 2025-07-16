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
    """Manages SQLite database operations with proper datetime handling"""

    def __init__(self, db_name: str = "socratic_projects.db"):
        self.db_name = db_name
        self.init_database()

    def _datetime_to_string(self, dt: datetime) -> str:
        """Convert datetime to ISO string"""
        return dt.isoformat()

    def _string_to_datetime(self, dt_str: str) -> datetime:
        """Convert ISO string to datetime"""
        return datetime.fromisoformat(dt_str)

    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Users table - using TEXT for timestamps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL
            )
        ''')

        # Projects table - using TEXT for timestamps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
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
                joined_at TEXT NOT NULL,
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
                updated_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        ''')

        # Conversations table - using TEXT for timestamps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                entry_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TEXT NOT NULL,
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
                created_at TEXT NOT NULL
            )
        ''')

        # User sessions table - using TEXT for timestamps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_user(self, username: str, email: str, password: str) -> str:
        """Create a new user"""
        user_id = str(uuid.uuid4())
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        now = self._datetime_to_string(datetime.now())

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO users (user_id, username, email, password_hash, created_at, last_active)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, email, password_hash, now, now))
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

        # Update last_active
        if result:
            user_id = result[0]
            now = self._datetime_to_string(datetime.now())
            cursor.execute('''
                UPDATE users SET last_active = ? WHERE user_id = ?
            ''', (now, user_id))
            conn.commit()

        conn.close()
        return result[0] if result else None

    def create_session(self, user_id: str, session_timeout: int = 24 * 60 * 60) -> str:
        """Create a new session for user"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(seconds=session_timeout)

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO user_sessions (session_id, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_id, self._datetime_to_string(now), self._datetime_to_string(expires_at)))

        conn.commit()
        conn.close()

        return session_id

    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return user_id"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT user_id, expires_at FROM user_sessions 
            WHERE session_id = ?
        ''', (session_id,))

        result = cursor.fetchone()

        if result:
            user_id, expires_at_str = result
            expires_at = self._string_to_datetime(expires_at_str)

            # Check if session is still valid
            if datetime.now() < expires_at:
                conn.close()
                return user_id
            else:
                # Session expired, delete it
                cursor.execute('DELETE FROM user_sessions WHERE session_id = ?', (session_id,))
                conn.commit()

        conn.close()
        return None

    def save_project(self, project_id: str, name: str, description: str, owner_id: str,
                     phase: str = "discovery", status: str = "active",
                     created_at: Optional[datetime] = None, updated_at: Optional[datetime] = None) -> None:
        """Save or update a project"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        now = datetime.now()
        created_at = created_at or now
        updated_at = updated_at or now

        cursor.execute('''
            INSERT OR REPLACE INTO projects 
            (project_id, name, description, owner_id, created_at, updated_at, phase, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (project_id, name, description, owner_id,
              self._datetime_to_string(created_at), self._datetime_to_string(updated_at),
              phase, status))

        conn.commit()
        conn.close()

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
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
            return {
                'project_id': result[0],
                'name': result[1],
                'description': result[2],
                'owner_id': result[3],
                'created_at': self._string_to_datetime(result[4]),
                'updated_at': self._string_to_datetime(result[5]),
                'phase': result[6],
                'status': result[7]
            }
        return None

    def get_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
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
            projects.append({
                'project_id': result[0],
                'name': result[1],
                'description': result[2],
                'owner_id': result[3],
                'created_at': self._string_to_datetime(result[4]),
                'updated_at': self._string_to_datetime(result[5]),
                'phase': result[6],
                'status': result[7]
            })

        return projects

    def save_project_context(self, project_id: str, context_data: Dict[str, Any]) -> None:
        """Save project context"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        context_json = json.dumps(context_data)
        now = self._datetime_to_string(datetime.now())

        cursor.execute('''
            INSERT OR REPLACE INTO project_contexts (project_id, context_data, updated_at)
            VALUES (?, ?, ?)
        ''', (project_id, context_json, now))

        conn.commit()
        conn.close()

    def get_project_context(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project context"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT context_data FROM project_contexts WHERE project_id = ?
        ''', (project_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return json.loads(result[0])
        return None

    def save_conversation(self, entry_id: str, project_id: str, user_id: str,
                          message: str, response: str, phase: str,
                          context_updates: Optional[Dict[str, Any]] = None,
                          timestamp: Optional[datetime] = None) -> None:
        """Save conversation entry"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        timestamp = timestamp or datetime.now()
        context_updates = context_updates or {}
        context_updates_json = json.dumps(context_updates)

        cursor.execute('''
            INSERT INTO conversations 
            (entry_id, project_id, user_id, message, response, timestamp, phase, context_updates)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (entry_id, project_id, user_id, message, response,
              self._datetime_to_string(timestamp), phase, context_updates_json))

        conn.commit()
        conn.close()

    def get_conversation_history(self, project_id: str, limit: int = 50) -> List[Dict[str, Any]]:
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
            conversations.append({
                'entry_id': result[0],
                'project_id': result[1],
                'user_id': result[2],
                'message': result[3],
                'response': result[4],
                'timestamp': self._string_to_datetime(result[5]),
                'phase': result[6],
                'context_updates': json.loads(result[7]) if result[7] else {}
            })

        return conversations

    def add_collaborator(self, project_id: str, user_id: str, role: str = "collaborator") -> None:
        """Add collaborator to project"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        now = self._datetime_to_string(datetime.now())

        cursor.execute('''
            INSERT OR REPLACE INTO project_collaborators (project_id, user_id, role, joined_at)
            VALUES (?, ?, ?, ?)
        ''', (project_id, user_id, role, now))

        conn.commit()
        conn.close()

    def get_user_by_username(self, username: str) -> Optional[str]:
        """Get user_id by username"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('SELECT user_id FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        now = self._datetime_to_string(datetime.now())
        cursor.execute('DELETE FROM user_sessions WHERE expires_at < ?', (now,))

        conn.commit()
        conn.close()

    def save_knowledge_entry(self, entry_id: str, content: str, category: str,
                             embedding: bytes, tags: List[str]) -> None:
        """Save knowledge base entry"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        now = self._datetime_to_string(datetime.now())
        tags_json = json.dumps(tags)

        cursor.execute('''
            INSERT OR REPLACE INTO knowledge_base 
            (entry_id, content, category, embedding, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (entry_id, content, category, embedding, tags_json, now))

        conn.commit()
        conn.close()

    def get_knowledge_entries(self) -> List[Dict[str, Any]]:
        """Get all knowledge base entries"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT entry_id, content, category, embedding, tags, created_at
            FROM knowledge_base
        ''')

        results = cursor.fetchall()
        conn.close()

        entries = []
        for result in results:
            entries.append({
                'entry_id': result[0],
                'content': result[1],
                'category': result[2],
                'embedding': result[3],  # Keep as bytes for now
                'tags': json.loads(result[4]),
                'created_at': self._string_to_datetime(result[5])
            })

        return entries


class EnhancedSocraticRAG:
    """Enhanced Socratic RAG system with multi-project and multi-user support"""

    def __init__(self, api_key: str = None):
        self.db = DatabaseManager()
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("API_KEY_CLAUDE"))
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

        # Save project using individual parameters instead of Project object
        self.db.save_project(
            project_id=project_id,
            name=name,
            description=description,
            owner_id=self.current_user_id,
            phase="discovery",
            status="active",
            created_at=now,
            updated_at=now
        )

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

        self.db.save_project_context(project_id, asdict(context))

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

    # def remove_collaborator(self, project_id: str, user_id: str) -> None:
    #     """Remove collaborator from project"""
    #     conn = sqlite3.connect(self.db_name)
    #     cursor = conn.cursor()
    #
    #     cursor.execute('''
    #         DELETE FROM project_collaborators
    #         WHERE project_id = ? AND user_id = ?
    #     ''', (project_id, user_id))
    #
    #     conn.commit()
    #     conn.close()
    #
    # def get_project_collaborators(self, project_id: str) -> List[Dict[str, Any]]:
    #     """Get all collaborators for a project"""
    #     conn = sqlite3.connect(self.db_name)
    #     cursor = conn.cursor()
    #
    #     cursor.execute('''
    #         SELECT u.user_id, u.username, u.email, pc.role, pc.joined_at
    #         FROM project_collaborators pc
    #         JOIN users u ON pc.user_id = u.user_id
    #         WHERE pc.project_id = ?
    #     ''', (project_id,))
    #
    #     results = cursor.fetchall()
    #     conn.close()
    #
    #     collaborators = []
    #     for result in results:
    #         collaborators.append({
    #             'user_id': result[0],
    #             'username': result[1],
    #             'email': result[2],
    #             'role': result[3],
    #             'joined_at': self._string_to_datetime(result[4])
    #         })
    #
    #     return collaborators

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
    You are a Socratic counselor helping with software development projects. 
    Use the Socratic method to guide the user through discovery rather than providing direct answers.

    Current Project Context:
    - Phase: {context.get('phase', 'discovery')}
    - Goals: {', '.join(context.get('goals', [])) if context.get('goals') else 'Not defined'}
    - Requirements: {', '.join(context.get('requirements', [])) if context.get('requirements') else 'Not defined'}
    - Tech Stack: {', '.join(context.get('tech_stack', [])) if context.get('tech_stack') else 'Not defined'}
    - Constraints: {', '.join(context.get('constraints', [])) if context.get('constraints') else 'None specified'}

    Recent Conversation:
    {chr(10).join([f"User: {entry['message']}" for entry in history[:3]])}

    Relevant Knowledge:
    {chr(10).join(relevant_context[:3])}

    User's current message: {message}

    Respond with a thoughtful Socratic question that helps the user think deeper about their project. Focus on the 
    current phase ({context.get('phase', 'discovery')}) and guide them toward the next insight or decision they need to make.
    """

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return (f"I'm having trouble processing your message right now. Can you tell me more about what you're "
                    f"trying to achieve with this project?")

    def update_context(self, message: str, response: str) -> None:
        """Update project context based on conversation"""
        context = self.db.get_project_context(self.current_project_id)
        if not context:
            return

        analysis = self.analyze_message(message)
        context_updates = {}

        # Extract goals
        if "goal" in message.lower() or "want to" in message.lower():
            if message not in context.get("goals", []):
                context.setdefault("goals", []).append(message)
                context_updates["goals"] = context["goals"]

        # Extract requirements
        if "need" in message.lower() or "require" in message.lower():
            if message not in context.get("requirements", []):
                context.setdefault("requirements", []).append(message)
                context_updates["requirements"] = context["requirements"]

        # Track user contributions
        if "user_contributions" not in context:
            context["user_contributions"] = {}
        if self.current_user_id not in context["user_contributions"]:
            context["user_contributions"][self.current_user_id] = []
        context["user_contributions"][self.current_user_id].append(message)

        # Update phase if needed
        if analysis["phase_indicators"]:
            new_phase = analysis["phase_indicators"][0]
            if new_phase != context.get("phase"):
                context["phase"] = new_phase
                context_updates["phase"] = new_phase

        # Save updated context
        self.db.save_project_context(self.current_project_id, context)

        # Save conversation entry - pass individual parameters
        self.db.save_conversation(
            entry_id=str(uuid.uuid4()),
            project_id=self.current_project_id,
            user_id=self.current_user_id,
            message=message,
            response=response,
            timestamp=datetime.now(),
            phase=context.get("phase", "discovery"),
            context_updates=context_updates
        )

    def get_project_summary(self) -> str:
        """Get project summary"""
        context = self.db.get_project_context(self.current_project_id)
        if not context:
            return "No project context available."

        project = self.db.get_project(self.current_project_id)
        if not project:
            return "Project not found."

        summary = f"""
    Project: {project['name']}
    Phase: {context.get('phase', 'discovery')}
    Status: {project['status']}

    Goals:
    {chr(10).join(f"• {goal}" for goal in context.get('goals', [])) if context.get('goals') else "• Not defined yet"}

    Requirements:
    {chr(10).join(f"• {req}" for req in context.get('requirements', [])) if context.get('requirements') else "• Not defined yet"}

    Tech Stack:
    {chr(10).join(f"• {tech}" for tech in context.get('tech_stack', [])) if context.get('tech_stack') else "• Not defined yet"}

    Constraints:
    {chr(10).join(f"• {constraint}" for constraint in context.get('constraints', [])) if context.get('constraints') else "• None specified"}

    Team Structure: {context.get('team_structure', 'individual')}

    Progress Markers:
    {chr(10).join(f"• {marker}" for marker in context.get('progress_markers', [])) if context.get('progress_markers') else "• None yet"}

    Contributors:
    {chr(10).join(f"• User {user_id}: {len(contributions)} contributions"
                  for user_id, contributions in context.get('user_contributions', {}).items())}
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
    print("Ουδέν οίδα, ούτε διδάσκω τι, αλλά διαπορώ μόνον.")
    print("Enhanced Socratic Counselor v5.2")
    print("Multi-Project & Multi-User Support")
    print("=" * 50)

    # Initialize system
    try:
        api_key = os.getenv("API_KEY_CLAUDE")
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

        # Project selection loop
        while True:
            projects = socratic.get_user_projects()

            if projects:
                print("\nYour Projects:")
                for i, project in enumerate(projects, 1):
                    print(f"{i}. {project['name']} ({project['phase']}) - {project['status']}")

                print(f"{len(projects) + 1}. Create new project")
                print(f"{len(projects) + 2}. Exit")

                choice = input(f"Choose project (1-{len(projects) + 2}): ")

                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(projects):
                        selected_project = projects[choice_num - 1]
                        socratic.select_project(selected_project['project_id'])
                        print(f"✅ Selected project: {selected_project['name']}")
                        break

                    elif choice_num == len(projects) + 1:
                        # Create new project
                        name = input("Project name: ")
                        description = input("Project description (optional): ")
                        project_id = socratic.create_project(name, description)
                        socratic.select_project(project_id)
                        print(f"✅ Created and selected project: {name}")
                        break

                    elif choice_num == len(projects) + 2:
                        return

                except ValueError:
                    print("❌ Invalid choice. Please enter a number.")

            else:
                # No projects, create first one
                print("\nNo projects found. Let's create your first project!")
                name = input("Project name: ")
                description = input("Project description (optional): ")
                project_id = socratic.create_project(name, description)
                socratic.select_project(project_id)
                print(f"✅ Created and selected project: {name}")
                break

        # Main conversation loop
        print("\n🚀 Socratic Counselor is ready!")
        print("Type 'help' for commands, 'summary' for project overview, or 'exit' to quit.")
        print("=" * 50)

        while True:
            try:
                user_message = input("\n💬 You: ").strip()

                if not user_message:
                    continue

                if user_message.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye! Your progress has been saved.")
                    break

                elif user_message.lower() == 'help':
                    print("""
Available commands:
• 'summary' - View project summary
• 'add collaborator <username>' - Add a collaborator
• 'list projects' - Switch to project selection
• 'help' - Show this help
• 'exit' - Quit the application

Just type your questions or thoughts to get Socratic guidance!
                    """)
                    continue

                elif user_message.lower() == 'list projects':
                    # Go back to project selection
                    print("\nSwitching to project selection...")
                    break

                # Process the message
                response = socratic.process_message(user_message)
                print(f"\n🤖 Socratic: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Your progress has been saved.")
                break

            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or contact support if the issue persists.")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        print("..τω Ασκληπιώ οφείλομεν αλετρυόνα, απόδοτε και μη αμελήσετε..")
        print("\n📊 Session ended. All progress has been automatically saved.")


if __name__ == "__main__":
    main()
