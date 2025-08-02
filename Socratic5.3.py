#!/usr/bin/env python3
"""
Enhanced Socratic System v5.3
Using ContextAnalysisAgent
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


class ContextAnalysisAgent:
    """Intelligent context analysis using Claude as an agent"""

    def __init__(self, client):
        self.client = client

    def analyze_context(self, message: str, conversation_summary: str, current_context: Dict) -> Dict:
        """Analyze message context using Claude as an intelligent agent"""

        prompt = f"""
    You are a context analysis agent for a software development project. Analyze the user's message and return structured information.

    CURRENT PROJECT CONTEXT:
    {json.dumps(current_context, indent=2)}

    CONVERSATION HISTORY SUMMARY:
    {conversation_summary}

    USER'S NEW MESSAGE:
    "{message}"

    Analyze this message and return a JSON object with:

    {{
        "intent": "one of: providing_requirement|clarifying_goal|asking_question|providing_feedback|changing_direction|expressing_concern",
        "confidence": 0.8,
        "extracted_entities": {{
            "technologies": ["list of tech mentions"],
            "requirements": ["specific requirements mentioned"],
            "constraints": ["limitations or constraints"],
            "goals": ["objectives or goals"],
            "concerns": ["worries or issues raised"]
        }},
        "context_updates": {{
            "should_update_goals": true/false,
            "should_update_requirements": true/false,
            "should_update_tech_stack": true/false,
            "should_update_constraints": true/false
        }},
        "conversation_analysis": {{
            "is_repeating_previous_info": true/false,
            "builds_on_previous": true/false,
            "introduces_new_topic": true/false,
            "shows_confusion": true/false
        }},
        "recommended_response_focus": "what the Socratic response should focus on",
        "phase_progression": {{
            "current_phase_appropriate": true/false,
            "suggested_phase": "discovery|analysis|design|implementation",
            "readiness_for_next_phase": 0.7
        }},
        "priority_level": "high|medium|low",
        "follow_up_suggestions": ["suggested next questions to explore"]
    }}

    Return ONLY the JSON object, no other text.
    """

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",  # Use Haiku for faster, cheaper analysis
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            analysis = json.loads(response.content[0].text)
            return analysis

        except Exception as e:
            # Fallback to simple analysis if agent fails
            return self._fallback_analysis(message)

    def _fallback_analysis(self, message: str) -> Dict:
        """Simple fallback analysis if Claude agent fails"""
        message_lower = message.lower()

        return {
            "intent": "providing_requirement" if any(
                word in message_lower for word in ["need", "want", "require"]) else "asking_question",
            "confidence": 0.5,
            "extracted_entities": {
                "technologies": [],
                "requirements": [message] if "need" in message_lower else [],
                "constraints": [],
                "goals": [message] if "goal" in message_lower else [],
                "concerns": []
            },
            "context_updates": {
                "should_update_goals": "goal" in message_lower,
                "should_update_requirements": "need" in message_lower,
                "should_update_tech_stack": False,
                "should_update_constraints": False
            },
            "conversation_analysis": {
                "is_repeating_previous_info": False,
                "builds_on_previous": True,
                "introduces_new_topic": True,
                "shows_confusion": False
            },
            "recommended_response_focus": "clarify the user's input",
            "phase_progression": {
                "current_phase_appropriate": True,
                "suggested_phase": "discovery",
                "readiness_for_next_phase": 0.5
            },
            "priority_level": "medium",
            "follow_up_suggestions": ["Ask for more details"]
        }


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

    def build_conversation_summary(self, project_id: str) -> str:
        """Build a comprehensive summary of previous conversations"""
        history = self.db.get_conversation_history(project_id, 50)  # Get more history

        if not history:
            return "No previous conversations."

        # Group conversations by topic/phase
        summary_parts = []

        # Extract established facts
        established_facts = {
            'goals': set(),
            'requirements': set(),
            'tech_preferences': set(),
            'constraints': set(),
            'decisions_made': []
        }

        conversation_summary = []

        for entry in reversed(history):  # Process chronologically
            message = entry['message'].lower()
            response = entry['response']

            # Extract established information
            if any(keyword in message for keyword in ['goal', 'want to', 'trying to']):
                established_facts['goals'].add(entry['message'])

            if any(keyword in message for keyword in ['need', 'require', 'must have']):
                established_facts['requirements'].add(entry['message'])

            if any(keyword in message for keyword in ['use', 'prefer', 'language', 'framework']):
                established_facts['tech_preferences'].add(entry['message'])

            if any(keyword in message for keyword in ['cannot', 'limit', 'constraint', 'budget']):
                established_facts['constraints'].add(entry['message'])

            # Keep recent conversation flow
            if len(conversation_summary) < 6:  # Last 6 exchanges
                conversation_summary.append(f"User: {entry['message']}")
                conversation_summary.append(f"Assistant: {entry['response']}")

        # Build summary
        summary_parts.append("=== ESTABLISHED PROJECT FACTS ===")

        if established_facts['goals']:
            summary_parts.append("GOALS DISCUSSED:")
            for goal in list(established_facts['goals'])[:5]:
                summary_parts.append(f"• {goal}")

        if established_facts['requirements']:
            summary_parts.append("\nREQUIREMENTS IDENTIFIED:")
            for req in list(established_facts['requirements'])[:5]:
                summary_parts.append(f"• {req}")

        if established_facts['tech_preferences']:
            summary_parts.append("\nTECH PREFERENCES MENTIONED:")
            for tech in list(established_facts['tech_preferences'])[:5]:
                summary_parts.append(f"• {tech}")

        if established_facts['constraints']:
            summary_parts.append("\nCONSTRAINTS IDENTIFIED:")
            for constraint in list(established_facts['constraints'])[:3]:
                summary_parts.append(f"• {constraint}")

        summary_parts.append("\n=== RECENT CONVERSATION FLOW ===")
        summary_parts.extend(conversation_summary)

        return "\n".join(summary_parts)

    def get_unanswered_questions(self, project_id: str) -> List[str]:
        """Identify what key questions still need to be answered"""
        context = self.db.get_project_context(project_id)
        if not context:
            return ["What is the main goal of your project?"]

        history = self.db.get_conversation_history(project_id, 30)
        discussed_topics = set()

        # Track what's been discussed
        for entry in history:
            message = entry['message'].lower()
            if any(word in message for word in ['goal', 'purpose', 'objective']):
                discussed_topics.add('goals')
            if any(word in message for word in ['user', 'audience', 'customer']):
                discussed_topics.add('target_users')
            if any(word in message for word in ['feature', 'function', 'capability']):
                discussed_topics.add('features')
            if any(word in message for word in ['data', 'database', 'storage']):
                discussed_topics.add('data_requirements')
            if any(word in message for word in ['deploy', 'host', 'platform']):
                discussed_topics.add('deployment')
            if any(word in message for word in ['timeline', 'deadline', 'schedule']):
                discussed_topics.add('timeline')

        # Essential questions that should be answered
        essential_questions = {
            'goals': "What specific problem does your project solve?",
            'target_users': "Who will be using this application?",
            'features': "What are the core features you need?",
            'data_requirements': "What kind of data will your application handle?",
            'deployment': "Where do you plan to deploy this application?",
            'timeline': "What's your timeline for this project?"
        }

        unanswered = []
        for topic, question in essential_questions.items():
            if topic not in discussed_topics:
                unanswered.append(question)

        return unanswered

    def analyze_message(self, message: str) -> Dict[str, Any]:
        """Enhanced message analysis using the context agent"""
        context = self.db.get_project_context(self.current_project_id)
        conversation_summary = self.build_conversation_summary(self.current_project_id)

        # Use the context agent for intelligent analysis
        if not hasattr(self, 'context_agent'):
            self.context_agent = ContextAnalysisAgent(self.client)

        return self.context_agent.analyze_context(message, conversation_summary, context or {})

    def generate_socratic_response(self, message: str) -> str:
        """Generate Socratic response using agent-enhanced context understanding"""
        context = self.db.get_project_context(self.current_project_id)
        if not context:
            return "Let's start by understanding your project. What exactly do you want to achieve?"

        # Get intelligent context analysis
        analysis = self.analyze_message(message)

        # Get conversation summary
        conversation_summary = self.build_conversation_summary(self.current_project_id)

        # Get relevant knowledge
        relevant_context = self.get_relevant_context(message)

        # Build enhanced prompt using agent insights
        prompt = f"""
    You are a Socratic counselor helping with software development projects. 
    Use the Socratic method to guide the user through discovery rather than providing direct answers.

    CONTEXT ANALYSIS FROM INTELLIGENT AGENT:
    - User's Intent: {analysis.get('intent', 'unknown')}
    - Confidence: {analysis.get('confidence', 0)}
    - Priority Level: {analysis.get('priority_level', 'medium')}
    - Is Repeating Info: {analysis.get('conversation_analysis', {}).get('is_repeating_previous_info', False)}
    - Shows Confusion: {analysis.get('conversation_analysis', {}).get('shows_confusion', False)}
    - Recommended Focus: {analysis.get('recommended_response_focus', 'general discussion')}

    EXTRACTED ENTITIES:
    - Technologies: {', '.join(analysis.get('extracted_entities', {}).get('technologies', []))}
    - Requirements: {', '.join(analysis.get('extracted_entities', {}).get('requirements', []))}
    - Goals: {', '.join(analysis.get('extracted_entities', {}).get('goals', []))}
    - Concerns: {', '.join(analysis.get('extracted_entities', {}).get('concerns', []))}

    PHASE ANALYSIS:
    - Current Phase: {context.get('phase', 'discovery')}
    - Suggested Phase: {analysis.get('phase_progression', {}).get('suggested_phase', 'discovery')}
    - Ready for Next Phase: {analysis.get('phase_progression', {}).get('readiness_for_next_phase', 0)}

    CONVERSATION HISTORY & ESTABLISHED FACTS:
    {conversation_summary}

    SUGGESTED FOLLOW-UP QUESTIONS:
    {chr(10).join(f"• {q}" for q in analysis.get('follow_up_suggestions', []))}

    USER'S CURRENT MESSAGE: {message}

    INSTRUCTIONS:
    1. Use the agent analysis to understand the user's true intent
    2. If they're repeating info, acknowledge it and build forward
    3. If they show confusion, clarify gently
    4. Focus on the recommended response area
    5. Use suggested follow-up questions as inspiration
    6. Respect the priority level - high priority needs immediate attention

    Generate a thoughtful Socratic response that leverages the intelligent context analysis.
    """

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"I'm having trouble processing your message right now. Can you tell me more about {analysis.get('recommended_response_focus', 'your project')}?"

    def update_context(self, message: str, response: str) -> None:
        """Enhanced context update using agent analysis"""
        context = self.db.get_project_context(self.current_project_id)
        if not context:
            return

        # Get agent analysis (reuse from previous call if available)
        analysis = self.analyze_message(message)
        context_updates = {}

        # Use agent recommendations for context updates
        entities = analysis.get('extracted_entities', {})
        should_update = analysis.get('context_updates', {})

        # Update based on agent analysis
        if should_update.get('should_update_goals') and entities.get('goals'):
            goals = context.setdefault("goals", [])
            for goal in entities['goals']:
                if goal not in goals:
                    goals.append(goal)
            context_updates["goals"] = goals

        if should_update.get('should_update_requirements') and entities.get('requirements'):
            requirements = context.setdefault("requirements", [])
            for req in entities['requirements']:
                if req not in requirements:
                    requirements.append(req)
            context_updates["requirements"] = requirements

        if should_update.get('should_update_tech_stack') and entities.get('technologies'):
            tech_stack = context.setdefault("tech_stack", [])
            for tech in entities['technologies']:
                if tech not in tech_stack:
                    tech_stack.append(tech)
            context_updates["tech_stack"] = tech_stack

        if should_update.get('should_update_constraints') and entities.get('constraints'):
            constraints = context.setdefault("constraints", [])
            for constraint in entities['constraints']:
                if constraint not in constraints:
                    constraints.append(constraint)
            context_updates["constraints"] = constraints

        # Track user contributions with agent insights
        if "user_contributions" not in context:
            context["user_contributions"] = {}
        if self.current_user_id not in context["user_contributions"]:
            context["user_contributions"][self.current_user_id] = []

        contribution = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "phase": context.get("phase", "discovery"),
            "intent": analysis.get('intent'),
            "priority": analysis.get('priority_level'),
            "confidence": analysis.get('confidence'),
            "entities_extracted": entities
        }

        context["user_contributions"][self.current_user_id].append(contribution)

        # Smart phase progression using agent analysis
        phase_analysis = analysis.get('phase_progression', {})
        if not phase_analysis.get('current_phase_appropriate', True):
            suggested_phase = phase_analysis.get('suggested_phase')
            readiness = phase_analysis.get('readiness_for_next_phase', 0)

            if readiness > 0.7 and suggested_phase:
                context["phase"] = suggested_phase
                context_updates["phase"] = suggested_phase
                progress_markers = context.setdefault("progress_markers", [])
                progress_markers.append(
                    f"Agent-recommended advancement to {suggested_phase} phase (readiness: {readiness})")
                context_updates["progress_markers"] = progress_markers

        # Save updated context
        self.db.save_project_context(self.current_project_id, context)

        # Save conversation with agent analysis
        enhanced_context_updates = {**context_updates, "agent_analysis": analysis}

        self.db.save_conversation(
            entry_id=str(uuid.uuid4()),
            project_id=self.current_project_id,
            user_id=self.current_user_id,
            message=message,
            response=response,
            timestamp=datetime.now(),
            phase=context.get("phase", "discovery"),
            context_updates=enhanced_context_updates
        )

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
    print("Enhanced Socratic Counselor v5.3")
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
