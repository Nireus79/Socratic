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
    project_name: str = ""  # NEW: Project name
    goals: List[str] = None
    requirements: List[str] = None
    constraints: List[str] = None
    tech_stack: List[str] = None
    experience_level: str = "beginner"
    domain: str = ""
    timeline: str = ""
    current_challenges: List[str] = None
    # Project specifications
    target_audience: str = ""
    code_style: str = "clean"
    deployment_target: str = ""
    team_size: str = "individual"
    maintenance_requirements: str = ""
    internationalization: bool = False
    created_at: str = ""  # NEW: Project creation timestamp
    last_updated: str = ""  # NEW: Last update timestamp

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
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()


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
    project_id: str  # NEW: Project identifier
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

            # Users table - ENHANCED
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    current_project_id TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')

            # NEW: Projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    project_name TEXT,
                    context_data TEXT,
                    current_phase TEXT DEFAULT 'discovery',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')

            # Remove old project_contexts table and update conversations
            cursor.execute('DROP TABLE IF EXISTS project_contexts')

            # Enhanced conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations_new (
                    message_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    project_id TEXT,
                    timestamp TIMESTAMP,
                    user_input TEXT,
                    assistant_response TEXT,
                    phase TEXT,
                    context_snapshot TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (project_id) REFERENCES projects (project_id)
                )
            ''')

            # Migrate old conversations if they exist
            try:
                cursor.execute('''
                    INSERT INTO conversations_new
                    (message_id, user_id, project_id, timestamp, user_input, assistant_response, phase, context_snapshot)
                    SELECT message_id, user_id, 'default_project', timestamp, user_input, assistant_response, phase, context_snapshot
                    FROM conversations
                ''')
                cursor.execute('DROP TABLE conversations')
            except sqlite3.OperationalError:
                pass  # Table doesn't exist

            cursor.execute('ALTER TABLE conversations_new RENAME TO conversations')

            conn.commit()

    def create_user(self, user_id: str, display_name: str = None) -> Dict[str, Any]:
        """Create new user"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                display_name = display_name or user_id

                cursor.execute('''
                    INSERT INTO users (user_id, display_name)
                    VALUES (?, ?)
                ''', (user_id, display_name))
                conn.commit()

                return {"user_id": user_id, "display_name": display_name, "is_new": True}

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, display_name, current_project_id, is_active
                    FROM users WHERE user_id = ? AND is_active = 1
                ''', (user_id,))

                result = cursor.fetchone()
                if result:
                    return {
                        "user_id": result[0],
                        "display_name": result[1],
                        "current_project_id": result[2],
                        "is_active": bool(result[3])
                    }
                return None

    def update_user_name(self, user_id: str, new_display_name: str) -> bool:
        """Update user's display name"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users SET display_name = ?, last_active = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND is_active = 1
                ''', (new_display_name, user_id))
                conn.commit()
                return cursor.rowcount > 0

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user (soft delete)"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Deactivate user and their projects
                cursor.execute('UPDATE users SET is_active = 0 WHERE user_id = ?', (user_id,))
                cursor.execute('UPDATE projects SET is_active = 0 WHERE user_id = ?', (user_id,))
                conn.commit()
                return cursor.rowcount > 0

    def create_project(self, user_id: str, project_name: str, context: ProjectContext = None) -> str:
        """Create new project for user"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                project_id = str(uuid.uuid4())
                context = context or ProjectContext(project_name=project_name)
                context.project_name = project_name
                context_json = json.dumps(asdict(context))

                cursor.execute('''
                    INSERT INTO projects
                    (project_id, user_id, project_name, context_data)
                    VALUES (?, ?, ?, ?)
                ''', (project_id, user_id, project_name, context_json))

                # Update user's current project
                cursor.execute('''
                    UPDATE users SET current_project_id = ?, last_active = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (project_id, user_id))

                conn.commit()
                return project_id

    def get_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all projects for a user"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT project_id, project_name, current_phase, created_at, updated_at
                    FROM projects
                    WHERE user_id = ? AND is_active = 1
                    ORDER BY updated_at DESC
                ''', (user_id,))

                return [
                    {
                        "project_id": row[0],
                        "project_name": row[1],
                        "current_phase": row[2],
                        "created_at": row[3],
                        "updated_at": row[4]
                    }
                    for row in cursor.fetchall()
                ]

    def switch_project(self, user_id: str, project_id: str) -> bool:
        """Switch user's current project"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Verify project belongs to user
                cursor.execute('''
                    SELECT 1 FROM projects
                    WHERE project_id = ? AND user_id = ? AND is_active = 1
                ''', (project_id, user_id))

                if not cursor.fetchone():
                    return False

                cursor.execute('''
                    UPDATE users SET current_project_id = ?, last_active = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (project_id, user_id))
                conn.commit()
                return True

    def load_project_context(self, project_id: str) -> Optional[ProjectContext]:
        """Load project context"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT context_data FROM projects
                    WHERE project_id = ? AND is_active = 1
                ''', (project_id,))

                result = cursor.fetchone()
                if result:
                    context_dict = json.loads(result[0])
                    return ProjectContext(**context_dict)
                return None

    def save_project_context(self, project_id: str, context: ProjectContext):
        """Save project context"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                context.last_updated = datetime.now().isoformat()
                context_json = json.dumps(asdict(context))

                cursor.execute('''
                    UPDATE projects
                    SET context_data = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE project_id = ?
                ''', (context_json, project_id))
                conn.commit()

    def update_project_phase(self, project_id: str, phase: str):
        """Update project's current phase"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE projects SET current_phase = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE project_id = ?
                ''', (phase, project_id))
                conn.commit()

    def save_conversation_message(self, message: ConversationMessage):
        """Save conversation message"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations
                    (message_id, user_id, project_id, timestamp, user_input, assistant_response,
                     phase, context_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message.message_id,
                    message.user_id,
                    message.project_id,
                    message.timestamp.isoformat(),
                    message.user_input,
                    message.assistant_response,
                    message.phase,
                    json.dumps(message.context_snapshot)
                ))
                conn.commit()

    def load_conversation_history(self, project_id: str, limit: int = 50) -> List[ConversationMessage]:
        """Load conversation history for project"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT message_id, user_id, project_id, timestamp, user_input,
                           assistant_response, phase, context_snapshot
                    FROM conversations
                    WHERE project_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (project_id, limit))

                messages = []
                for row in cursor.fetchall():
                    messages.append(ConversationMessage(
                        message_id=row[0],
                        user_id=row[1],
                        project_id=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        user_input=row[4],
                        assistant_response=row[5],
                        phase=row[6],
                        context_snapshot=json.loads(row[7])
                    ))

                return list(reversed(messages))

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all active users"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, display_name, last_active, current_project_id
                    FROM users
                    WHERE is_active = 1
                    ORDER BY last_active DESC
                ''')

                return [
                    {
                        "user_id": row[0],
                        "display_name": row[1],
                        "last_active": row[2],
                        "current_project_id": row[3]
                    }
                    for row in cursor.fetchall()
                ]


class MultiUserSocraticRAG:
    def __init__(self, claude_api_key: str, knowledge_base_path: str = "knowledge_base.pkl",
                 db_path: str = "socratic_rag.db"):
        """
        Initialize the Multi-User Socratic RAG system with enhanced user management
        """
        self.client = anthropic.Anthropic(api_key=claude_api_key)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base: List[KnowledgeEntry] = []
        self.knowledge_base_path = knowledge_base_path
        self.db = DatabaseManager(db_path)

        # User sessions cache (for performance)
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()

        # Enhanced Socratic questions
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

        # Context validation rules
        self.context_requirements = {
            "code_style": ["minimal", "documented", "enterprise", "academic"],
            "deployment_target": ["cloud", "local", "mobile", "embedded", "web"],
            "team_size": ["individual", "small_team", "large_team", "open_source"],
            "experience_level": ["beginner", "intermediate", "advanced", "expert"]
        }

        # Load existing knowledge base
        self.load_knowledge_base()

    # USER MANAGEMENT METHODS
    def create_user(self, user_id: str, display_name: str = None) -> Dict[str, Any]:
        """Create a new user"""
        return self.db.create_user(user_id, display_name)

    def update_user_name(self, user_id: str, new_name: str) -> bool:
        """Update user's display name"""
        success = self.db.update_user_name(user_id, new_name)
        if success and user_id in self.user_sessions:
            # Update cached session
            self.user_sessions[user_id]["display_name"] = new_name
        return success

    def remove_user(self, user_id: str) -> bool:
        """Remove/deactivate a user"""
        success = self.db.deactivate_user(user_id)
        if success and user_id in self.user_sessions:
            # Remove from cache
            with self.session_lock:
                del self.user_sessions[user_id]
        return success

    def list_users(self) -> List[Dict[str, Any]]:
        """Get list of all users"""
        return self.db.get_all_users()

    # PROJECT MANAGEMENT METHODS
    def create_project(self, user_id: str, project_name: str) -> Optional[str]:
        """Create new project for user"""
        user = self.db.get_user(user_id)
        if not user:
            return None

        project_id = self.db.create_project(user_id, project_name)

        # Update cached session
        if user_id in self.user_sessions:
            self.user_sessions[user_id]["current_project_id"] = project_id

        return project_id

    def switch_project(self, user_id: str, project_id: str) -> bool:
        """Switch user's current project"""
        success = self.db.switch_project(user_id, project_id)
        if success and user_id in self.user_sessions:
            # Update cached session
            self.user_sessions[user_id]["current_project_id"] = project_id
            # Reload project context
            project_context = self.db.load_project_context(project_id)
            self.user_sessions[user_id]["project_context"] = project_context
        return success

    def list_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's projects"""
        return self.db.get_user_projects(user_id)

    def get_or_create_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get or create user session with current project"""
        with self.session_lock:
            if user_id not in self.user_sessions:
                # Get user info from database
                user_info = self.db.get_user(user_id)
                if not user_info:
                    return None

                current_project_id = user_info["current_project_id"]

                # If no current project, create a default one
                if not current_project_id:
                    current_project_id = self.db.create_project(
                        user_id,
                        f"{user_info['display_name']}'s Project"
                    )

                # Load project context
                project_context = self.db.load_project_context(current_project_id)
                if not project_context:
                    project_context = ProjectContext(project_name=f"{user_info['display_name']}'s Project")

                # Load conversation history
                conversation_history = self.db.load_conversation_history(current_project_id)

                self.user_sessions[user_id] = {
                    "user_info": user_info,
                    "current_project_id": current_project_id,
                    "project_context": project_context,
                    "conversation_history": conversation_history
                }

            return self.user_sessions[user_id]

    # KNOWLEDGE BASE METHODS
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

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:top_k]]

    # CONTEXT MANAGEMENT METHODS
    def update_context_from_response(self, user_id: str, user_response: str):
        """Enhanced context update - considers previous information"""
        session = self.get_or_create_user_session(user_id)
        if not session:
            return

        project_context = session["project_context"]
        response_lower = user_response.lower()

        # Track what was added in this response
        additions = []

        # Detect goals (avoid duplicates)
        goal_indicators = ["want to", "goal", "purpose", "build", "create", "develop", "aim to"]
        if any(indicator in response_lower for indicator in goal_indicators):
            if user_response not in project_context.goals:
                project_context.goals.append(user_response)
                additions.append(f"Goal: {user_response[:50]}...")

        # Detect requirements
        req_indicators = ["need", "must", "should", "require", "essential", "important"]
        if any(indicator in response_lower for indicator in req_indicators):
            if user_response not in project_context.requirements:
                project_context.requirements.append(user_response)
                additions.append(f"Requirement: {user_response[:50]}...")

        # Detect technologies (avoid duplicates)
        tech_keywords = {
            "python": "Python", "javascript": "JavaScript", "react": "React",
            "django": "Django", "flask": "Flask", "sql": "SQL", "mongodb": "MongoDB",
            "api": "API", "node": "Node.js", "vue": "Vue.js", "angular": "Angular",
            "docker": "Docker", "kubernetes": "Kubernetes", "aws": "AWS", "azure": "Azure"
        }

        for keyword, tech_name in tech_keywords.items():
            if keyword in response_lower and tech_name not in project_context.tech_stack:
                project_context.tech_stack.append(tech_name)
                additions.append(f"Technology: {tech_name}")

        # Detect constraints
        constraint_indicators = ["can't", "cannot", "limitation", "budget", "time", "deadline", "limited"]
        if any(indicator in response_lower for indicator in constraint_indicators):
            if user_response not in project_context.constraints:
                project_context.constraints.append(user_response)
                additions.append(f"Constraint: {user_response[:50]}...")

        # Detect challenges
        challenge_indicators = ["difficult", "challenge", "problem", "issue", "struggle", "hard", "complex"]
        if any(indicator in response_lower for indicator in challenge_indicators):
            if user_response not in project_context.current_challenges:
                project_context.current_challenges.append(user_response)
                additions.append(f"Challenge: {user_response[:50]}...")

        # Update other context attributes
        if any(word in response_lower for word in ["team", "colleagues", "developers", "group"]):
            if "large" in response_lower or "big" in response_lower:
                project_context.team_size = "large_team"
            elif "small" in response_lower:
                project_context.team_size = "small_team"
            else:
                project_context.team_size = "small_team"
        elif any(word in response_lower for word in ["alone", "myself", "solo", "individual"]):
            project_context.team_size = "individual"

        # Save updated context to database
        self.db.save_project_context(session["current_project_id"], project_context)

        # Log what was learned (for debugging/transparency)
        if additions:
            print(f"🧠 Learned from {session['user_info']['display_name']}: {', '.join(additions)}")

    def identify_missing_context(self, project_context: ProjectContext) -> List[str]:
        """Identify what context information is still missing"""
        missing = []

        if not project_context.deployment_target:
            missing.append("deployment_target")
        if not project_context.team_size or project_context.team_size == "individual":
            missing.append("team_size")
        if not project_context.code_style or project_context.code_style == "clean":
            missing.append("code_style")
        if not project_context.goals:
            missing.append("goals")
        if not project_context.tech_stack:
            missing.append("tech_stack")

        return missing

    def get_context_gathering_question(self, missing_item: str) -> str:
        """Get a specific question to gather missing context"""
        context_questions = {
            "deployment_target": "Where do you plan to deploy this - cloud platforms, local servers, or mobile devices?",
            "team_size": "Will you be working on this alone, with a small team, or as part of a larger organization?",
            "code_style": "Do you prefer minimal code with fewer comments, or comprehensive documentation and detailed comments?",
            "goals": "What are the main goals you want to achieve with this project?",
            "tech_stack": "What technologies or programming languages are you considering for this project?",
            "internationalization": "Do you need this system to work in multiple languages/regions?",
            "maintenance_requirements": "How long do you expect to maintain this project - short-term prototype or long-term production system?"
        }

        return context_questions.get(missing_item, "Can you tell me more about your project requirements?")

    def determine_next_phase(self, user_id: str) -> str:
        """Determine next phase based on context completeness"""
        session = self.get_or_create_user_session(user_id)
        if not session:
            return "discovery"

        project_context = session["project_context"]

        if len(project_context.goals) < 2:
            return "discovery"
        elif len(project_context.requirements) < 3:
            return "analysis"
        elif not project_context.tech_stack:
            return "analysis"
        elif len(project_context.tech_stack) > 0 and not project_context.deployment_target:
            return "design"
        else:
            return "implementation"

    # CONVERSATION METHODS
    def generate_socratic_response(self, user_id: str, user_input: str) -> str:
        """Generate Socratic response using RAG and context"""
        session = self.get_or_create_user_session(user_id)
        if not session:
            return "I'm sorry, I couldn't access your session. Please try again."

        project_context = session["project_context"]
        conversation_history = session["conversation_history"]

        # Update context based on user input
        self.update_context_from_response(user_id, user_input)

        # Determine current phase
        current_phase = self.determine_next_phase(user_id)

        # Update project phase in database
        self.db.update_project_phase(session["current_project_id"], current_phase)

        # Retrieve relevant knowledge
        relevant_knowledge = self.retrieve_relevant_knowledge(user_input)

        # Build context for Claude
        context_summary = self.build_context_summary(project_context)
        knowledge_context = self.build_knowledge_context(relevant_knowledge)

        # Get conversation history context
        history_context = ""
        if len(conversation_history) > 0:
            recent_messages = conversation_history[-3:]  # Last 3 exchanges
            history_context = "\n".join([
                f"User: {msg.user_input}\nAssistant: {msg.assistant_response}"
                for msg in recent_messages
            ])

        # Build the prompt
        prompt = self.build_claude_prompt(
            user_input, context_summary, knowledge_context,
            current_phase, history_context, session["user_info"]["display_name"]
        )

        try:
            # Generate response using Claude
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            assistant_response = response.content[0].text

            # Save conversation
            message = ConversationMessage(
                user_id=user_id,
                project_id=session["current_project_id"],
                message_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                user_input=user_input,
                assistant_response=assistant_response,
                phase=current_phase,
                context_snapshot=asdict(project_context)
            )

            self.db.save_conversation_message(message)

            # Update session history
            session["conversation_history"].append(message)

            return assistant_response

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."

    def build_context_summary(self, context: ProjectContext) -> str:
        """Build a summary of the current project context"""
        summary = f"Project: {context.project_name}\n"

        if context.goals:
            summary += f"Goals: {'; '.join(context.goals[:3])}\n"

        if context.requirements:
            summary += f"Requirements: {'; '.join(context.requirements[:3])}\n"

        if context.tech_stack:
            summary += f"Technologies: {', '.join(context.tech_stack)}\n"

        if context.constraints:
            summary += f"Constraints: {'; '.join(context.constraints[:2])}\n"

        if context.current_challenges:
            summary += f"Challenges: {'; '.join(context.current_challenges[:2])}\n"

        summary += f"Experience Level: {context.experience_level}\n"
        summary += f"Team Size: {context.team_size}\n"
        summary += f"Code Style: {context.code_style}\n"

        if context.deployment_target:
            summary += f"Deployment: {context.deployment_target}\n"

        return summary

    def build_knowledge_context(self, knowledge_entries: List[KnowledgeEntry]) -> str:
        """Build knowledge context from retrieved entries"""
        if not knowledge_entries:
            return "No specific knowledge retrieved."

        context = "Relevant Knowledge:\n"
        for i, entry in enumerate(knowledge_entries, 1):
            context += f"{i}. [{entry.category}] {entry.content}\n"

        return context

    def build_claude_prompt(self, user_input: str, context_summary: str,
                            knowledge_context: str, phase: str, history_context: str,
                            user_name: str) -> str:
        """Build the complete prompt for Claude"""

        missing_context = self.identify_missing_context(
            self.user_sessions[list(self.user_sessions.keys())[0]]["project_context"]
            if self.user_sessions else ProjectContext()
        )

        prompt = f"""You are an expert software development mentor using the Socratic method to guide {user_name} through their project.

CURRENT PROJECT CONTEXT:
{context_summary}

{knowledge_context}

CURRENT PHASE: {phase}

RECENT CONVERSATION:
{history_context}

USER'S CURRENT INPUT: {user_input}

INSTRUCTIONS:
1. Use the Socratic method - ask thoughtful questions that lead {user_name} to discover solutions
2. Draw from the relevant knowledge provided when appropriate
3. Focus on the current phase: {phase}
4. Consider the project context when formulating questions
5. If {user_name} seems stuck, provide gentle guidance but still encourage critical thinking
6. Keep responses concise but insightful
7. Match the complexity of your language to their experience level
8. Consider their team size and deployment target when giving advice

"""

        # Add phase-specific guidance
        if phase == "discovery":
            prompt += "Focus on helping them clarify their goals, requirements, and project scope.\n"
        elif phase == "analysis":
            prompt += "Help them analyze requirements, identify challenges, and consider alternatives.\n"
        elif phase == "design":
            prompt += "Guide them through architectural decisions and design considerations.\n"
        elif phase == "implementation":
            prompt += "Support them with implementation strategies and best practices.\n"

        # Add context-gathering questions if needed
        if missing_context:
            prompt += f"\nMISSING CONTEXT to consider asking about: {', '.join(missing_context)}\n"

        prompt += f"\nRespond as their mentor, using the Socratic method to guide their learning:"

        return prompt

    def get_project_summary(self, user_id: str) -> str:
        """Get a summary of the current project"""
        session = self.get_or_create_user_session(user_id)
        if not session:
            return "No active project found."

        context = session["project_context"]
        project_name = context.project_name

        summary = f"📋 **{project_name}** Summary:\n\n"

        if context.goals:
            summary += f"🎯 **Goals:** {', '.join(context.goals[:3])}\n"

        if context.tech_stack:
            summary += f"⚙️ **Technologies:** {', '.join(context.tech_stack)}\n"

        if context.requirements:
            summary += f"📝 **Requirements:** {', '.join(context.requirements[:3])}\n"

        if context.current_challenges:
            summary += f"🚧 **Challenges:** {', '.join(context.current_challenges[:2])}\n"

        summary += f"👥 **Team:** {context.team_size}\n"
        summary += f"🎨 **Code Style:** {context.code_style}\n"

        if context.deployment_target:
            summary += f"🚀 **Deployment:** {context.deployment_target}\n"

        # Add phase and progress info
        current_phase = self.determine_next_phase(user_id)
        summary += f"🔄 **Current Phase:** {current_phase.title()}\n"

        return summary

    def reset_project_context(self, user_id: str) -> bool:
        """Reset the current project context"""
        session = self.get_or_create_user_session(user_id)
        if not session:
            return False

        # Create new context with same project name
        old_context = session["project_context"]
        new_context = ProjectContext(project_name=old_context.project_name)

        # Save to database
        self.db.save_project_context(session["current_project_id"], new_context)

        # Update session
        session["project_context"] = new_context

        return True

    def export_project_data(self, user_id: str) -> Dict[str, Any]:
        """Export all project data for a user"""
        session = self.get_or_create_user_session(user_id)
        if not session:
            return {}

        return {
            "user_info": session["user_info"],
            "project_context": asdict(session["project_context"]),
            "conversation_history": [
                {
                    "timestamp": msg.timestamp.isoformat(),
                    "user_input": msg.user_input,
                    "assistant_response": msg.assistant_response,
                    "phase": msg.phase
                }
                for msg in session["conversation_history"]
            ],
            "projects": self.list_user_projects(user_id)
        }

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user"""
        session = self.get_or_create_user_session(user_id)
        if not session:
            return {}

        history = session["conversation_history"]
        context = session["project_context"]

        return {
            "total_messages": len(history),
            "project_name": context.project_name,
            "goals_defined": len(context.goals),
            "requirements_gathered": len(context.requirements),
            "technologies_identified": len(context.tech_stack),
            "challenges_identified": len(context.current_challenges),
            "current_phase": self.determine_next_phase(user_id),
            "project_created": context.created_at,
            "last_updated": context.last_updated
        }


# USAGE EXAMPLE AND MAIN INTERFACE
def main():
    """Main interface for the Multi-User Socratic RAG system"""

    # Initialize system
    rag = MultiUserSocraticRAG(claude_api_key=api_key)

    print("🧠 Multi-User Socratic RAG System Initialized")
    print("Ουδέν οίδα ούτε διδάσκω τι, αλλά διαπορώ μόνον.")
    print(
        "Commands: create_user, switch_user, new_project, switch_project, chat, summary, users, projects, stats, "
        "export, reset, help, quit")

    current_user = None

    while True:
        try:
            if current_user:
                user_info = rag.db.get_user(current_user)
                display_name = user_info["display_name"] if user_info else current_user
                command = input(f"\n[{display_name}] Enter command: ").strip().lower()
            else:
                command = input("\n[No User] Enter command: ").strip().lower()

            if command == "quit" or command == "exit":
                print("Goodbye! 👋")
                break

            elif command == "help":
                print("""
📚 Available Commands:
• create_user - Create a new user account
• switch_user - Switch to a different user
• new_project - Create a new project
• switch_project - Switch to a different project
• chat - Start/continue conversation
• summary - View current project summary
• users - List all users
• projects - List current user's projects
• stats - View user statistics
• export - Export project data
• reset - Reset current project context
• help - Show this help
• quit - Exit the system
                """)

            elif command == "create_user":
                user_id = input("Enter user ID: ").strip()
                display_name = input("Enter display name (optional): ").strip()

                try:
                    result = rag.create_user(user_id, display_name or None)
                    print(f"✅ User created: {result['display_name']}")
                    current_user = user_id
                except Exception as e:
                    print(f"❌ Error creating user: {e}")

            elif command == "switch_user":
                users = rag.list_users()
                if not users:
                    print("No users found. Create a user first.")
                    continue

                print("\n👥 Available Users:")
                for i, user in enumerate(users, 1):
                    print(f"{i}. {user['display_name']} ({user['user_id']})")

                try:
                    choice = int(input("Select user (number): ")) - 1
                    if 0 <= choice < len(users):
                        current_user = users[choice]["user_id"]
                        print(f"✅ Switched to user: {users[choice]['display_name']}")
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Please enter a valid number")

            elif command == "new_project":
                if not current_user:
                    print("Please select a user first")
                    continue

                project_name = input("Enter project name: ").strip()
                if project_name:
                    project_id = rag.create_project(current_user, project_name)
                    print(f"✅ Project created: {project_name}")

            elif command == "switch_project":
                if not current_user:
                    print("Please select a user first")
                    continue

                projects = rag.list_user_projects(current_user)
                if not projects:
                    print("No projects found. Create a project first.")
                    continue

                print("\n📂 Your Projects:")
                for i, project in enumerate(projects, 1):
                    print(f"{i}. {project['project_name']} ({project['current_phase']})")

                try:
                    choice = int(input("Select project (number): ")) - 1
                    if 0 <= choice < len(projects):
                        success = rag.switch_project(current_user, projects[choice]["project_id"])
                        if success:
                            print(f"✅ Switched to project: {projects[choice]['project_name']}")
                        else:
                            print("❌ Failed to switch project")
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Please enter a valid number")

            elif command == "chat":
                if not current_user:
                    print("Please select a user first")
                    continue

                print("\n💬 Chat Mode (type 'exit' to return to main menu)")
                while True:
                    user_input = input(f"\n[You]: ").strip()
                    if user_input.lower() == "exit":
                        break
                    if user_input:
                        print(f"\n[Assistant]: ", end="")
                        response = rag.generate_socratic_response(current_user, user_input)
                        print(response)

            elif command == "summary":
                if not current_user:
                    print("Please select a user first")
                    continue
                print(rag.get_project_summary(current_user))

            elif command == "users":
                users = rag.list_users()
                print(f"\n👥 Users ({len(users)}):")
                for user in users:
                    status = "🟢" if user["user_id"] == current_user else "⚪"
                    print(f"{status} {user['display_name']} - Last active: {user['last_active']}")

            elif command == "projects":
                if not current_user:
                    print("Please select a user first")
                    continue
                projects = rag.list_user_projects(current_user)
                print(f"\n📂 Your Projects ({len(projects)}):")
                for project in projects:
                    print(f"• {project['project_name']} - Phase: {project['current_phase']}")

            elif command == "stats":
                if not current_user:
                    print("Please select a user first")
                    continue
                stats = rag.get_user_statistics(current_user)
                print(f"\n📊 Statistics:")
                for key, value in stats.items():
                    print(f"• {key.replace('_', ' ').title()}: {value}")

            elif command == "export":
                if not current_user:
                    print("Please select a user first")
                    continue
                data = rag.export_project_data(current_user)
                filename = f"export_{current_user}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                print(f"✅ Data exported to {filename}")

            elif command == "reset":
                if not current_user:
                    print("Please select a user first")
                    continue
                confirm = input("Are you sure you want to reset the current project context? (yes/no): ")
                if confirm.lower() == "yes":
                    success = rag.reset_project_context(current_user)
                    if success:
                        print("✅ Project context reset")
                    else:
                        print("❌ Failed to reset context")

            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

