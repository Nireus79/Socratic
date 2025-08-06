#!/usr/bin/env python3
"""
Enhanced Socratic Agentic RAG System with Database Support
Supports multiple users, multiple projects, and structured data for code generation
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
import uuid
import hashlib
import requests
import os
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class User:
    """User information"""
    id: str
    username: str
    email: str
    created_at: str
    role: str = "developer"  # developer, manager, designer, etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        return cls(**data)


@dataclass
class ProjectMetadata:
    """Project metadata and settings"""
    id: str
    name: str
    description: str
    owner_id: str
    created_at: str
    updated_at: str
    status: str = "active"  # active, paused, completed, archived
    visibility: str = "private"  # private, team, public

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectMetadata':
        return cls(**data)


@dataclass
class TechnicalSpecification:
    """Detailed technical specification for code generation"""
    project_id: str
    database_schema: Dict[str, Any]
    api_design: Dict[str, Any]
    file_structure: Dict[str, Any]
    component_architecture: Dict[str, Any]
    implementation_plan: List[Dict[str, Any]]
    test_requirements: List[str]
    deployment_config: Dict[str, Any]
    dependencies: List[str]
    environment_variables: Dict[str, str]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TechnicalSpecification':
        return cls(**data)


@dataclass
class ProjectContext:
    """Enhanced project context with technical details"""
    project_id: str
    goals: List[str]
    requirements: List[str]
    tech_stack: List[str]
    constraints: List[str]
    team_structure: str
    language_preferences: List[str]
    deployment_target: str
    code_style: str
    phase: str
    business_logic: List[str]
    user_stories: List[str]
    non_functional_requirements: List[str]
    integration_requirements: List[str]
    data_entities: List[Dict[str, Any]]
    api_endpoints: List[Dict[str, Any]]
    ui_components: List[Dict[str, Any]]
    security_requirements: List[str]
    performance_requirements: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectContext':
        return cls(**data)


@dataclass
class KnowledgeEntry:
    """Enhanced knowledge base entry"""
    id: str
    content: str
    category: str
    phase: str
    keywords: List[str]
    project_id: Optional[str] = None  # For project-specific knowledge
    user_id: Optional[str] = None  # For user-contributed knowledge
    embedding: Optional[np.ndarray] = None
    created_at: str = ""
    confidence_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)


@dataclass
class ConversationEntry:
    """Individual conversation entry with enhanced metadata"""
    id: str
    project_id: str
    user_id: str
    role: str  # user, assistant, system
    content: str
    agent_name: Optional[str] = None  # Which agent generated this
    analysis_data: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        return cls(**data)


class DatabaseManager:
    """Manages SQLite database operations"""

    def __init__(self, db_path: str = "socratic_rag.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    role TEXT DEFAULT 'developer'
                )
            """)

            # Projects table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    owner_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    visibility TEXT DEFAULT 'private',
                    FOREIGN KEY (owner_id) REFERENCES users (id)
                )
            """)

            # Project members table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_members (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    joined_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects (id),
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(project_id, user_id)
                )
            """)

            # Project context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_context (
                    project_id TEXT PRIMARY KEY,
                    context_data TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)

            # Technical specifications table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technical_specifications (
                    project_id TEXT PRIMARY KEY,
                    specification_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_name TEXT,
                    analysis_data TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # Knowledge base table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    project_id TEXT,
                    user_id TEXT,
                    embedding BLOB,
                    created_at TEXT NOT NULL,
                    confidence_score REAL DEFAULT 1.0,
                    FOREIGN KEY (project_id) REFERENCES projects (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            conn.commit()

    def create_user(self, username: str, email: str, role: str = "developer") -> User:
        """Create a new user"""
        user_id = str(uuid.uuid4())
        user = User(
            id=user_id,
            username=username,
            email=email,
            created_at=datetime.now().isoformat(),
            role=role
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (id, username, email, created_at, role)
                VALUES (?, ?, ?, ?, ?)
            """, (user.id, user.username, user.email, user.created_at, user.role))
            conn.commit()

        return user

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()

            if row:
                return User(
                    id=row[0], username=row[1], email=row[2],
                    created_at=row[3], role=row[4]
                )
        return None

    def create_project(self, name: str, description: str, owner_id: str) -> ProjectMetadata:
        """Create a new project"""
        project_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        project = ProjectMetadata(
            id=project_id,
            name=name,
            description=description,
            owner_id=owner_id,
            created_at=now,
            updated_at=now
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO projects (id, name, description, owner_id, created_at, updated_at, status, visibility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (project.id, project.name, project.description, project.owner_id,
                  project.created_at, project.updated_at, project.status, project.visibility))

            # Add owner as project member
            cursor.execute("""
                INSERT INTO project_members (id, project_id, user_id, role, joined_at)
                VALUES (?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), project_id, owner_id, "owner", now))

            conn.commit()

        return project

    def add_project_member(self, project_id: str, user_id: str, role: str = "member"):
        """Add a user to a project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO project_members (id, project_id, user_id, role, joined_at)
                VALUES (?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), project_id, user_id, role, datetime.now().isoformat()))
            conn.commit()

    def get_user_projects(self, user_id: str) -> List[ProjectMetadata]:
        """Get all projects for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.* FROM projects p
                JOIN project_members pm ON p.id = pm.project_id
                WHERE pm.user_id = ?
                ORDER BY p.updated_at DESC
            """, (user_id,))

            projects = []
            for row in cursor.fetchall():
                projects.append(ProjectMetadata(
                    id=row[0], name=row[1], description=row[2],
                    owner_id=row[3], created_at=row[4], updated_at=row[5],
                    status=row[6], visibility=row[7]
                ))

            return projects

    def save_project_context(self, context: ProjectContext):
        """Save project context to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO project_context (project_id, context_data, updated_at)
                VALUES (?, ?, ?)
            """, (context.project_id, json.dumps(context.to_dict()), datetime.now().isoformat()))
            conn.commit()

    def load_project_context(self, project_id: str) -> Optional[ProjectContext]:
        """Load project context from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT context_data FROM project_context WHERE project_id = ?", (project_id,))
            row = cursor.fetchone()

            if row:
                context_data = json.loads(row[0])
                return ProjectContext.from_dict(context_data)

        return None

    def save_technical_specification(self, spec: TechnicalSpecification):
        """Save technical specification to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO technical_specifications 
                (project_id, specification_data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (spec.project_id, json.dumps(spec.to_dict()),
                  spec.created_at, spec.updated_at))
            conn.commit()

    def load_technical_specification(self, project_id: str) -> Optional[TechnicalSpecification]:
        """Load technical specification from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT specification_data FROM technical_specifications 
                WHERE project_id = ?
            """, (project_id,))
            row = cursor.fetchone()

            if row:
                spec_data = json.loads(row[0])
                return TechnicalSpecification.from_dict(spec_data)

        return None

    def save_conversation_entry(self, entry: ConversationEntry):
        """Save conversation entry to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations 
                (id, project_id, user_id, role, content, agent_name, analysis_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry.id, entry.project_id, entry.user_id, entry.role,
                  entry.content, entry.agent_name,
                  json.dumps(entry.analysis_data) if entry.analysis_data else None,
                  entry.timestamp))
            conn.commit()

    def get_conversation_history(self, project_id: str, limit: int = 50) -> List[ConversationEntry]:
        """Get conversation history for a project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM conversations 
                WHERE project_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (project_id, limit))

            entries = []
            for row in cursor.fetchall():
                analysis_data = json.loads(row[6]) if row[6] else None
                entries.append(ConversationEntry(
                    id=row[0], project_id=row[1], user_id=row[2],
                    role=row[3], content=row[4], agent_name=row[5],
                    analysis_data=analysis_data, timestamp=row[7]
                ))

            return list(reversed(entries))  # Return in chronological order

    def save_knowledge_entry(self, entry: KnowledgeEntry):
        """Save knowledge entry to database"""
        embedding_blob = pickle.dumps(entry.embedding) if entry.embedding is not None else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_base 
                (id, content, category, phase, keywords, project_id, user_id, 
                 embedding, created_at, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry.id, entry.content, entry.category, entry.phase,
                  json.dumps(entry.keywords), entry.project_id, entry.user_id,
                  embedding_blob, entry.created_at, entry.confidence_score))
            conn.commit()

    def load_knowledge_entries(self, project_id: Optional[str] = None) -> List[KnowledgeEntry]:
        """Load knowledge entries, optionally filtered by project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if project_id:
                cursor.execute("""
                    SELECT * FROM knowledge_base 
                    WHERE project_id IS NULL OR project_id = ?
                    ORDER BY confidence_score DESC
                """, (project_id,))
            else:
                cursor.execute("SELECT * FROM knowledge_base ORDER BY confidence_score DESC")

            entries = []
            for row in cursor.fetchall():
                embedding = pickle.loads(row[7]) if row[7] else None
                entries.append(KnowledgeEntry(
                    id=row[0], content=row[1], category=row[2], phase=row[3],
                    keywords=json.loads(row[4]), project_id=row[5], user_id=row[6],
                    embedding=embedding, created_at=row[8], confidence_score=row[9]
                ))

            return entries


class SimpleEmbedding:
    """Simple embedding system for free Claude version compatibility"""

    def __init__(self):
        self.vocab = {}
        self.vocab_size = 1000

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate simple bag-of-words embedding"""
        words = text.lower().split()
        embedding = np.zeros(self.vocab_size)

        for word in words:
            if word not in self.vocab:
                if len(self.vocab) < self.vocab_size:
                    self.vocab[word] = len(self.vocab)
                else:
                    continue

            if word in self.vocab:
                embedding[self.vocab[word]] = 1.0

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, name: str, role: str, personality: str):
        self.name = name
        self.role = role
        self.personality = personality

    def get_context_summary(self, conversation_history: List[ConversationEntry]) -> str:
        """Get summary of recent conversation"""
        if not conversation_history:
            return "No previous conversation."

        recent = conversation_history[-3:]
        summary = "Recent conversation:\n"
        for entry in recent:
            summary += f"{entry.role}: {entry.content}\n"
        return summary


class SocraticAgent(BaseAgent):
    """Enhanced Socratic questioning agent"""

    def __init__(self):
        super().__init__(
            "Socrates",
            "Primary Questioner",
            "Asks probing questions to help users discover solutions themselves"
        )

    def generate_question(self, context: ProjectContext, user_input: str,
                          knowledge_base: List[KnowledgeEntry], conversation_history: List[ConversationEntry]) -> str:
        """Generate Socratic question based on context and input"""
        phase = context.phase.lower()

        if phase == "discovery":
            return self._discovery_questions(user_input, context)
        elif phase == "analysis":
            return self._analysis_questions(user_input, context)
        elif phase == "design":
            return self._design_questions(user_input, context)
        elif phase == "implementation":
            return self._implementation_questions(user_input, context)
        else:
            return self._general_questions(user_input, context)

    def _discovery_questions(self, user_input: str, context: ProjectContext) -> str:
        """Enhanced discovery phase questions"""
        questions = [
            "What specific problem are you trying to solve for your users?",
            "Who is your target audience, and what are their current pain points?",
            "What would success look like for this project?",
            "What assumptions are you making about your users' needs?",
            "How will you measure whether your solution actually works?",
            "What similar solutions already exist, and why aren't they sufficient?",
            "What constraints or limitations should we consider from the start?",
            "Can you describe the user journey through your proposed solution?",
            "What data will your system need to collect and process?",
            "How will users authenticate and authorize with your system?"
        ]

        if not context.goals:
            return questions[0]
        elif not context.user_stories:
            return questions[7]  # User journey question
        elif not context.requirements:
            return questions[1]
        elif len(context.goals) < 2:
            return questions[2]
        elif not context.data_entities:
            return questions[8]  # Data question
        else:
            return questions[3]

    def _analysis_questions(self, user_input: str, context: ProjectContext) -> str:
        """Enhanced analysis phase questions"""
        questions = [
            "What are the core technical challenges in implementing this solution?",
            "Which components will need to communicate with each other?",
            "What data relationships and dependencies exist in your system?",
            "What are the performance requirements and expected load?",
            "What security considerations are critical for your use case?",
            "How will you handle data validation and error scenarios?",
            "What external services or APIs will you need to integrate with?",
            "How will you structure your database to support your requirements?",
            "What caching and optimization strategies should you consider?"
        ]

        if not context.tech_stack:
            return "What technologies or tools are you considering, and why?"
        elif not context.data_entities:
            return questions[7]  # Database structure
        elif not context.integration_requirements:
            return questions[6]  # External integrations
        elif not context.performance_requirements:
            return questions[3]  # Performance
        else:
            return questions[0]

    def _design_questions(self, user_input: str, context: ProjectContext) -> str:
        """Enhanced design phase questions"""
        questions = [
            "How will users interact with your system's interface?",
            "What API endpoints will you need to support your functionality?",
            "How will you organize your code structure and modules?",
            "What testing strategy will ensure reliability?",
            "How will you handle configuration and environment management?",
            "What deployment architecture will you use?",
            "How will you monitor and log system behavior?",
            "What error handling and user feedback mechanisms will you implement?"
        ]

        if not context.ui_components:
            return questions[0]  # UI design
        elif not context.api_endpoints:
            return questions[1]  # API design
        elif not context.non_functional_requirements:
            return questions[3]  # Testing
        else:
            return questions[2]  # Code structure

    def _implementation_questions(self, user_input: str, context: ProjectContext) -> str:
        """Enhanced implementation phase questions"""
        questions = [
            "What's the most critical component to implement first?",
            "How will you set up your development environment and tooling?",
            "What coding standards and practices will your team follow?",
            "How will you manage database migrations and schema changes?",
            "What's your strategy for handling secrets and configuration?",
            "How will you implement automated testing and CI/CD?",
            "What's your plan for performance monitoring and optimization?"
        ]

        return questions[0] if len(context.goals) < 3 else questions[1]

    def _general_questions(self, user_input: str, context: ProjectContext) -> str:
        """General probing questions"""
        return "What specific aspect would you like to explore further?"


class AnalystAgent(BaseAgent):
    """Enhanced analysis and synthesis agent"""

    def __init__(self):
        super().__init__(
            "Theaetetus",
            "Analyst",
            "Analyzes responses and synthesizes insights"
        )

    def analyze_response(self, user_response: str, context: ProjectContext) -> Dict[str, Any]:
        """Enhanced analysis of user response"""
        analysis = {
            "key_points": self._extract_key_points(user_response),
            "technical_insights": self._extract_technical_insights(user_response),
            "business_logic": self._extract_business_logic(user_response),
            "data_entities": self._extract_data_entities(user_response),
            "api_requirements": self._extract_api_requirements(user_response),
            "ui_requirements": self._extract_ui_requirements(user_response),
            "implications": self._identify_implications(user_response, context),
            "missing_info": self._identify_gaps(user_response, context),
            "next_focus": self._suggest_next_focus(user_response, context)
        }
        return analysis

    def _extract_technical_insights(self, response: str) -> List[str]:
        """Extract technical insights from response"""
        insights = []
        tech_indicators = [
            ("database", "Database storage and retrieval needed"),
            ("api", "API endpoints and web services required"),
            ("authentication", "User authentication and security needed"),
            ("real-time", "Real-time functionality or WebSocket connections"),
            ("mobile", "Mobile-responsive or native app considerations"),
            ("scale", "Scalability and performance optimization important"),
            ("integration", "Third-party service integration required")
        ]

        response_lower = response.lower()
        for indicator, insight in tech_indicators:
            if indicator in response_lower:
                insights.append(insight)

        return insights

    def _extract_business_logic(self, response: str) -> List[str]:
        """Extract business logic from response"""
        business_rules = []
        sentences = response.split('.')

        rule_indicators = ["when", "if", "must", "should", "cannot", "only", "always", "never"]
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in rule_indicators):
                if len(sentence) > 10:  # Avoid very short sentences
                    business_rules.append(sentence)

        return business_rules[:5]  # Limit to top 5

    def _extract_data_entities(self, response: str) -> List[Dict[str, Any]]:
        """Extract data entities and their relationships"""
        entities = []
        entity_indicators = ["user", "customer", "product", "order", "task", "project", "file", "message", "comment"]

        for indicator in entity_indicators:
            if indicator in response.lower():
                # Try to extract attributes mentioned nearby
                entity = {
                    "name": indicator,
                    "attributes": self._extract_entity_attributes(response, indicator),
                    "relationships": []
                }
                entities.append(entity)

        return entities

    def _extract_entity_attributes(self, response: str, entity: str) -> List[str]:
        """Extract attributes for a given entity"""
        # Simple pattern matching for attributes
        attributes = []
        common_attributes = {
            "user": ["name", "email", "password", "role", "created_at"],
            "customer": ["name", "email", "phone", "address", "company"],
            "product": ["name", "price", "description", "category", "stock"],
            "order": ["total", "status", "date", "items", "customer_id"],
            "task": ["title", "description", "status", "priority", "due_date"],
            "project": ["name", "description", "status", "owner", "deadline"],
            "file": ["name", "size", "type", "path", "uploaded_at"],
            "message": ["content", "sender", "recipient", "timestamp", "status"],
            "comment": ["content", "author", "post_id", "created_at", "likes"]
        }

        if entity in common_attributes:
            # Check which attributes are mentioned in the response
            response_lower = response.lower()
            for attr in common_attributes[entity]:
                if attr in response_lower or attr.replace("_", " ") in response_lower:
                    attributes.append(attr)

        return attributes

    def _extract_api_requirements(self, response: str) -> List[Dict[str, Any]]:
        """Extract API endpoint requirements"""
        endpoints = []
        response_lower = response.lower()

        # Common CRUD operations
        crud_patterns = [
            ("create", "POST"),
            ("add", "POST"),
            ("get", "GET"),
            ("fetch", "GET"),
            ("retrieve", "GET"),
            ("update", "PUT"),
            ("edit", "PUT"),
            ("modify", "PUT"),
            ("delete", "DELETE"),
            ("remove", "DELETE")
        ]

        entities = ["user", "customer", "product", "order", "task", "project", "file", "message"]

        for entity in entities:
            if entity in response_lower:
                for operation, method in crud_patterns:
                    if operation in response_lower:
                        endpoint = {
                            "path": f"/api/{entity}s",
                            "method": method,
                            "description": f"{operation.title()} {entity}",
                            "auth_required": True if entity in ["user", "customer"] else False
                        }
                        endpoints.append(endpoint)

        return endpoints

    def _extract_ui_requirements(self, response: str) -> List[Dict[str, Any]]:
        """Extract UI component requirements"""
        components = []
        ui_indicators = [
            ("form", "form"),
            ("table", "data-table"),
            ("list", "list"),
            ("dashboard", "dashboard"),
            ("login", "login-form"),
            ("profile", "user-profile"),
            ("search", "search-bar"),
            ("navigation", "navbar"),
            ("sidebar", "sidebar"),
            ("modal", "modal"),
            ("button", "button"),
            ("dropdown", "dropdown")
        ]

        response_lower = response.lower()
        for indicator, component_type in ui_indicators:
            if indicator in response_lower:
                component = {
                    "type": component_type,
                    "description": f"{indicator.title()} component",
                    "priority": "high" if indicator in ["login", "dashboard"] else "medium"
                }
                components.append(component)

        return components

    def _extract_key_points(self, response: str) -> List[str]:
        """Extract key points from user response"""
        key_indicators = [
            "want to", "need to", "must", "should", "will",
            "problem", "solution", "challenge", "goal",
            "user", "customer", "client", "team"
        ]

        sentences = response.split('.')
        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in key_indicators):
                if len(sentence) > 15:  # Avoid very short sentences
                    key_points.append(sentence)

        return key_points[:5]  # Limit to top 5

    def _identify_implications(self, response: str, context: ProjectContext) -> List[str]:
        """Identify implications and consequences"""
        implications = []

        # Technical implications
        if "real-time" in response.lower():
            implications.append("Real-time functionality requires WebSocket connections or similar technology")

        if "mobile" in response.lower():
            implications.append("Mobile support requires responsive design or native app development")

        if "scale" in response.lower() or "many users" in response.lower():
            implications.append("Scalability concerns require careful architecture and possibly caching strategies")

        if "secure" in response.lower() or "privacy" in response.lower():
            implications.append("Security requirements may need encryption, authentication, and authorization")

        if "integrate" in response.lower() or "third-party" in response.lower():
            implications.append("External integrations require API management and error handling")

        return implications

    def _identify_gaps(self, response: str, context: ProjectContext) -> List[str]:
        """Identify missing information or gaps"""
        gaps = []

        if not context.tech_stack:
            gaps.append("Technology stack not defined")

        if not context.user_stories:
            gaps.append("User stories need to be defined")

        if not context.data_entities:
            gaps.append("Data model and entities need clarification")

        if not context.api_endpoints:
            gaps.append("API design and endpoints need specification")

        if not context.security_requirements:
            gaps.append("Security requirements need to be addressed")

        if not context.performance_requirements:
            gaps.append("Performance and scalability requirements need definition")

        return gaps

    def _suggest_next_focus(self, response: str, context: ProjectContext) -> str:
        """Suggest what to focus on next"""
        phase = context.phase.lower()

        if phase == "discovery":
            if not context.user_stories:
                return "Define user stories and user journeys"
            elif not context.data_entities:
                return "Identify data entities and relationships"
            else:
                return "Move to analysis phase - technical requirements"

        elif phase == "analysis":
            if not context.tech_stack:
                return "Choose technology stack"
            elif not context.api_endpoints:
                return "Design API structure and endpoints"
            else:
                return "Move to design phase - detailed architecture"

        elif phase == "design":
            if not context.ui_components:
                return "Design user interface components"
            else:
                return "Move to implementation phase - start coding"

        else:
            return "Continue with current implementation tasks"


class CreatorAgent(BaseAgent):
    """Enhanced content creation and synthesis agent"""

    def __init__(self):
        super().__init__(
            "Demiurge",
            "Creator",
            "Synthesizes insights into actionable content and specifications"
        )

    def create_technical_specification(self, context: ProjectContext,
                                       analysis: Dict[str, Any]) -> TechnicalSpecification:
        """Create comprehensive technical specification"""
        now = datetime.now().isoformat()

        spec = TechnicalSpecification(
            project_id=context.project_id,
            database_schema=self._generate_database_schema(context, analysis),
            api_design=self._generate_api_design(context, analysis),
            file_structure=self._generate_file_structure(context),
            component_architecture=self._generate_component_architecture(context, analysis),
            implementation_plan=self._generate_implementation_plan(context, analysis),
            test_requirements=self._generate_test_requirements(context),
            deployment_config=self._generate_deployment_config(context),
            dependencies=self._generate_dependencies(context),
            environment_variables=self._generate_environment_variables(context),
            created_at=now,
            updated_at=now
        )

        return spec

    def _generate_database_schema(self, context: ProjectContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate database schema from context and analysis"""
        schema = {
            "database_type": "postgresql",  # Default choice
            "tables": {},
            "relationships": [],
            "indexes": []
        }

        # Extract entities from analysis
        entities = analysis.get("data_entities", [])

        for entity in entities:
            table_name = f"{entity['name']}s"
            schema["tables"][table_name] = {
                "columns": {
                    "id": {"type": "uuid", "primary_key": True, "default": "gen_random_uuid()"},
                    "created_at": {"type": "timestamp", "default": "now()"},
                    "updated_at": {"type": "timestamp", "default": "now()"}
                }
            }

            # Add entity-specific attributes
            for attr in entity.get("attributes", []):
                column_type = self._infer_column_type(attr)
                schema["tables"][table_name]["columns"][attr] = {"type": column_type}

        return schema

    def _infer_column_type(self, attribute: str) -> str:
        """Infer database column type from attribute name"""
        type_mapping = {
            "email": "varchar(255)",
            "password": "varchar(255)",
            "name": "varchar(255)",
            "title": "varchar(255)",
            "description": "text",
            "content": "text",
            "price": "decimal(10,2)",
            "total": "decimal(10,2)",
            "status": "varchar(50)",
            "role": "varchar(50)",
            "phone": "varchar(20)",
            "address": "text",
            "date": "date",
            "timestamp": "timestamp",
            "created_at": "timestamp",
            "updated_at": "timestamp",
            "size": "bigint",
            "count": "integer",
            "priority": "integer"
        }

        return type_mapping.get(attribute, "varchar(255)")

    def _generate_api_design(self, context: ProjectContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate API design specification"""
        api_design = {
            "base_url": "/api/v1",
            "authentication": "JWT",
            "endpoints": [],
            "middleware": ["cors", "helmet", "rate-limiting", "logging"],
            "response_format": "JSON",
            "error_handling": "standardized"
        }

        # Add endpoints from analysis
        api_requirements = analysis.get("api_requirements", [])
        for req in api_requirements:
            endpoint = {
                "path": req["path"],
                "method": req["method"],
                "description": req["description"],
                "auth_required": req.get("auth_required", False),
                "parameters": [],
                "response_schema": {}
            }
            api_design["endpoints"].append(endpoint)

        return api_design

    def _generate_file_structure(self, context: ProjectContext) -> Dict[str, Any]:
        """Generate project file structure"""
        tech_stack = context.tech_stack

        if any("react" in tech.lower() for tech in tech_stack):
            return self._react_file_structure()
        elif any("python" in tech.lower() or "flask" in tech.lower() or "django" in tech.lower() for tech in
                 tech_stack):
            return self._python_file_structure()
        elif any("node" in tech.lower() or "express" in tech.lower() for tech in tech_stack):
            return self._node_file_structure()
        else:
            return self._generic_file_structure()

    def _react_file_structure(self) -> Dict[str, Any]:
        """Generate React project structure"""
        return {
            "src/": {
                "components/": {"common/": {}, "forms/": {}, "layout/": {}},
                "pages/": {},
                "hooks/": {},
                "services/": {"api.js": None},
                "utils/": {"helpers.js": None},
                "styles/": {"globals.css": None},
                "context/": {"AuthContext.js": None},
                "App.js": None,
                "index.js": None
            },
            "public/": {"index.html": None},
            "package.json": None,
            ".env": None,
            "README.md": None
        }

    def _python_file_structure(self) -> Dict[str, Any]:
        """Generate Python project structure"""
        return {
            "app/": {
                "models/": {"__init__.py": None},
                "routes/": {"__init__.py": None},
                "services/": {"__init__.py": None},
                "utils/": {"__init__.py": None},
                "templates/": {},
                "static/": {"css/": {}, "js/": {}},
                "__init__.py": None
            },
            "migrations/": {},
            "tests/": {"__init__.py": None},
            "config.py": None,
            "requirements.txt": None,
            "app.py": None,
            ".env": None,
            "README.md": None
        }

    def _node_file_structure(self) -> Dict[str, Any]:
        """Generate Node.js project structure"""
        return {
            "src/": {
                "controllers/": {},
                "models/": {},
                "routes/": {},
                "middleware/": {},
                "services/": {},
                "utils/": {},
                "config/": {"database.js": None}
            },
            "tests/": {},
            "package.json": None,
            "server.js": None,
            ".env": None,
            "README.md": None
        }

    def _generic_file_structure(self) -> Dict[str, Any]:
        """Generate generic project structure"""
        return {
            "src/": {},
            "tests/": {},
            "docs/": {},
            "config/": {},
            "README.md": None,
            ".env": None
        }

    def _generate_component_architecture(self, context: ProjectContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate component architecture"""
        architecture = {
            "pattern": "MVC",  # Default
            "layers": {
                "presentation": {
                    "components": analysis.get("ui_requirements", []),
                    "responsibilities": ["User interface", "User interactions", "Data presentation"]
                },
                "business": {
                    "services": [],
                    "responsibilities": ["Business logic", "Data validation", "Process orchestration"]
                },
                "data": {
                    "repositories": [],
                    "responsibilities": ["Data access", "Database operations", "External API calls"]
                }
            },
            "communication": "dependency_injection"
        }

        # Add services based on entities
        entities = analysis.get("data_entities", [])
        for entity in entities:
            service_name = f"{entity['name'].title()}Service"
            architecture["layers"]["business"]["services"].append(service_name)

            repo_name = f"{entity['name'].title()}Repository"
            architecture["layers"]["data"]["repositories"].append(repo_name)

        return architecture

    def _generate_implementation_plan(self, context: ProjectContext, analysis: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """Generate step-by-step implementation plan"""
        plan = [
            {
                "phase": "Setup",
                "tasks": [
                    "Initialize project structure",
                    "Set up development environment",
                    "Configure version control",
                    "Set up database"
                ],
                "estimated_hours": 8,
                "dependencies": []
            },
            {
                "phase": "Core Backend",
                "tasks": [
                    "Implement database models",
                    "Create basic API endpoints",
                    "Set up authentication",
                    "Implement error handling"
                ],
                "estimated_hours": 24,
                "dependencies": ["Setup"]
            },
            {
                "phase": "Business Logic",
                "tasks": [
                    "Implement core business services",
                    "Add data validation",
                    "Create business rule processing"
                ],
                "estimated_hours": 32,
                "dependencies": ["Core Backend"]
            },
            {
                "phase": "Frontend",
                "tasks": [
                    "Create basic UI components",
                    "Implement user authentication flow",
                    "Build main application views",
                    "Add responsive design"
                ],
                "estimated_hours": 40,
                "dependencies": ["Business Logic"]
            },
            {
                "phase": "Integration & Testing",
                "tasks": [
                    "Write unit tests",
                    "Implement integration tests",
                    "Perform end-to-end testing",
                    "Fix bugs and optimize"
                ],
                "estimated_hours": 24,
                "dependencies": ["Frontend"]
            },
            {
                "phase": "Deployment",
                "tasks": [
                    "Set up production environment",
                    "Configure CI/CD pipeline",
                    "Deploy application",
                    "Monitor and maintain"
                ],
                "estimated_hours": 16,
                "dependencies": ["Integration & Testing"]
            }
        ]

        return plan

    def _generate_test_requirements(self, context: ProjectContext) -> List[str]:
        """Generate testing requirements"""
        return [
            "Unit tests for all business logic functions",
            "Integration tests for API endpoints",
            "Database transaction tests",
            "Authentication and authorization tests",
            "Input validation and error handling tests",
            "Performance tests for critical paths",
            "End-to-end user workflow tests",
            "Security vulnerability tests",
            "Cross-browser compatibility tests (if web app)",
            "Mobile responsiveness tests (if applicable)"
        ]

    def _generate_deployment_config(self, context: ProjectContext) -> Dict[str, Any]:
        """Generate deployment configuration"""
        return {
            "environment": "production",
            "platform": "docker",
            "orchestration": "docker-compose",
            "database": {
                "type": "postgresql",
                "backup_strategy": "daily",
                "connection_pooling": True
            },
            "caching": {
                "type": "redis",
                "ttl": 3600
            },
            "monitoring": {
                "logging": "structured",
                "metrics": "prometheus",
                "alerting": "email"
            },
            "security": {
                "https": True,
                "security_headers": True,
                "rate_limiting": True
            },
            "scaling": {
                "auto_scaling": False,
                "load_balancer": False,
                "cdn": False
            }
        }

    def _generate_dependencies(self, context: ProjectContext) -> List[str]:
        """Generate project dependencies based on tech stack"""
        tech_stack = context.tech_stack
        dependencies = []

        # Python dependencies
        if any("python" in tech.lower() for tech in tech_stack):
            dependencies.extend([
                "flask>=2.0.0",
                "sqlalchemy>=1.4.0",
                "alembic>=1.7.0",
                "psycopg2-binary>=2.9.0",
                "flask-jwt-extended>=4.2.0",
                "marshmallow>=3.14.0",
                "python-dotenv>=0.19.0",
                "pytest>=6.2.0",
                "black>=21.0.0",
                "flake8>=4.0.0"
            ])

        # Node.js dependencies
        if any("node" in tech.lower() or "express" in tech.lower() for tech in tech_stack):
            dependencies.extend([
                "express>=4.18.0",
                "mongoose>=6.0.0",
                "jsonwebtoken>=8.5.0",
                "bcrypt>=5.0.0",
                "cors>=2.8.0",
                "helmet>=5.0.0",
                "dotenv>=16.0.0",
                "jest>=28.0.0",
                "nodemon>=2.0.0",
                "eslint>=8.0.0"
            ])

        # React dependencies
        if any("react" in tech.lower() for tech in tech_stack):
            dependencies.extend([
                "react>=18.0.0",
                "react-dom>=18.0.0",
                "react-router-dom>=6.0.0",
                "axios>=0.27.0",
                "styled-components>=5.3.0",
                "@testing-library/react>=13.0.0",
                "@testing-library/jest-dom>=5.16.0"
            ])

        return dependencies

    def _generate_environment_variables(self, context: ProjectContext) -> Dict[str, str]:
        """Generate environment variables template"""
        return {
            "NODE_ENV": "production",
            "PORT": "3000",
            "DATABASE_URL": "postgresql://user:password@localhost:5432/dbname",
            "JWT_SECRET": "your-super-secret-jwt-key",
            "JWT_EXPIRATION": "24h",
            "REDIS_URL": "redis://localhost:6379",
            "LOG_LEVEL": "info",
            "API_BASE_URL": "http://localhost:3000/api/v1",
            "CORS_ORIGIN": "http://localhost:3000",
            "RATE_LIMIT_WINDOW": "15",
            "RATE_LIMIT_MAX": "100"
        }


class EnhancedSocraticRAG:
    """Enhanced Socratic RAG system with database support"""

    def __init__(self, db_path: str = "socratic_rag.db"):
        self.db_manager = DatabaseManager(db_path)
        self.embedding_system = SimpleEmbedding()

        # Initialize agents
        self.socratic_agent = SocraticAgent()
        self.analyst_agent = AnalystAgent()
        self.creator_agent = CreatorAgent()

        # Initialize global knowledge base
        self._initialize_knowledge_base()

        logger.info("Enhanced Socratic RAG system initialized")

    def _initialize_knowledge_base(self):
        """Initialize the global knowledge base with enhanced entries"""
        knowledge_entries = [
            # Discovery Phase Knowledge
            KnowledgeEntry(
                id=str(uuid.uuid4()),
                content="User stories should follow the format: As a [user type], I want [functionality] so that "
                        "[benefit]. This helps clarify who needs what and why.",
                category="methodology",
                phase="discovery",
                keywords=["user stories", "requirements", "agile"],
                created_at=datetime.now().isoformat()
            ),
            KnowledgeEntry(
                id=str(uuid.uuid4()),
                content="Data entities represent the core objects in your system. Identify what data you need to "
                        "store, how it relates, and what operations you'll perform on it.",
                category="data_modeling",
                phase="discovery",
                keywords=["data model", "entities", "database"],
                created_at=datetime.now().isoformat()
            ),

            # Analysis Phase Knowledge
            KnowledgeEntry(
                id=str(uuid.uuid4()),
                content="RESTful API design principles: Use HTTP methods appropriately (GET for retrieval, "
                        "POST for creation, PUT for updates, DELETE for removal). Use consistent URL patterns and "
                        "status codes.",
                category="api_design",
                phase="analysis",
                keywords=["REST", "API", "HTTP", "web services"],
                created_at=datetime.now().isoformat()
            ),
            KnowledgeEntry(
                id=str(uuid.uuid4()),
                content="Database normalization reduces redundancy. First normal form eliminates repeating "
                        "groups, second normal form eliminates partial dependencies, third normal form eliminates "
                        "transitive dependencies.",
                category="database",
                phase="analysis",
                keywords=["normalization", "database design", "schema"],
                created_at=datetime.now().isoformat()
            ),

            # Design Phase Knowledge
            KnowledgeEntry(
                id=str(uuid.uuid4()),
                content="MVC (Model-View-Controller) pattern separates concerns: Models handle data, Views handle "
                        "presentation, Controllers handle user input and coordinate between Models and Views.",
                category="architecture",
                phase="design",
                keywords=["MVC", "architecture", "separation of concerns"],
                created_at=datetime.now().isoformat()
            ),
            KnowledgeEntry(
                id=str(uuid.uuid4()),
                content="Authentication verifies identity, Authorization determines permissions. Use JWT tokens "
                        "for stateless authentication in web applications.",
                category="security",
                phase="design",
                keywords=["authentication", "authorization", "JWT", "security"],
                created_at=datetime.now().isoformat()
            ),

            # Implementation Phase Knowledge
            KnowledgeEntry(
                id=str(uuid.uuid4()),
                content="Test-driven development (TDD): Write tests first, then implement code to pass tests. "
                        "This ensures better code quality and test coverage.",
                category="testing",
                phase="implementation",
                keywords=["TDD", "testing", "quality"],
                created_at=datetime.now().isoformat()
            ),
            KnowledgeEntry(
                id=str(uuid.uuid4()),
                content="Environment variables keep sensitive configuration out of code. Use .env files for local "
                        "development and secure secret management in production.",
                category="configuration",
                phase="implementation",
                keywords=["environment", "configuration", "secrets"],
                created_at=datetime.now().isoformat()
            )
        ]

        # Save knowledge entries to database
        for entry in knowledge_entries:
            # Generate embedding
            entry.embedding = self.embedding_system.get_embedding(entry.content)
            self.db_manager.save_knowledge_entry(entry)

    def create_user(self, username: str, email: str, role: str = "developer") -> User:
        """Create a new user"""
        return self.db_manager.create_user(username, email, role)

    def create_project(self, name: str, description: str, owner_username: str) -> ProjectMetadata:
        """Create a new project"""
        user = self.db_manager.get_user(owner_username)
        if not user:
            raise ValueError(f"User {owner_username} not found")

        return self.db_manager.create_project(name, description, user.id)

    def start_conversation(self, project_id: str, user_id: str, initial_message: str) -> str:
        """Start a conversation for a project"""
        # Save user message
        user_entry = ConversationEntry(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            role="user",
            content=initial_message,
            timestamp=datetime.now().isoformat()
        )
        self.db_manager.save_conversation_entry(user_entry)

        # Load or create project context
        context = self.db_manager.load_project_context(project_id)
        if not context:
            context = ProjectContext(
                project_id=project_id,
                goals=[],
                requirements=[],
                tech_stack=[],
                constraints=[],
                team_structure="",
                language_preferences=[],
                deployment_target="",
                code_style="",
                phase="discovery",
                business_logic=[],
                user_stories=[],
                non_functional_requirements=[],
                integration_requirements=[],
                data_entities=[],
                api_endpoints=[],
                ui_components=[],
                security_requirements=[],
                performance_requirements=[],
                timestamp=datetime.now().isoformat()
            )

        # Get conversation history
        conversation_history = self.db_manager.get_conversation_history(project_id)

        # Load relevant knowledge
        knowledge_base = self.db_manager.load_knowledge_entries(project_id)
        relevant_knowledge = self._find_relevant_knowledge(initial_message, knowledge_base)

        # Generate Socratic question
        question = self.socratic_agent.generate_question(
            context, initial_message, relevant_knowledge, conversation_history
        )

        # Save assistant response
        assistant_entry = ConversationEntry(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            role="assistant",
            content=question,
            agent_name=self.socratic_agent.name,
            timestamp=datetime.now().isoformat()
        )
        self.db_manager.save_conversation_entry(assistant_entry)

        return question

    def continue_conversation(self, project_id: str, user_id: str, user_response: str) -> str:
        """Continue an existing conversation"""
        # Save user response
        user_entry = ConversationEntry(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            role="user",
            content=user_response,
            timestamp=datetime.now().isoformat()
        )
        self.db_manager.save_conversation_entry(user_entry)

        # Load project context
        context = self.db_manager.load_project_context(project_id)
        if not context:
            return "Error: Project context not found. Please start a new conversation."

        # Analyze user response
        analysis = self.analyst_agent.analyze_response(user_response, context)

        # Update project context based on analysis
        context = self._update_context_from_analysis(context, analysis, user_response)
        self.db_manager.save_project_context(context)

        # Get conversation history
        conversation_history = self.db_manager.get_conversation_history(project_id)

        # Load relevant knowledge
        knowledge_base = self.db_manager.load_knowledge_entries(project_id)
        relevant_knowledge = self._find_relevant_knowledge(user_response, knowledge_base)

        # Generate next question
        question = self.socratic_agent.generate_question(
            context, user_response, relevant_knowledge, conversation_history
        )

        # Save assistant response with analysis
        assistant_entry = ConversationEntry(
            id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            role="assistant",
            content=question,
            agent_name=self.socratic_agent.name,
            analysis_data=analysis,
            timestamp=datetime.now().isoformat()
        )
        self.db_manager.save_conversation_entry(assistant_entry)

        return question

    def load_github_repository(self, repo_url: str, project_id: str,
                               branch: str = "main") -> Dict[str, Any]:
        """Load scripts and files from a GitHub repository"""
        try:
            # Extract owner and repo from URL
            parts = repo_url.rstrip('/').split('/')
            owner, repo = parts[-2], parts[-1]

            # GitHub API endpoint for repository contents
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"

            result = {
                "loaded_files": 0,
                "errors": [],
                "processed_files": []
            }

            # Get repository structure
            response = requests.get(api_url)
            if response.status_code != 200:
                result["errors"].append(f"Failed to access repository: {response.status_code}")
                return result

            files = response.json()
            self._process_github_files(files, owner, repo, branch, project_id, result)

            return result

        except Exception as e:
            return {"loaded_files": 0, "errors": [str(e)], "processed_files": []}

    def _process_github_files(self, files: List[Dict], owner: str, repo: str,
                              branch: str, project_id: str, result: Dict):
        """Recursively process GitHub repository files"""

        # File extensions to process
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.css', '.html', '.jsx', '.tsx'}
        doc_extensions = {'.md', '.txt', '.rst', '.json', '.yaml', '.yml', '.xml'}

        for file_info in files:
            if file_info['type'] == 'file':
                file_path = file_info['path']
                file_ext = Path(file_path).suffix.lower()

                # Skip binary files and large files
                if file_ext in code_extensions or file_ext in doc_extensions:
                    if file_info['size'] < 100000:  # Skip files larger than 100KB
                        try:
                            # Get file content
                            content_url = file_info['download_url']
                            content_response = requests.get(content_url)

                            if content_response.status_code == 200:
                                content = content_response.text

                                # Determine category based on file extension
                                if file_ext in code_extensions:
                                    category = "code"
                                    phase = "implementation"
                                else:
                                    category = "documentation"
                                    phase = "analysis"

                                # Create knowledge entry using existing structure
                                entry = KnowledgeEntry(
                                    id=str(uuid.uuid4()),
                                    content=f"File: {file_path}\n\n{content}",
                                    category=category,
                                    phase=phase,
                                    keywords=[file_path, Path(file_path).stem, file_ext[1:]],
                                    project_id=project_id,
                                    created_at=datetime.now().isoformat(),
                                    confidence_score=0.8
                                )

                                # Use existing embedding and save methods
                                entry.embedding = self.embedding_system.get_embedding(content)
                                self.db_manager.save_knowledge_entry(entry)

                                result["loaded_files"] += 1
                                result["processed_files"].append(file_path)

                        except Exception as e:
                            result["errors"].append(f"Error processing {file_path}: {str(e)}")

            elif file_info['type'] == 'dir':
                # Recursively process directories
                dir_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_info['path']}"
                try:
                    dir_response = requests.get(dir_url)
                    if dir_response.status_code == 200:
                        dir_files = dir_response.json()
                        self._process_github_files(dir_files, owner, repo, branch, project_id, result)
                except Exception as e:
                    result["errors"].append(f"Error processing directory {file_info['path']}: {str(e)}")

    def load_local_files(self, file_paths: List[str], project_id: str) -> Dict[str, Any]:
        """Load local files into the knowledge base"""
        result = {
            "loaded_files": 0,
            "errors": [],
            "processed_files": []
        }

        # Supported file extensions
        supported_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.css', '.html',
            '.jsx', '.tsx', '.md', '.txt', '.rst', '.json', '.yaml', '.yml',
            '.xml', '.sql', '.sh', '.bat', '.ps1'
        }

        for file_path in file_paths:
            try:
                path = Path(file_path)

                if not path.exists():
                    result["errors"].append(f"File not found: {file_path}")
                    continue

                if not path.is_file():
                    result["errors"].append(f"Not a file: {file_path}")
                    continue

                file_ext = path.suffix.lower()
                if file_ext not in supported_extensions:
                    result["errors"].append(f"Unsupported file type: {file_path}")
                    continue

                # Check file size (skip very large files)
                if path.stat().st_size > 1000000:  # 1MB limit
                    result["errors"].append(f"File too large: {file_path}")
                    continue

                # Read file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()

                # Determine category and phase
                code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.css', '.html', '.jsx', '.tsx'}

                if file_ext in code_extensions:
                    category = "code"
                    phase = "implementation"
                elif file_ext in {'.md', '.txt', '.rst'}:
                    category = "documentation"
                    phase = "analysis"
                elif file_ext in {'.json', '.yaml', '.yml', '.xml'}:
                    category = "configuration"
                    phase = "design"
                else:
                    category = "script"
                    phase = "implementation"

                # Extract keywords
                keywords = [path.name, path.stem, file_ext[1:], str(path.parent.name)]

                # Create knowledge entry using existing structure
                entry = KnowledgeEntry(
                    id=str(uuid.uuid4()),
                    content=f"File: {file_path}\nType: {file_ext}\n\n{content}",
                    category=category,
                    phase=phase,
                    keywords=list(set(keywords)),
                    project_id=project_id,
                    created_at=datetime.now().isoformat(),
                    confidence_score=0.9
                )

                # Use existing embedding and save methods
                entry.embedding = self.embedding_system.get_embedding(content)
                self.db_manager.save_knowledge_entry(entry)

                result["loaded_files"] += 1
                result["processed_files"].append(file_path)

            except Exception as e:
                result["errors"].append(f"Error processing {file_path}: {str(e)}")

        return result

    def load_directory_recursively(self, directory_path: str, project_id: str,
                                   max_depth: int = 3) -> Dict[str, Any]:
        """Recursively load all supported files from a directory"""
        result = {
            "loaded_files": 0,
            "errors": [],
            "processed_files": []
        }

        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                result["errors"].append(f"Directory not found: {directory_path}")
                return result

            # Find all files recursively
            file_paths = []

            def scan_directory(path: Path, current_depth: int = 0):
                if current_depth > max_depth:
                    return

                try:
                    for item in path.iterdir():
                        if item.is_file():
                            file_paths.append(str(item))
                        elif item.is_dir() and not item.name.startswith('.'):
                            # Skip hidden directories and common ignore patterns
                            skip_dirs = {'node_modules', '__pycache__', '.git', 'venv', 'env', 'dist', 'build'}
                            if item.name not in skip_dirs:
                                scan_directory(item, current_depth + 1)
                except PermissionError:
                    result["errors"].append(f"Permission denied: {path}")

            scan_directory(directory)

            # Load found files using existing method
            load_result = self.load_local_files(file_paths, project_id)

            # Merge results
            result["loaded_files"] = load_result["loaded_files"]
            result["errors"].extend(load_result["errors"])
            result["processed_files"] = load_result["processed_files"]

        except Exception as e:
            result["errors"].append(f"Error scanning directory: {str(e)}")

        return result

    def generate_technical_specification(self, project_id: str) -> TechnicalSpecification:
        """Generate comprehensive technical specification for a project"""
        context = self.db_manager.load_project_context(project_id)
        if not context:
            raise ValueError("Project context not found")

        # Get conversation history for analysis
        conversation_history = self.db_manager.get_conversation_history(project_id, limit=100)

        # Analyze all conversations to extract comprehensive insights
        combined_analysis = self._analyze_full_conversation(conversation_history, context)

        # Generate technical specification
        spec = self.creator_agent.create_technical_specification(context, combined_analysis)

        # Save specification to database
        self.db_manager.save_technical_specification(spec)

        return spec

    def _find_relevant_knowledge(self, query: str, knowledge_base: List[KnowledgeEntry], top_k: int = 3) -> List[
        KnowledgeEntry]:
        """Find relevant knowledge entries using simple embedding similarity"""
        if not knowledge_base:
            return []

        query_embedding = self.embedding_system.get_embedding(query)
        similarities = []

        for entry in knowledge_base:
            if entry.embedding is not None:
                similarity = cosine_similarity([query_embedding], [entry.embedding])[0][0]
                similarities.append((similarity, entry))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:top_k]]

    def _update_context_from_analysis(self, context: ProjectContext, analysis: Dict[str, Any],
                                      user_response: str) -> ProjectContext:
        """Update project context based on analysis results"""
        # Update goals
        key_points = analysis.get("key_points", [])
        for point in key_points:
            if "goal" in point.lower() or "want" in point.lower():
                if point not in context.goals:
                    context.goals.append(point)

        # Update requirements
        if "requirement" in user_response.lower() or "must" in user_response.lower():
            context.requirements.append(user_response)

        # Update tech stack
        tech_keywords = ["python", "javascript", "react", "node", "flask", "django", "postgresql", "mysql",
                         "mongodb"]
        response_lower = user_response.lower()
        for keyword in tech_keywords:
            if keyword in response_lower and keyword not in context.tech_stack:
                context.tech_stack.append(keyword)

        # Update business logic
        business_logic = analysis.get("business_logic", [])
        context.business_logic.extend(business_logic)

        # Update data entities
        data_entities = analysis.get("data_entities", [])
        context.data_entities.extend(data_entities)

        # Update API endpoints
        api_requirements = analysis.get("api_requirements", [])
        context.api_endpoints.extend(api_requirements)

        # Update UI components
        ui_requirements = analysis.get("ui_requirements", [])
        context.ui_components.extend(ui_requirements)

        # Update timestamp
        context.timestamp = datetime.now().isoformat()

        return context

    def _analyze_full_conversation(self, conversation_history: List[ConversationEntry], context: ProjectContext) -> \
            Dict[str, Any]:
        """Analyze full conversation history to extract comprehensive insights"""
        combined_analysis = {
            "key_points": [],
            "technical_insights": [],
            "business_logic": [],
            "data_entities": [],
            "api_requirements": [],
            "ui_requirements": [],
            "implications": [],
            "missing_info": [],
            "next_focus": ""
        }

        # Analyze all user responses
        for entry in conversation_history:
            if entry.role == "user":
                analysis = self.analyst_agent.analyze_response(entry.content, context)

                # Combine all insights
                for key in combined_analysis:
                    if key in analysis and isinstance(analysis[key], list):
                        combined_analysis[key].extend(analysis[key])
                    elif key in analysis and isinstance(analysis[key], str):
                        combined_analysis[key] = analysis[key]

        # Remove duplicates from lists
        for key, value in combined_analysis.items():
            if isinstance(value, list):
                combined_analysis[key] = list(set(value))

        return combined_analysis

    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive project summary"""
        context = self.db_manager.load_project_context(project_id)
        if not context:
            return {"error": "Project not found"}

        spec = self.db_manager.load_technical_specification(project_id)
        conversation_history = self.db_manager.get_conversation_history(project_id)

        summary = {
            "project_id": project_id,
            "phase": context.phase,
            "goals": context.goals,
            "requirements": context.requirements,
            "tech_stack": context.tech_stack,
            "conversation_count": len(conversation_history),
            "data_entities": len(context.data_entities),
            "api_endpoints": len(context.api_endpoints),
            "ui_components": len(context.ui_components),
            "has_technical_spec": spec is not None,
            "last_updated": context.timestamp
        }

        if spec:
            summary["implementation_plan"] = spec.implementation_plan
            summary["estimated_total_hours"] = sum(
                phase.get("estimated_hours", 0) for phase in spec.implementation_plan
            )

        return summary

    def add_custom_knowledge(self, project_id: str, user_id: str, content: str,
                             category: str, phase: str, keywords: List[str]) -> KnowledgeEntry:
        """Add custom knowledge entry for a project"""
        entry = KnowledgeEntry(
            id=str(uuid.uuid4()),
            content=content,
            category=category,
            phase=phase,
            keywords=keywords,
            project_id=project_id,
            user_id=user_id,
            created_at=datetime.now().isoformat()
        )

        # Generate embedding
        entry.embedding = self.embedding_system.get_embedding(content)

        # Save to database
        self.db_manager.save_knowledge_entry(entry)

        return entry

    def export_project_data(self, project_id: str) -> Dict[str, Any]:
        """Export all project data for backup or transfer"""
        context = self.db_manager.load_project_context(project_id)
        spec = self.db_manager.load_technical_specification(project_id)
        conversation_history = self.db_manager.get_conversation_history(project_id, limit=1000)
        knowledge_entries = self.db_manager.load_knowledge_entries(project_id)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "project_id": project_id,
            "context": context.to_dict() if context else None,
            "technical_specification": spec.to_dict() if spec else None,
            "conversation_history": [entry.to_dict() for entry in conversation_history],
            "custom_knowledge": [entry.to_dict() for entry in knowledge_entries if
                                 entry.project_id == project_id]
        }

        return export_data

    def change_project_phase(self, project_id: str, new_phase: str) -> bool:
        """Change project phase (discovery -> analysis -> design -> implementation)"""
        valid_phases = ["discovery", "analysis", "design", "implementation"]
        if new_phase not in valid_phases:
            return False

        context = self.db_manager.load_project_context(project_id)
        if not context:
            return False

        context.phase = new_phase
        context.timestamp = datetime.now().isoformat()
        self.db_manager.save_project_context(context)

        return True

    def get_conversation_insights(self, project_id: str) -> Dict[str, Any]:
        """Get insights from conversation patterns and analysis"""
        conversation_history = self.db_manager.get_conversation_history(project_id)

        insights = {
            "total_conversations": len(conversation_history),
            "user_messages": len([e for e in conversation_history if e.role == "user"]),
            "assistant_messages": len([e for e in conversation_history if e.role == "assistant"]),
            "agents_used": list(set([e.agent_name for e in conversation_history if e.agent_name])),
            "conversation_timeline": [],
            "key_decisions": [],
            "evolution_summary": ""
        }

        # Build timeline
        for entry in conversation_history:
            insights["conversation_timeline"].append({
                "timestamp": entry.timestamp,
                "role": entry.role,
                "agent": entry.agent_name,
                "content_length": len(entry.content)
            })

        # Extract key decisions from analysis data
        for entry in conversation_history:
            if entry.analysis_data and "key_points" in entry.analysis_data:
                for point in entry.analysis_data["key_points"]:
                    if any(keyword in point.lower() for keyword in
                           ["decide", "choose", "will use", "going with"]):
                        insights["key_decisions"].append({
                            "timestamp": entry.timestamp,
                            "decision": point
                        })

        return insights


def main():
    """Functional main interface for the Socratic RAG system"""
    rag_system = EnhancedSocraticRAG()
    current_user = None
    current_project = None

    def print_menu():
        print("\n=== Socratic RAG System ===")
        print("Ουδέν οίδα, ούτε διδάσκω τι, αλλά διαπορώ μόνον.")
        print("1. Create/Login User")
        print("2. Create New Project")
        print("3. List My Projects")
        print("4. Select Project")
        print("5. Start/Continue Conversation")
        print("6. Change Project Phase")
        print("7. Generate Technical Specification")
        print("8. View Project Summary")
        print("9. View Conversation Insights")
        print("10. Add Custom Knowledge")
        print("11. Export Project Data")
        print("12. Load GitHub Repository")  # NEW
        print("13. Load Local Files")  # NEW
        print("14. Load Directory")  # NEW
        print("0. Exit")
        print(f"Current User: {current_user.username if current_user else 'None'}")
        print(f"Current Project: {current_project.name if current_project else 'None'}")
        print("-" * 40)

    def create_or_login_user():
        nonlocal current_user
        username = input("Enter username: ").strip()
        existing_user = rag_system.db_manager.get_user(username)

        if existing_user:
            current_user = existing_user
            print(f"Logged in as: {username}")
        else:
            email = input("Enter email: ").strip()
            role = input("Enter role (developer/manager/designer) [developer]: ").strip() or "developer"
            try:
                current_user = rag_system.create_user(username, email, role)
                print(f"Created and logged in as: {username}")
            except Exception as e:
                print(f"Error creating user: {e}")

    def create_project():
        if not current_user:
            print("Please login first!")
            return

        name = input("Project name: ").strip()
        description = input("Project description: ").strip()

        try:
            project = rag_system.create_project(name, description, current_user.username)
            print(f"Created project: {project.name}")
            return project
        except Exception as e:
            print(f"Error creating project: {e}")
            return None

    def list_projects():
        if not current_user:
            print("Please login first!")
            return

        projects = rag_system.db_manager.get_user_projects(current_user.id)
        if not projects:
            print("No projects found.")
            return

        print("\nYour Projects:")
        for i, project in enumerate(projects, 1):
            print(f"{i}. {project.name} - {project.status} ({project.updated_at})")
        return projects

    def select_project():
        nonlocal current_project
        projects = list_projects()
        if not projects:
            return

        try:
            choice = int(input("Select project number: ")) - 1
            if 0 <= choice < len(projects):
                current_project = projects[choice]
                print(f"Selected project: {current_project.name}")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")

    def conversation():
        if not current_user or not current_project:
            print("Please login and select a project first!")
            return

        # Check if this is the first conversation
        history = rag_system.db_manager.get_conversation_history(current_project.id, limit=1)

        if not history:
            print("Starting new conversation for this project.")
            message = input("Describe your project idea: ").strip()
            if message:
                response = rag_system.start_conversation(current_project.id, current_user.id, message)
                print(f"\nAssistant: {response}")
        else:
            print("Continuing existing conversation.")
            message = input("Your response: ").strip()
            if message:
                response = rag_system.continue_conversation(current_project.id, current_user.id, message)
                print(f"\nAssistant: {response}")

    def change_phase():
        if not current_project:
            print("Please select a project first!")
            return

        phases = ["discovery", "analysis", "design", "implementation"]
        print("Available phases:")
        for i, phase in enumerate(phases, 1):
            print(f"{i}. {phase}")

        try:
            choice = int(input("Select phase number: ")) - 1
            if 0 <= choice < len(phases):
                success = rag_system.change_project_phase(current_project.id, phases[choice])
                if success:
                    print(f"Changed phase to: {phases[choice]}")
                else:
                    print("Failed to change phase.")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")

    def generate_spec():
        if not current_project:
            print("Please select a project first!")
            return

        try:
            print("Generating technical specification...")
            spec = rag_system.generate_technical_specification(current_project.id)
            print("Technical specification generated successfully!")

            print(f"\nSummary:")
            print(f"- Database tables: {len(spec.database_schema.get('tables', {}))}")
            print(f"- API endpoints: {len(spec.api_design.get('endpoints', []))}")
            print(f"- Implementation phases: {len(spec.implementation_plan)}")
            print(f"- Dependencies: {len(spec.dependencies)}")

            # Show estimated timeline
            total_hours = sum(phase.get("estimated_hours", 0) for phase in spec.implementation_plan)
            print(f"- Estimated total hours: {total_hours}")
            print(f"- Estimated weeks (40h/week): {total_hours / 40:.1f}")

        except Exception as e:
            print(f"Error generating specification: {e}")

    def view_summary():
        if not current_project:
            print("Please select a project first!")
            return

        summary = rag_system.get_project_summary(current_project.id)
        print("\nProject Summary:")
        for key, value in summary.items():
            if key != "implementation_plan":  # Skip detailed plan in summary
                print(f"- {key.replace('_', ' ').title()}: {value}")

    def view_insights():
        if not current_project:
            print("Please select a project first!")
            return

        insights = rag_system.get_conversation_insights(current_project.id)
        print("\nConversation Insights:")
        print(f"- Total conversations: {insights['total_conversations']}")
        print(f"- User messages: {insights['user_messages']}")
        print(f"- Assistant messages: {insights['assistant_messages']}")
        print(f"- Agents used: {', '.join(insights['agents_used'])}")
        print(f"- Key decisions made: {len(insights['key_decisions'])}")

        if insights['key_decisions']:
            print("\nKey Decisions:")
            for decision in insights['key_decisions'][-3:]:  # Show last 3
                print(f"  - {decision['decision']}")

    def add_knowledge():
        if not current_user or not current_project:
            print("Please login and select a project first!")
            return

        content = input("Knowledge content: ").strip()
        category = input("Category: ").strip()
        phase = input("Phase (discovery/analysis/design/implementation): ").strip()
        keywords = input("Keywords (comma-separated): ").strip().split(',')
        keywords = [k.strip() for k in keywords if k.strip()]

        try:
            entry = rag_system.add_custom_knowledge(
                current_project.id, current_user.id, content, category, phase, keywords
            )
            print(f"Added knowledge entry: {entry.id}")
        except Exception as e:
            print(f"Error adding knowledge: {e}")

    def export_data():
        if not current_project:
            print("Please select a project first!")
            return

        try:
            data = rag_system.export_project_data(current_project.id)
            filename = f"project_export_{current_project.id[:8]}.json"

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            print(f"Project data exported to: {filename}")
        except Exception as e:
            print(f"Error exporting data: {e}")

    def load_github_repo():
        if not current_user or not current_project:
            print("Please login and select a project first!")
            return

        repo_url = input("Enter GitHub repository URL: ").strip()
        if repo_url:
            print("Loading repository... (this may take a moment)")
            result = rag_system.load_github_repository(repo_url, current_project.id)

            print(f"✅ Loaded {result['loaded_files']} files")
            print(f"📁 Processed files: {len(result['processed_files'])}")

            if result['errors']:
                print("❌ Errors:")
                for error in result['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error}")

            if result['processed_files']:
                print("📂 Sample processed files:")
                for file_path in result['processed_files'][:5]:
                    print(f"  - {file_path}")

    def load_local_files():
        if not current_user or not current_project:
            print("Please login and select a project first!")
            return

        print("Enter file paths (one per line, empty line to finish):")
        file_paths = []
        while True:
            path = input().strip()
            if not path:
                break
            file_paths.append(path)

        if file_paths:
            print("Loading files...")
            result = rag_system.load_local_files(file_paths, current_project.id)

            print(f"✅ Loaded {result['loaded_files']} files")

            if result['errors']:
                print("❌ Errors:")
                for error in result['errors']:
                    print(f"  - {error}")

    def load_directory():
        if not current_user or not current_project:
            print("Please login and select a project first!")
            return

        dir_path = input("Enter directory path: ").strip()
        if dir_path:
            print("Scanning directory... (this may take a moment)")
            result = rag_system.load_directory_recursively(dir_path, current_project.id)

            print(f"✅ Loaded {result['loaded_files']} files from directory")
            print(f"📁 Processed {len(result['processed_files'])} files")

            if result['errors']:
                print("❌ Errors:")
                for error in result['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error}")

    # Main loop
    while True:
        try:
            print_menu()
            choice = input("Choose option: ").strip()

            if choice == "1":
                create_or_login_user()
            elif choice == "2":
                project = create_project()
                if project:
                    current_project = project
            elif choice == "3":
                list_projects()
            elif choice == "4":
                select_project()
            elif choice == "5":
                conversation()
            elif choice == "6":
                change_phase()
            elif choice == "7":
                generate_spec()
            elif choice == "8":
                view_summary()
            elif choice == "9":
                view_insights()
            elif choice == "10":
                add_knowledge()
            elif choice == "11":
                export_data()
            elif choice == '12':
                load_github_repo()
            elif choice == '13':
                load_local_files()
            elif choice == '14':
                load_directory()
            elif choice == "0":
                print("Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            print("..τω Ασκληπιώ οφείλομεν αλετρυόνα, απόδοτε και μη αμελήσετε..")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
