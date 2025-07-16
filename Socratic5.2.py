#!/usr/bin/env python3
"""
Enhanced Multi-Agent Socratic RAG System
A sophisticated project development system with persistent storage, multi-user, and multi-project support
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import anthropic
from datetime import datetime
import logging
import sqlite3
from pathlib import Path
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectPhase(Enum):
    DISCOVERY = "discovery"
    PLANNING = "planning"
    GENERATION = "generation"
    VALIDATION = "validation"
    COMPLETE = "complete"


@dataclass
class ProjectContext:
    """Comprehensive project context managed by agents"""
    # Core Requirements
    goals_requirements: List[str] = field(default_factory=list)
    functional_requirements: List[str] = field(default_factory=list)
    non_functional_requirements: List[str] = field(default_factory=list)

    # Technical Specifications
    technical_stack: List[str] = field(default_factory=list)
    architecture_pattern: str = ""
    database_requirements: List[str] = field(default_factory=list)
    api_specifications: List[str] = field(default_factory=list)

    # UI/UX Requirements
    ui_components: List[str] = field(default_factory=list)
    user_personas: List[str] = field(default_factory=list)
    user_flows: List[str] = field(default_factory=list)

    # Infrastructure
    deployment_target: str = ""
    scalability_requirements: List[str] = field(default_factory=list)
    security_requirements: List[str] = field(default_factory=list)

    # Project Management
    team_structure: str = ""
    timeline: str = ""
    budget_constraints: List[str] = field(default_factory=list)

    # Quality Metrics
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    completeness_score: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResponse:
    """Structured response from an agent"""
    agent_id: str
    content: str
    context_updates: Dict[str, Any]
    next_questions: List[str]
    confidence: float
    requires_followup: bool = False


@dataclass
class ProjectSpecification:
    """Detailed project specification for code generation"""
    project_name: str
    description: str
    technical_architecture: Dict[str, Any]
    database_schema: Dict[str, Any]
    api_endpoints: List[Dict[str, Any]]
    ui_components: List[Dict[str, Any]]
    deployment_config: Dict[str, Any]
    testing_strategy: Dict[str, Any]


@dataclass
class User:
    """User data structure"""
    user_id: str
    username: str
    email: str
    created_at: str
    last_login: str
    is_active: bool = True


@dataclass
class Project:
    """Project data structure"""
    project_id: str
    name: str
    description: str
    owner_id: str
    created_at: str
    updated_at: str
    current_phase: str
    context: ProjectContext
    collaborators: List[str] = field(default_factory=list)
    is_active: bool = True


class DatabaseManager:
    """Manages SQLite database for persistence"""

    def __init__(self, db_path: str = "socratic_rag.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                current_phase TEXT NOT NULL,
                context_json TEXT NOT NULL,
                collaborators_json TEXT DEFAULT '[]',
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (owner_id) REFERENCES users (user_id)
            )
        """)

        # Conversation history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_input TEXT NOT NULL,
                system_response TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (project_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)

        # Generated code table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_code (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        """)

        conn.commit()
        conn.close()

    def create_user(self, username: str, email: str) -> User:
        """Create a new user"""
        user_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO users (user_id, username, email, created_at, last_login)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, username, email, now, now))
            conn.commit()

            return User(user_id, username, email, now, now)
        except sqlite3.IntegrityError:
            raise ValueError(f"User with username '{username}' or email '{email}' already exists")
        finally:
            conn.close()

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return User(*row)
        return None

    def update_user_login(self, user_id: str):
        """Update user's last login time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE users SET last_login = ? WHERE user_id = ?
        """, (datetime.now().isoformat(), user_id))

        conn.commit()
        conn.close()

    def create_project(self, name: str, description: str, owner_id: str) -> Project:
        """Create a new project"""
        project_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        context = ProjectContext()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO projects (project_id, name, description, owner_id, created_at, 
                                updated_at, current_phase, context_json, collaborators_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (project_id, name, description, owner_id, now, now,
              ProjectPhase.DISCOVERY.value, json.dumps(asdict(context)), json.dumps([])))

        conn.commit()
        conn.close()

        return Project(project_id, name, description, owner_id, now, now,
                       ProjectPhase.DISCOVERY.value, context, [])

    def get_user_projects(self, user_id: str) -> List[Project]:
        """Get all projects for a user (owned or collaborated)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM projects 
            WHERE owner_id = ? OR collaborators_json LIKE ?
            ORDER BY updated_at DESC
        """, (user_id, f'%"{user_id}"%'))

        rows = cursor.fetchall()
        conn.close()

        projects = []
        for row in rows:
            context_dict = json.loads(row[7])
            context = ProjectContext(**context_dict)
            collaborators = json.loads(row[8])

            projects.append(Project(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], context, collaborators, row[9]
            ))

        return projects

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            context_dict = json.loads(row[7])
            context = ProjectContext(**context_dict)
            collaborators = json.loads(row[8])

            return Project(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], context, collaborators, row[9]
            )
        return None

    def update_project(self, project: Project):
        """Update project in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE projects 
            SET name = ?, description = ?, updated_at = ?, current_phase = ?, 
                context_json = ?, collaborators_json = ?
            WHERE project_id = ?
        """, (project.name, project.description, datetime.now().isoformat(),
              project.current_phase, json.dumps(asdict(project.context)),
              json.dumps(project.collaborators), project.project_id))

        conn.commit()
        conn.close()

    def add_collaborator(self, project_id: str, user_id: str):
        """Add collaborator to project"""
        project = self.get_project(project_id)
        if project and user_id not in project.collaborators:
            project.collaborators.append(user_id)
            self.update_project(project)

    def remove_collaborator(self, project_id: str, user_id: str):
        """Remove collaborator from project"""
        project = self.get_project(project_id)
        if project and user_id in project.collaborators:
            project.collaborators.remove(user_id)
            self.update_project(project)

    def save_conversation(self, project_id: str, user_id: str, user_input: str, system_response: str):
        """Save conversation to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO conversation_history (project_id, user_id, timestamp, user_input, system_response)
            VALUES (?, ?, ?, ?, ?)
        """, (project_id, user_id, datetime.now().isoformat(), user_input, system_response))

        conn.commit()
        conn.close()

    def get_conversation_history(self, project_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history for a project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT h.*, u.username 
            FROM conversation_history h
            JOIN users u ON h.user_id = u.user_id
            WHERE h.project_id = ?
            ORDER BY h.timestamp DESC
            LIMIT ?
        """, (project_id, limit))

        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'project_id': row[1],
                'user_id': row[2],
                'timestamp': row[3],
                'user_input': row[4],
                'system_response': row[5],
                'username': row[6]
            })

        return history

    def save_generated_code(self, project_id: str, files: Dict[str, str]):
        """Save generated code files"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get next version number
        cursor.execute("""
            SELECT MAX(version) FROM generated_code WHERE project_id = ?
        """, (project_id,))
        result = cursor.fetchone()
        version = (result[0] or 0) + 1

        # Save all files
        for filename, content in files.items():
            cursor.execute("""
                INSERT INTO generated_code (project_id, filename, content, generated_at, version)
                VALUES (?, ?, ?, ?, ?)
            """, (project_id, filename, content, datetime.now().isoformat(), version))

        conn.commit()
        conn.close()

    def get_generated_code(self, project_id: str, version: Optional[int] = None) -> Dict[str, str]:
        """Get generated code files"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if version is None:
            cursor.execute("""
                SELECT filename, content FROM generated_code
                WHERE project_id = ? AND version = (
                    SELECT MAX(version) FROM generated_code WHERE project_id = ?
                )
            """, (project_id, project_id))
        else:
            cursor.execute("""
                SELECT filename, content FROM generated_code
                WHERE project_id = ? AND version = ?
            """, (project_id, version))

        rows = cursor.fetchall()
        conn.close()

        return {row[0]: row[1] for row in rows}


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, agent_id: str, client: anthropic.Anthropic):
        self.agent_id = agent_id
        self.client = client
        self.expertise_areas = []
        self.conversation_history = []

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        """Process user input and return structured response"""
        raise NotImplementedError

    def _generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Claude"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating response for {self.agent_id}: {e}")
            return "I apologize, but I encountered an error processing your request."


class RequirementsAgent(BaseAgent):
    """Agent focused on functional and non-functional requirements"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("requirements_agent", client)
        self.expertise_areas = ["functional_requirements", "user_stories", "acceptance_criteria"]

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        prompt = f"""
        You are a Business Analysis Agent specializing in requirements gathering.

        User input: "{user_input}"

        Current requirements context:
        - Goals: {', '.join(context.goals_requirements)}
        - Functional: {', '.join(context.functional_requirements)}
        - Non-functional: {', '.join(context.non_functional_requirements)}

        Tasks:
        1. Extract any functional requirements from the user input
        2. Identify non-functional requirements (performance, security, usability)
        3. Suggest 1-2 specific follow-up questions to clarify requirements
        4. Rate your confidence in the current requirements (0-1)

        Return a JSON response with:
        {{
            "functional_requirements": ["new requirement 1", "new requirement 2"],
            "non_functional_requirements": ["new nfr 1"],
            "follow_up_questions": ["specific question 1", "specific question 2"],
            "confidence": 0.8,
            "analysis": "Brief analysis of what was extracted"
        }}
        """

        response_text = self._generate_response(prompt)

        try:
            # Parse JSON response (in real implementation, add better error handling)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                response_data = {"functional_requirements": [], "non_functional_requirements": [],
                                 "follow_up_questions": [], "confidence": 0.5, "analysis": response_text}
        except:
            response_data = {"functional_requirements": [], "non_functional_requirements": [],
                             "follow_up_questions": [], "confidence": 0.5, "analysis": response_text}

        context_updates = {
            "functional_requirements": response_data.get("functional_requirements", []),
            "non_functional_requirements": response_data.get("non_functional_requirements", [])
        }

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", ""),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5)
        )


class TechnicalAgent(BaseAgent):
    """Agent focused on technical architecture and implementation"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("technical_agent", client)
        self.expertise_areas = ["technical_stack", "architecture", "database", "apis"]

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        prompt = f"""
        You are a Technical Architecture Agent specializing in technology decisions.

        User input: "{user_input}"

        Current technical context:
        - Tech Stack: {', '.join(context.technical_stack)}
        - Architecture: {context.architecture_pattern}
        - Database: {', '.join(context.database_requirements)}
        - APIs: {', '.join(context.api_specifications)}

        Tasks:
        1. Identify technology stack preferences and requirements
        2. Determine architecture patterns that would fit
        3. Suggest database and API requirements
        4. Ask technical clarification questions
        5. Rate confidence in technical decisions (0-1)

        Return JSON response with:
        {{
            "technical_stack": ["technology1", "technology2"],
            "architecture_pattern": "pattern name",
            "database_requirements": ["requirement1"],
            "api_specifications": ["api spec1"],
            "follow_up_questions": ["tech question 1"],
            "confidence": 0.8,
            "analysis": "Technical analysis"
        }}
        """

        response_text = self._generate_response(prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                response_data = {"technical_stack": [], "architecture_pattern": "",
                                 "database_requirements": [], "api_specifications": [],
                                 "follow_up_questions": [], "confidence": 0.5, "analysis": response_text}
        except:
            response_data = {"technical_stack": [], "architecture_pattern": "",
                             "database_requirements": [], "api_specifications": [],
                             "follow_up_questions": [], "confidence": 0.5, "analysis": response_text}

        context_updates = {
            "technical_stack": response_data.get("technical_stack", []),
            "architecture_pattern": response_data.get("architecture_pattern", ""),
            "database_requirements": response_data.get("database_requirements", []),
            "api_specifications": response_data.get("api_specifications", [])
        }

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", ""),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5)
        )


class UXAgent(BaseAgent):
    """Agent focused on user experience and interface design"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("ux_agent", client)
        self.expertise_areas = ["ui_components", "user_flows", "user_personas"]

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        prompt = f"""
        You are a UX/UI Agent specializing in user experience design.

        User input: "{user_input}"

        Current UX context:
        - UI Components: {', '.join(context.ui_components)}
        - User Personas: {', '.join(context.user_personas)}
        - User Flows: {', '.join(context.user_flows)}

        Tasks:
        1. Identify UI components and interface requirements
        2. Determine user personas and their needs
        3. Map out user flows and interactions
        4. Ask UX-focused questions
        5. Rate confidence in UX understanding (0-1)

        Return JSON response with:
        {{
            "ui_components": ["component1", "component2"],
            "user_personas": ["persona1"],
            "user_flows": ["flow1"],
            "follow_up_questions": ["ux question 1"],
            "confidence": 0.8,
            "analysis": "UX analysis"
        }}
        """

        response_text = self._generate_response(prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                response_data = {"ui_components": [], "user_personas": [], "user_flows": [],
                                 "follow_up_questions": [], "confidence": 0.5, "analysis": response_text}
        except:
            response_data = {"ui_components": [], "user_personas": [], "user_flows": [],
                             "follow_up_questions": [], "confidence": 0.5, "analysis": response_text}

        context_updates = {
            "ui_components": response_data.get("ui_components", []),
            "user_personas": response_data.get("user_personas", []),
            "user_flows": response_data.get("user_flows", [])
        }

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", ""),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5)
        )


class InfrastructureAgent(BaseAgent):
    """Agent focused on deployment and infrastructure"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("infrastructure_agent", client)
        self.expertise_areas = ["deployment", "scalability", "security"]

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        prompt = f"""
        You are an Infrastructure Agent specializing in deployment and scalability.

        User input: "{user_input}"

        Current infrastructure context:
        - Deployment: {context.deployment_target}
        - Scalability: {', '.join(context.scalability_requirements)}
        - Security: {', '.join(context.security_requirements)}

        Tasks:
        1. Identify deployment requirements and preferences
        2. Determine scalability needs
        3. Assess security requirements
        4. Ask infrastructure-focused questions
        5. Rate confidence in infrastructure understanding (0-1)

        Return JSON response with:
        {{
            "deployment_target": "deployment preference",
            "scalability_requirements": ["scalability req1"],
            "security_requirements": ["security req1"],
            "follow_up_questions": ["infra question 1"],
            "confidence": 0.8,
            "analysis": "Infrastructure analysis"
        }}
        """

        response_text = self._generate_response(prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                response_data = {"deployment_target": "", "scalability_requirements": [],
                                 "security_requirements": [], "follow_up_questions": [],
                                 "confidence": 0.5, "analysis": response_text}
        except:
            response_data = {"deployment_target": "", "scalability_requirements": [],
                             "security_requirements": [], "follow_up_questions": [],
                             "confidence": 0.5, "analysis": response_text}

        context_updates = {
            "deployment_target": response_data.get("deployment_target", ""),
            "scalability_requirements": response_data.get("scalability_requirements", []),
            "security_requirements": response_data.get("security_requirements", [])
        }

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", ""),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5)
        )


class PlanningAgent(BaseAgent):
    """Agent that creates detailed project specifications"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("planning_agent", client)
        self.expertise_areas = ["project_planning", "specifications", "architecture_design"]

    async def create_project_specification(self, context: ProjectContext) -> ProjectSpecification:
        """Create detailed project specification from context"""
        prompt = f"""
        Create a comprehensive project specification based on this context:

        Requirements:
        - Goals: {', '.join(context.goals_requirements)}
        - Functional: {', '.join(context.functional_requirements)}
        - Non-functional: {', '.join(context.non_functional_requirements)}

        Technical:
        - Stack: {', '.join(context.technical_stack)}
        - Architecture: {context.architecture_pattern}
        - Database: {', '.join(context.database_requirements)}
        - APIs: {', '.join(context.api_specifications)}

        UX:
        - UI Components: {', '.join(context.ui_components)}
        - User Flows: {', '.join(context.user_flows)}

        Infrastructure:
        - Deployment: {context.deployment_target}
        - Scalability: {', '.join(context.scalability_requirements)}
        - Security: {', '.join(context.security_requirements)}

        Create a detailed specification including:
        1. Project name and description
        2. Technical architecture details
        3. Database schema design
        4. API endpoint specifications
        5. UI component specifications
        6. Deployment configuration
        7. Testing strategy

        Return as detailed JSON specification.
        """

        response_text = self._generate_response(prompt, max_tokens=3000)

        # Create project specification (simplified for demo)
        project_name = "Generated Project"
        description = "Project generated from user requirements"

        return ProjectSpecification(
            project_name=project_name,
            description=description,
            technical_architecture={
                "stack": context.technical_stack,
                "pattern": context.architecture_pattern,
                "specification": response_text
            },
            database_schema={"requirements": context.database_requirements},
            api_endpoints=[{"specs": context.api_specifications}],
            ui_components=[{"components": context.ui_components}],
            deployment_config={"target": context.deployment_target},
            testing_strategy={"approach": "comprehensive testing strategy"}
        )


class CodeGenerationAgent(BaseAgent):
    """Agent that generates actual code from specifications"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("code_generation_agent", client)
        self.expertise_areas = ["code_generation", "software_engineering"]

    async def generate_code(self, specification: ProjectSpecification) -> Dict[str, str]:
        """Generate complete codebase from specification"""
        generated_files = {}

        # Generate main application file
        main_code = await self._generate_main_application(specification)
        generated_files["main.py"] = main_code

        # Generate additional files based on specification
        if specification.database_schema:
            models_code = await self._generate_models(specification)
            generated_files["models.py"] = models_code

        if specification.api_endpoints:
            routes_code = await self._generate_routes(specification)
            generated_files["routes.py"] = routes_code

        # Generate configuration
        config_code = await self._generate_config(specification)
        generated_files["config.py"] = config_code

        # Generate requirements
        requirements = await self._generate_requirements(specification)
        generated_files["requirements.txt"] = requirements

        # Generate README
        readme = await self._generate_readme(specification)
        generated_files["README.md"] = readme

        return generated_files

    async def _generate_main_application(self, spec: ProjectSpecification) -> str:
        """Generate the main application file"""
        prompt = f"""
        Generate a complete main application file for this project:

        Project: {spec.project_name}
        Description: {spec.description}
        Tech Stack: {', '.join(spec.technical_architecture.get('stack', []))}
        Architecture: {spec.technical_architecture.get('pattern', '')}

        Requirements:
        - Production-ready code with error handling
        - Proper logging and security
        - Clean, documented code
        - Include all necessary imports
        - Follow best practices

        Generate complete, functional Python code.
        """

        return self._generate_response(prompt, max_tokens=3000)

    async def _generate_models(self, spec: ProjectSpecification) -> str:
        """Generate database models"""
        prompt = f"""
        Generate database models for this project:

        Project: {spec.project_name}
        Database Requirements: {spec.database_schema.get('requirements', [])}
        Tech Stack: {', '.join(spec.technical_architecture.get('stack', []))}

        Generate complete SQLAlchemy models with:
        - Proper relationships
        - Validation
        - Indexes where appropriate
        - Clean code with docstrings
        """

        return self._generate_response(prompt, max_tokens=2000)

    async def _generate_routes(self, spec: ProjectSpecification) -> str:
        """Generate API routes"""
        prompt = f"""
        Generate API routes for this project:

        Project: {spec.project_name}
        API Endpoints: {spec.api_endpoints}
        Tech Stack: {', '.join(spec.technical_architecture.get('stack', []))}

        Generate RESTful API routes with:
        - Proper HTTP methods
        - Input validation
        - Error handling
        - Documentation
        - Security considerations

        Generate complete, functional API routes.
        """

        return self._generate_response(prompt, max_tokens=2000)

    async def _generate_config(self, spec: ProjectSpecification) -> str:
        """Generate configuration file"""
        prompt = f"""
        Generate configuration file for this project:

        Project: {spec.project_name}
        Deployment: {spec.deployment_config.get('target', '')}
        Tech Stack: {', '.join(spec.technical_architecture.get('stack', []))}

        Include:
        - Environment variables
        - Database configuration
        - Security settings
        - Logging configuration
        - Deployment settings

        Generate complete configuration code.
        """

        return self._generate_response(prompt, max_tokens=1500)

    async def _generate_requirements(self, spec: ProjectSpecification) -> str:
        """Generate requirements.txt file"""
        prompt = f"""
        Generate requirements.txt for this project:

        Tech Stack: {', '.join(spec.technical_architecture.get('stack', []))}
        Database Requirements: {spec.database_schema.get('requirements', [])}

        Include all necessary Python packages with appropriate versions.
        Consider security and compatibility.

        Generate complete requirements.txt content.
        """

        return self._generate_response(prompt, max_tokens=800)

    async def _generate_readme(self, spec: ProjectSpecification) -> str:
        """Generate README.md file"""
        prompt = f"""
        Generate comprehensive README.md for this project:

        Project: {spec.project_name}
        Description: {spec.description}
        Tech Stack: {', '.join(spec.technical_architecture.get('stack', []))}
        Deployment: {spec.deployment_config.get('target', '')}

        Include:
        - Project description
        - Installation instructions
        - Usage examples
        - API documentation
        - Contributing guidelines
        - License information

        Generate complete, professional README.md.
        """

        return self._generate_response(prompt, max_tokens=2000)


class ValidationAgent(BaseAgent):
    """Agent that validates and reviews generated code"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("validation_agent", client)
        self.expertise_areas = ["code_review", "testing", "quality_assurance"]

    async def validate_code(self, generated_files: Dict[str, str], specification: ProjectSpecification) -> Dict[str, Any]:
        """Validate generated code against specification"""
        validation_results = {
            "overall_score": 0.0,
            "file_scores": {},
            "issues": [],
            "recommendations": [],
            "test_suggestions": []
        }

        total_score = 0
        file_count = 0

        for filename, content in generated_files.items():
            file_validation = await self._validate_file(filename, content, specification)
            validation_results["file_scores"][filename] = file_validation["score"]
            validation_results["issues"].extend(file_validation["issues"])
            validation_results["recommendations"].extend(file_validation["recommendations"])

            total_score += file_validation["score"]
            file_count += 1

        validation_results["overall_score"] = total_score / file_count if file_count > 0 else 0

        # Generate test suggestions
        test_suggestions = await self._generate_test_suggestions(specification)
        validation_results["test_suggestions"] = test_suggestions

        return validation_results

    async def _validate_file(self, filename: str, content: str, specification: ProjectSpecification) -> Dict[str, Any]:
        """Validate individual file"""
        prompt = f"""
        Review this generated code file:

        Filename: {filename}
        Project: {specification.project_name}
        Content: {content[:2000]}...

        Evaluate:
        1. Code quality and best practices
        2. Security considerations
        3. Performance implications
        4. Maintainability
        5. Alignment with specification

        Return JSON with:
        {{
            "score": 0.85,
            "issues": ["issue1", "issue2"],
            "recommendations": ["rec1", "rec2"]
        }}
        """

        response_text = self._generate_response(prompt, max_tokens=1000)

        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {
            "score": 0.7,
            "issues": ["Could not parse validation results"],
            "recommendations": ["Manual review recommended"]
        }

    async def _generate_test_suggestions(self, specification: ProjectSpecification) -> List[str]:
        """Generate test suggestions"""
        prompt = f"""
        Generate test suggestions for this project:

        Project: {specification.project_name}
        Description: {specification.description}
        Tech Stack: {', '.join(specification.technical_architecture.get('stack', []))}

        Suggest:
        1. Unit tests
        2. Integration tests
        3. End-to-end tests
        4. Performance tests
        5. Security tests

        Return as list of specific test scenarios.
        """

        response_text = self._generate_response(prompt, max_tokens=1000)

        # Extract test suggestions from response
        lines = response_text.split('\n')
        suggestions = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

        return suggestions[:10]  # Limit to 10 suggestions


class SocraticOrchestrator:
    """Main orchestrator that manages all agents and project flow"""

    def __init__(self, api_key: str, db_path: str = "socratic_rag.db"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.db_manager = DatabaseManager(db_path)

        # Initialize agents
        self.agents = {
            "requirements": RequirementsAgent(self.client),
            "technical": TechnicalAgent(self.client),
            "ux": UXAgent(self.client),
            "infrastructure": InfrastructureAgent(self.client),
            "planning": PlanningAgent(self.client),
            "code_generation": CodeGenerationAgent(self.client),
            "validation": ValidationAgent(self.client)
        }

        self.current_user = None
        self.current_project = None

    async def authenticate_user(self, username: str, email: str = None) -> bool:
        """Authenticate or create user"""
        user = self.db_manager.get_user(username)

        if user:
            self.current_user = user
            self.db_manager.update_user_login(user.user_id)
            return True
        elif email:
            try:
                self.current_user = self.db_manager.create_user(username, email)
                return True
            except ValueError:
                return False

        return False

    async def create_project(self, name: str, description: str = "") -> str:
        """Create new project"""
        if not self.current_user:
            raise ValueError("User must be authenticated")

        project = self.db_manager.create_project(name, description, self.current_user.user_id)
        self.current_project = project
        return project.project_id

    async def load_project(self, project_id: str) -> bool:
        """Load existing project"""
        if not self.current_user:
            raise ValueError("User must be authenticated")

        project = self.db_manager.get_project(project_id)

        if project and (project.owner_id == self.current_user.user_id or
                       self.current_user.user_id in project.collaborators):
            self.current_project = project
            return True

        return False

    async def get_user_projects(self) -> List[Dict[str, Any]]:
        """Get all projects for current user"""
        if not self.current_user:
            return []

        projects = self.db_manager.get_user_projects(self.current_user.user_id)
        return [
            {
                "project_id": p.project_id,
                "name": p.name,
                "description": p.description,
                "current_phase": p.current_phase,
                "updated_at": p.updated_at,
                "is_owner": p.owner_id == self.current_user.user_id
            }
            for p in projects
        ]

    async def process_user_input(self, user_input: str) -> str:
        """Process user input and return response"""
        if not self.current_user or not self.current_project:
            return "Please authenticate and select a project first."

        # Determine current phase and relevant agents
        current_phase = ProjectPhase(self.current_project.current_phase)

        if current_phase == ProjectPhase.DISCOVERY:
            return await self._handle_discovery_phase(user_input)
        elif current_phase == ProjectPhase.PLANNING:
            return await self._handle_planning_phase(user_input)
        elif current_phase == ProjectPhase.GENERATION:
            return await self._handle_generation_phase(user_input)
        elif current_phase == ProjectPhase.VALIDATION:
            return await self._handle_validation_phase(user_input)
        else:
            return "Project is complete. You can start a new project or review the generated code."

    async def _handle_discovery_phase(self, user_input: str) -> str:
        """Handle discovery phase with multiple agents"""
        responses = []
        all_questions = []
        context_updates = {}

        # Get responses from all relevant agents
        for agent_name in ["requirements", "technical", "ux", "infrastructure"]:
            agent = self.agents[agent_name]
            response = await agent.process_input(user_input, self.current_project.context)
            responses.append(f"**{agent_name.title()} Agent**: {response.content}")
            all_questions.extend(response.next_questions)

            # Update context
            for key, value in response.context_updates.items():
                if isinstance(value, list):
                    current_list = getattr(self.current_project.context, key, [])
                    updated_list = current_list + [item for item in value if item not in current_list]
                    setattr(self.current_project.context, key, updated_list)
                else:
                    setattr(self.current_project.context, key, value)

        # Check if ready to move to planning
        completeness_score = self._calculate_completeness_score(self.current_project.context)

        if completeness_score > 0.8:
            self.current_project.current_phase = ProjectPhase.PLANNING.value
            responses.append("\n**🎯 Ready for Planning Phase!**")
            responses.append("The requirements are sufficiently detailed. Use 'plan' to create the project specification.")

        # Update project in database
        self.db_manager.update_project(self.current_project)

        # Save conversation
        response_text = "\n\n".join(responses)
        if all_questions:
            response_text += "\n\n**Follow-up Questions:**\n" + "\n".join(f"• {q}" for q in all_questions[:3])

        self.db_manager.save_conversation(
            self.current_project.project_id,
            self.current_user.user_id,
            user_input,
            response_text
        )

        return response_text

    async def _handle_planning_phase(self, user_input: str) -> str:
        """Handle planning phase"""
        if user_input.lower() == "plan":
            planning_agent = self.agents["planning"]
            specification = await planning_agent.create_project_specification(self.current_project.context)

            # Store specification in project context (simplified)
            self.current_project.context.confidence_scores["specification"] = 0.9
            self.current_project.current_phase = ProjectPhase.GENERATION.value
            self.db_manager.update_project(self.current_project)

            response = f"""
**📋 Project Specification Created**

**Project**: {specification.project_name}
**Description**: {specification.description}

**Technical Architecture**:
- Stack: {', '.join(specification.technical_architecture.get('stack', []))}
- Pattern: {specification.technical_architecture.get('pattern', 'Not specified')}

**Next Steps**: Use 'generate' to create the codebase.
            """

            self.db_manager.save_conversation(
                self.current_project.project_id,
                self.current_user.user_id,
                user_input,
                response
            )

            return response
        else:
            return "In planning phase. Use 'plan' to create the project specification."

    async def _handle_generation_phase(self, user_input: str) -> str:
        """Handle code generation phase"""
        if user_input.lower() == "generate":
            # Create specification from context
            planning_agent = self.agents["planning"]
            specification = await planning_agent.create_project_specification(self.current_project.context)

            # Generate code
            code_generation_agent = self.agents["code_generation"]
            generated_files = await code_generation_agent.generate_code(specification)

            # Save generated code
            self.db_manager.save_generated_code(self.current_project.project_id, generated_files)

            # Move to validation phase
            self.current_project.current_phase = ProjectPhase.VALIDATION.value
            self.db_manager.update_project(self.current_project)

            response = f"""
**🚀 Code Generated Successfully!**

Generated {len(generated_files)} files:
{chr(10).join(f"• {filename}" for filename in generated_files.keys())}

**Next Steps**: Use 'validate' to review the generated code.
            """

            self.db_manager.save_conversation(
                self.current_project.project_id,
                self.current_user.user_id,
                user_input,
                response
            )

            return response
        else:
            return "In generation phase. Use 'generate' to create the codebase."

    async def _handle_validation_phase(self, user_input: str) -> str:
        """Handle validation phase"""
        if user_input.lower() == "validate":
            # Get generated code
            generated_files = self.db_manager.get_generated_code(self.current_project.project_id)

            if not generated_files:
                return "No generated code found. Please generate code first."

            # Create specification for validation
            planning_agent = self.agents["planning"]
            specification = await planning_agent.create_project_specification(self.current_project.context)

            # Validate code
            validation_agent = self.agents["validation"]
            validation_results = await validation_agent.validate_code(generated_files, specification)

            # Move to complete phase
            self.current_project.current_phase = ProjectPhase.COMPLETE.value
            self.db_manager.update_project(self.current_project)

            response = f"""
**✅ Code Validation Complete**

**Overall Score**: {validation_results['overall_score']:.2f}/1.0

**File Scores**:
{chr(10).join(f"• {filename}: {score:.2f}" for filename, score in validation_results['file_scores'].items())}

**Issues Found**: {len(validation_results['issues'])}
{chr(10).join(f"• {issue}" for issue in validation_results['issues'][:3])}

**Recommendations**:
{chr(10).join(f"• {rec}" for rec in validation_results['recommendations'][:3])}

**Project Status**: COMPLETE ✅
            """

            self.db_manager.save_conversation(
                self.current_project.project_id,
                self.current_user.user_id,
                user_input,
                response
            )

            return response
        else:
            return "In validation phase. Use 'validate' to review the generated code."

    def _calculate_completeness_score(self, context: ProjectContext) -> float:
        """Calculate how complete the project context is"""
        scores = []

        # Requirements completeness
        req_score = min(len(context.functional_requirements) / 5, 1.0)
        scores.append(req_score)

        # Technical completeness
        tech_score = min(len(context.technical_stack) / 3, 1.0)
        if context.architecture_pattern:
            tech_score += 0.2
        scores.append(min(tech_score, 1.0))

        # UX completeness
        ux_score = min(len(context.ui_components) / 3, 1.0)
        scores.append(ux_score)

        # Infrastructure completeness
        infra_score = 1.0 if context.deployment_target else 0.5
        scores.append(infra_score)

        return sum(scores) / len(scores)

    async def get_project_status(self) -> Dict[str, Any]:
        """Get current project status"""
        if not self.current_project:
            return {"error": "No project selected"}

        return {
            "project_id": self.current_project.project_id,
            "name": self.current_project.name,
            "phase": self.current_project.current_phase,
            "completeness_score": self._calculate_completeness_score(self.current_project.context),
            "context_summary": {
                "functional_requirements": len(self.current_project.context.functional_requirements),
                "technical_stack": len(self.current_project.context.technical_stack),
                "ui_components": len(self.current_project.context.ui_components),
                "deployment_target": bool(self.current_project.context.deployment_target)
            }
        }

    async def get_generated_code(self) -> Dict[str, str]:
        """Get generated code files"""
        if not self.current_project:
            return {}

        return self.db_manager.get_generated_code(self.current_project.project_id)

    async def get_conversation_history(self) -> List[Dict]:
        """Get conversation history for current project"""
        if not self.current_project:
            return []

        return self.db_manager.get_conversation_history(self.current_project.project_id)


# Example usage and CLI interface
async def main():
    """Example usage of the Socratic RAG system"""

    # Initialize system
    orchestrator = SocraticOrchestrator(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        db_path="socratic_rag.db"
    )

    print("🧠 Enhanced Socratic RAG System")
    print("=" * 50)

    # Authentication
    username = input("Enter username: ")
    email = input("Enter email (for new users): ")

    if await orchestrator.authenticate_user(username, email):
        print(f"✅ Authenticated as {username}")
    else:
        print("❌ Authentication failed")
        return

    # Project selection
    projects = await orchestrator.get_user_projects()

    if projects:
        print("\n📁 Your Projects:")
        for i, project in enumerate(projects):
            print(f"{i+1}. {project['name']} ({project['current_phase']})")

        choice = input("\nSelect project (number) or 'new' for new project: ")

        if choice.lower() == 'new':
            name = input("Project name: ")
            description = input("Project description: ")
            project_id = await orchestrator.create_project(name, description)
            print(f"✅ Created project: {project_id}")
        else:
            try:
                project_idx = int(choice) - 1
                if 0 <= project_idx < len(projects):
                    project_id = projects[project_idx]['project_id']
                    if await orchestrator.load_project(project_id):
                        print(f"✅ Loaded project: {projects[project_idx]['name']}")
                    else:
                        print("❌ Failed to load project")
                        return
                else:
                    print("❌ Invalid project selection")
                    return
            except ValueError:
                print("❌ Invalid input")
                return
    else:
        name = input("Project name: ")
        description = input("Project description: ")
        project_id = await orchestrator.create_project(name, description)
        print(f"✅ Created project: {project_id}")

    # Main interaction loop
    print("\n🚀 Starting Socratic dialogue...")
    print("Commands: 'status', 'history', 'code', 'quit'")
    print("=" * 50)

    while True:
        user_input = input("\n💬 You: ")

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'status':
            status = await orchestrator.get_project_status()
            print(f"\n📊 Project Status: {json.dumps(status, indent=2)}")
        elif user_input.lower() == 'history':
            history = await orchestrator.get_conversation_history()
            print(f"\n📜 Conversation History: {len(history)} messages")
            for msg in history[-3:]:  # Show last 3 messages
                print(f"[{msg['timestamp']}] {msg['username']}: {msg['user_input'][:50]}...")
        elif user_input.lower() == 'code':
            code = await orchestrator.get_generated_code()
            if code:
                print(f"\n💻 Generated Code: {len(code)} files")
                for filename in code.keys():
                    print(f"• {filename}")
            else:
                print("No code generated yet")
        else:
            response = await orchestrator.process_user_input(user_input)
            print(f"\n🤖 System: {response}")

if __name__ == "__main__":
    asyncio.run(main())
