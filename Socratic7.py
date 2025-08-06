#!/usr/bin/env python3
"""
Enhanced Socratic RAG System v7.0
Multi-agent architecture with vector database and improved user experience
"""

import os
import json
import hashlib
import getpass
import datetime
import pickle
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import sqlite3
import threading
import time

# Third-party imports
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not found. Install with: pip install chromadb")
    exit(1)

try:
    import anthropic
except ImportError:
    print("Anthropic package not found. Install with: pip install anthropic")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Sentence Transformers not found. Install with: pip install sentence-transformers")
    exit(1)

import numpy as np
from colorama import init, Fore, Back, Style

init(autoreset=True)

# Configuration
CONFIG = {
    'MAX_CONTEXT_LENGTH': 8000,
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    'CLAUDE_MODEL': 'claude-3-sonnet-20240229',
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 1,
    'TOKEN_WARNING_THRESHOLD': 0.8,
    'SESSION_TIMEOUT': 3600,  # 1 hour
    'DATA_DIR': 'socratic_data'
}


# Data Models
@dataclass
class User:
    username: str
    passcode_hash: str
    created_at: datetime.datetime
    projects: List[str]


@dataclass
class ProjectContext:
    project_id: str
    name: str
    owner: str
    collaborators: List[str]
    goals: str
    requirements: List[str]
    tech_stack: List[str]
    constraints: List[str]
    team_structure: str
    language_preferences: str
    deployment_target: str
    code_style: str
    phase: str
    conversation_history: List[Dict]
    created_at: datetime.datetime
    updated_at: datetime.datetime


@dataclass
class KnowledgeEntry:
    id: str
    content: str
    category: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float
    timestamp: datetime.datetime


# Base Agent Class
class Agent(ABC):
    def __init__(self, name: str, orchestrator: 'AgentOrchestrator'):
        self.name = name
        self.orchestrator = orchestrator

    @abstractmethod
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = Fore.GREEN if level == "INFO" else Fore.RED if level == "ERROR" else Fore.YELLOW
        print(f"{color}[{timestamp}] {self.name}: {message}")


# Specialized Agents
class ProjectManagerAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("ProjectManager", orchestrator)

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        if action == 'create_project':
            return self._create_project(request)
        elif action == 'load_project':
            return self._load_project(request)
        elif action == 'save_project':
            return self._save_project(request)
        elif action == 'add_collaborator':
            return self._add_collaborator(request)
        elif action == 'list_projects':
            return self._list_projects(request)

        return {'status': 'error', 'message': 'Unknown action'}

    def _create_project(self, request: Dict) -> Dict:
        project_name = request.get('project_name')
        owner = request.get('owner')

        project_id = str(uuid.uuid4())
        project = ProjectContext(
            project_id=project_id,
            name=project_name,
            owner=owner,
            collaborators=[],
            goals="",
            requirements=[],
            tech_stack=[],
            constraints=[],
            team_structure="individual",
            language_preferences="python",
            deployment_target="local",
            code_style="documented",
            phase="discovery",
            conversation_history=[],
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )

        self.orchestrator.database.save_project(project)
        self.log(f"Created project '{project_name}' with ID {project_id}")

        return {'status': 'success', 'project': project}

    def _load_project(self, request: Dict) -> Dict:
        project_id = request.get('project_id')
        project = self.orchestrator.database.load_project(project_id)

        if project:
            self.log(f"Loaded project '{project.name}'")
            return {'status': 'success', 'project': project}
        else:
            return {'status': 'error', 'message': 'Project not found'}

    def _save_project(self, request: Dict) -> Dict:
        project = request.get('project')
        project.updated_at = datetime.datetime.now()
        self.orchestrator.database.save_project(project)
        self.log(f"Saved project '{project.name}'")
        return {'status': 'success'}

    def _add_collaborator(self, request: Dict) -> Dict:
        project = request.get('project')
        username = request.get('username')

        if username not in project.collaborators:
            project.collaborators.append(username)
            self.orchestrator.database.save_project(project)
            self.log(f"Added collaborator '{username}' to project '{project.name}'")
            return {'status': 'success'}
        else:
            return {'status': 'error', 'message': 'User already a collaborator'}

    def _list_projects(self, request: Dict) -> Dict:
        username = request.get('username')
        projects = self.orchestrator.database.get_user_projects(username)
        return {'status': 'success', 'projects': projects}


class SocraticCounselorAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("SocraticCounselor", orchestrator)
        self.phases = {
            'discovery': self._discovery_questions,
            'analysis': self._analysis_questions,
            'design': self._design_questions,
            'implementation': self._implementation_questions
        }

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        if action == 'generate_question':
            return self._generate_question(request)
        elif action == 'process_response':
            return self._process_response(request)
        elif action == 'advance_phase':
            return self._advance_phase(request)

        return {'status': 'error', 'message': 'Unknown action'}

    def _generate_question(self, request: Dict) -> Dict:
        project = request.get('project')
        context = self.orchestrator.context_analyzer.get_context_summary(project)

        phase_func = self.phases.get(project.phase, self._discovery_questions)
        question = phase_func(project, context)

        return {'status': 'success', 'question': question}

    def _process_response(self, request: Dict) -> Dict:
        project = request.get('project')
        user_response = request.get('response')

        # Add to conversation history
        project.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'user',
            'content': user_response
        })

        # Extract insights using Claude
        insights = self.orchestrator.claude_client.extract_insights(user_response, project)
        self._update_project_context(project, insights)

        return {'status': 'success', 'insights': insights}

    def _discovery_questions(self, project: ProjectContext, context: str) -> str:
        questions = [
            "What specific problem does your project solve?",
            "Who is your target audience or user base?",
            "What are the core features you envision?",
            "Are there similar solutions that exist? How will yours differ?",
            "What are your success criteria for this project?"
        ]

        # Choose question based on conversation history
        answered_topics = len(project.conversation_history)
        if answered_topics < len(questions):
            return questions[answered_topics]
        else:
            return "Based on what you've shared, what aspect would you like to explore deeper?"

    def _analysis_questions(self, project: ProjectContext, context: str) -> str:
        questions = [
            "What technical challenges do you anticipate?",
            "What are your performance requirements?",
            "How will you handle user authentication and security?",
            "What third-party integrations might you need?",
            "How will you test and validate your solution?"
        ]

        phase_responses = [msg for msg in project.conversation_history
                           if msg.get('phase') == 'analysis']

        if len(phase_responses) < len(questions):
            return questions[len(phase_responses)]
        else:
            return "What technical aspect concerns you most at this stage?"

    def _design_questions(self, project: ProjectContext, context: str) -> str:
        questions = [
            "How will you structure your application architecture?",
            "What design patterns will you use?",
            "How will you organize your code and modules?",
            "What development workflow will you follow?",
            "How will you handle error cases and edge scenarios?"
        ]

        phase_responses = [msg for msg in project.conversation_history
                           if msg.get('phase') == 'design']

        if len(phase_responses) < len(questions):
            return questions[len(phase_responses)]
        else:
            return "What design decision would you like to validate?"

    def _implementation_questions(self, project: ProjectContext, context: str) -> str:
        questions = [
            "What will be your first implementation milestone?",
            "How will you handle deployment and DevOps?",
            "What monitoring and logging will you implement?",
            "How will you document your code and API?",
            "What's your plan for maintenance and updates?"
        ]

        phase_responses = [msg for msg in project.conversation_history
                           if msg.get('phase') == 'implementation']

        if len(phase_responses) < len(questions):
            return questions[len(phase_responses)]
        else:
            return "Ready to generate your implementation plan?"

    def _advance_phase(self, request: Dict) -> Dict:
        project = request.get('project')
        phases = ['discovery', 'analysis', 'design', 'implementation']

        current_index = phases.index(project.phase)
        if current_index < len(phases) - 1:
            project.phase = phases[current_index + 1]
            self.log(f"Advanced project to {project.phase} phase")

        return {'status': 'success', 'new_phase': project.phase}

    def _update_project_context(self, project: ProjectContext, insights: Dict):
        """Update project context based on extracted insights"""
        if 'goals' in insights:
            project.goals = insights['goals']
        if 'requirements' in insights:
            project.requirements.extend(insights.get('requirements', []))
        if 'tech_stack' in insights:
            project.tech_stack.extend(insights.get('tech_stack', []))
        if 'constraints' in insights:
            project.constraints.extend(insights.get('constraints', []))


class ContextAnalyzerAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("ContextAnalyzer", orchestrator)

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        if action == 'analyze_context':
            return self._analyze_context(request)
        elif action == 'get_summary':
            return self._get_summary(request)
        elif action == 'find_similar':
            return self._find_similar(request)

        return {'status': 'error', 'message': 'Unknown action'}

    def _analyze_context(self, request: Dict) -> Dict:
        project = request.get('project')

        # Analyze conversation patterns
        patterns = self._identify_patterns(project.conversation_history)

        # Get relevant knowledge
        relevant_knowledge = self.orchestrator.vector_db.search_similar(
            project.goals, top_k=5
        )

        return {
            'status': 'success',
            'patterns': patterns,
            'relevant_knowledge': relevant_knowledge
        }

    def _get_summary(self, project: ProjectContext) -> str:
        """Generate comprehensive project summary"""
        summary_parts = []

        if project.goals:
            summary_parts.append(f"Goals: {project.goals}")
        if project.requirements:
            summary_parts.append(f"Requirements: {', '.join(project.requirements)}")
        if project.tech_stack:
            summary_parts.append(f"Tech Stack: {', '.join(project.tech_stack)}")
        if project.constraints:
            summary_parts.append(f"Constraints: {', '.join(project.constraints)}")

        return "\n".join(summary_parts)

    def _find_similar(self, request: Dict) -> Dict:
        query = request.get('query')
        results = self.orchestrator.vector_db.search_similar(query, top_k=3)
        return {'status': 'success', 'similar_projects': results}

    def _identify_patterns(self, history: List[Dict]) -> Dict:
        """Analyze conversation history for patterns"""
        patterns = {
            'question_count': len([msg for msg in history if msg.get('type') == 'assistant']),
            'response_count': len([msg for msg in history if msg.get('type') == 'user']),
            'topics_covered': [],
            'engagement_level': 'high' if len(history) > 10 else 'medium' if len(history) > 5 else 'low'
        }

        return patterns


class CodeGeneratorAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("CodeGenerator", orchestrator)

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        if action == 'generate_script':
            return self._generate_script(request)
        elif action == 'generate_documentation':
            return self._generate_documentation(request)

        return {'status': 'error', 'message': 'Unknown action'}

    def _generate_script(self, request: Dict) -> Dict:
        project = request.get('project')

        # Build comprehensive context
        context = self._build_generation_context(project)

        # Generate using Claude
        script = self.orchestrator.claude_client.generate_code(context)

        self.log(f"Generated script for project '{project.name}'")

        return {
            'status': 'success',
            'script': script,
            'context_used': context
        }

    def _generate_documentation(self, request: Dict) -> Dict:
        project = request.get('project')
        script = request.get('script')

        documentation = self.orchestrator.claude_client.generate_documentation(
            project, script
        )

        return {
            'status': 'success',
            'documentation': documentation
        }

    def _build_generation_context(self, project: ProjectContext) -> str:
        """Build comprehensive context for code generation"""
        context_parts = [
            f"Project: {project.name}",
            f"Phase: {project.phase}",
            f"Goals: {project.goals}",
            f"Tech Stack: {', '.join(project.tech_stack)}",
            f"Requirements: {', '.join(project.requirements)}",
            f"Constraints: {', '.join(project.constraints)}",
            f"Target: {project.deployment_target}",
            f"Style: {project.code_style}"
        ]

        # Add conversation insights
        if project.conversation_history:
            recent_responses = project.conversation_history[-5:]
            context_parts.append("Recent Discussion:")
            for msg in recent_responses:
                if msg.get('type') == 'user':
                    context_parts.append(f"- {msg['content']}")

        return "\n".join(context_parts)


class SystemMonitorAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("SystemMonitor", orchestrator)
        self.token_usage = []
        self.connection_status = True
        self.last_health_check = datetime.datetime.now()

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        if action == 'track_tokens':
            return self._track_tokens(request)
        elif action == 'check_health':
            return self._check_health(request)
        elif action == 'get_stats':
            return self._get_stats(request)
        elif action == 'check_limits':
            return self._check_limits(request)

        return {'status': 'error', 'message': 'Unknown action'}

    def _track_tokens(self, request: Dict) -> Dict:
        usage = TokenUsage(
            input_tokens=request.get('input_tokens', 0),
            output_tokens=request.get('output_tokens', 0),
            total_tokens=request.get('total_tokens', 0),
            cost_estimate=request.get('cost_estimate', 0.0),
            timestamp=datetime.datetime.now()
        )

        self.token_usage.append(usage)

        # Check if approaching limits
        total_tokens = sum(u.total_tokens for u in self.token_usage[-10:])
        warning = total_tokens > 50000  # Warning threshold

        return {
            'status': 'success',
            'current_usage': usage,
            'warning': warning,
            'total_recent': total_tokens
        }

    def _check_health(self, request: Dict) -> Dict:
        # Test Claude API connection
        try:
            self.orchestrator.claude_client.test_connection()
            self.connection_status = True
            self.last_health_check = datetime.datetime.now()

            return {
                'status': 'success',
                'connection': True,
                'last_check': self.last_health_check
            }
        except Exception as e:
            self.connection_status = False
            self.log(f"Health check failed: {e}", "ERROR")

            return {
                'status': 'error',
                'connection': False,
                'error': str(e)
            }

    def _get_stats(self, request: Dict) -> Dict:
        total_tokens = sum(u.total_tokens for u in self.token_usage)
        total_cost = sum(u.cost_estimate for u in self.token_usage)

        return {
            'status': 'success',
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'api_calls': len(self.token_usage),
            'connection_status': self.connection_status
        }

    def _check_limits(self, request: Dict) -> Dict:
        recent_usage = sum(u.total_tokens for u in self.token_usage[-5:])
        warnings = []

        if recent_usage > 40000:
            warnings.append("High token usage detected")
        if not self.connection_status:
            warnings.append("API connection issues")

        return {
            'status': 'success',
            'warnings': warnings,
            'recent_usage': recent_usage
        }


# Database and Storage Classes
class VectorDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("socratic_knowledge")
        self.embedding_model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])

    def add_knowledge(self, entry: KnowledgeEntry):
        """Add knowledge entry to vector database"""
        if not entry.embedding:
            entry.embedding = self.embedding_model.encode(entry.content).tolist()

        self.collection.add(
            documents=[entry.content],
            metadatas=[entry.metadata],
            ids=[entry.id],
            embeddings=[entry.embedding]
        )

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar knowledge entries"""
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return [{
            'content': doc,
            'metadata': meta,
            'score': dist
        } for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )]

    def delete_entry(self, entry_id: str):
        """Delete knowledge entry"""
        self.collection.delete(ids=[entry_id])


class ProjectDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for project metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                data BLOB,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                passcode_hash TEXT,
                data BLOB,
                created_at TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def save_project(self, project: ProjectContext):
        """Save project to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = pickle.dumps(asdict(project))

        cursor.execute('''
            INSERT OR REPLACE INTO projects (project_id, data, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (project.project_id, data, project.created_at, project.updated_at))

        conn.commit()
        conn.close()

    def load_project(self, project_id: str) -> Optional[ProjectContext]:
        """Load project from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT data FROM projects WHERE project_id = ?', (project_id,))
        result = cursor.fetchone()

        conn.close()

        if result:
            data = pickle.loads(result[0])
            return ProjectContext(**data)
        return None

    def get_user_projects(self, username: str) -> List[Dict]:
        """Get all projects for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT project_id, data FROM projects 
            WHERE json_extract(data, '$.owner') = ? OR 
                  json_extract(data, '$.collaborators') LIKE '%' || ? || '%'
        ''', (username, username))

        results = cursor.fetchall()
        conn.close()

        projects = []
        for project_id, data in results:
            project_data = pickle.loads(data)
            projects.append({
                'project_id': project_id,
                'name': project_data['name'],
                'phase': project_data['phase'],
                'updated_at': project_data['updated_at']
            })

        return projects

    def save_user(self, user: User):
        """Save user to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        data = pickle.dumps(asdict(user))

        cursor.execute('''
            INSERT OR REPLACE INTO users (username, passcode_hash, data, created_at)
            VALUES (?, ?, ?, ?)
        ''', (user.username, user.passcode_hash, data, user.created_at))

        conn.commit()
        conn.close()

    def load_user(self, username: str) -> Optional[User]:
        """Load user from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT data FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()

        conn.close()

        if result:
            data = pickle.loads(result[0])
            return User(**data)
        return None


# Claude API Client
class ClaudeClient:
    def __init__(self, api_key: str, orchestrator: 'AgentOrchestrator'):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.orchestrator = orchestrator

    def extract_insights(self, user_response: str, project: ProjectContext) -> Dict:
        """Extract insights from user response using Claude"""
        prompt = f"""
        Analyze this user response in the context of their project and extract structured insights:

        Project Context:
        - Goals: {project.goals}
        - Phase: {project.phase}
        - Tech Stack: {', '.join(project.tech_stack)}

        User Response: "{user_response}"

        Please extract and return any mentions of:
        1. Goals or objectives
        2. Technical requirements 
        3. Technology preferences
        4. Constraints or limitations
        5. Team structure preferences

        Return as JSON with keys: goals, requirements, tech_stack, constraints, team_structure
        """

        try:
            response = self.client.messages.create(
                model=CONFIG['CLAUDE_MODEL'],
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Track token usage
            self.orchestrator.system_monitor.process({
                'action': 'track_tokens',
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens,
                'cost_estimate': self._calculate_cost(response.usage)
            })

            # Try to parse JSON response
            try:
                import json
                return json.loads(response.content[0].text)
            except:
                # Fallback to simple text analysis
                return {'extracted_text': response.content[0].text}

        except Exception as e:
            print(f"{Fore.RED}Error extracting insights: {e}")
            return {}

    def generate_code(self, context: str) -> str:
        """Generate code based on project context"""
        prompt = f"""
        Generate a complete, functional script based on this project context:

        {context}

        Please create:
        1. A well-structured, documented script
        2. Include proper error handling
        3. Follow best practices for the chosen technology
        4. Add helpful comments explaining key functionality
        5. Include basic testing or validation

        Make it production-ready and maintainable.
        """

        try:
            response = self.client.messages.create(
                model=CONFIG['CLAUDE_MODEL'],
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            # Track token usage
            self.orchestrator.system_monitor.process({
                'action': 'track_tokens',
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens,
                'cost_estimate': self._calculate_cost(response.usage)
            })

            return response.content[0].text

        except Exception as e:
            return f"Error generating code: {e}"

    def generate_documentation(self, project: ProjectContext, script: str) -> str:
        """Generate documentation for the project and script"""
        prompt = f"""
        Create comprehensive documentation for this project:

        Project: {project.name}
        Goals: {project.goals}
        Tech Stack: {', '.join(project.tech_stack)}

        Script:
        {script[:2000]}...  # Truncated for context

        Please include:
        1. Project overview and purpose
        2. Installation instructions
        3. Usage examples
        4. API documentation (if applicable)
        5. Configuration options
        6. Troubleshooting section
        """

        try:
            response = self.client.messages.create(
                model=CONFIG['CLAUDE_MODEL'],
                max_tokens=3000,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )

            # Track token usage
            self.orchestrator.system_monitor.process({
                'action': 'track_tokens',
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens,
                'cost_estimate': self._calculate_cost(response.usage)
            })

            return response.content[0].text

        except Exception as e:
            return f"Error generating documentation: {e}"

    def test_connection(self) -> bool:
        """Test Claude API connection"""
        try:
            response = self.client.messages.create(
                model=CONFIG['CLAUDE_MODEL'],
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception as e:
            raise e

    def _calculate_cost(self, usage) -> float:
        """Calculate estimated cost based on token usage"""
        # Claude 3 Sonnet pricing (approximate)
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015

        input_cost = (usage.input_tokens / 1000) * input_cost_per_1k
        output_cost = (usage.output_tokens / 1000) * output_cost_per_1k

        return input_cost + output_cost


# User Management System
class UserManager:
    def __init__(self, database: ProjectDatabase):
        self.database = database

    def create_user(self, username: str, passcode: str) -> bool:
        """Create new user with passcode"""
        if self.database.load_user(username):
            return False  # User already exists

        passcode_hash = hashlib.sha256(passcode.encode()).hexdigest()
        user = User(
            username=username,
            passcode_hash=passcode_hash,
            created_at=datetime.datetime.now(),
            projects=[]
        )

        self.database.save_user(user)
        return True

    def authenticate_user(self, username: str, passcode: str) -> bool:
        """Authenticate user with passcode"""
        user = self.database.load_user(username)
        if not user:
            return False

        passcode_hash = hashlib.sha256(passcode.encode()).hexdigest()
        return user.passcode_hash == passcode_hash

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.database.load_user(username)


# Session Management
class SessionManager:
    def __init__(self):
        self.active_sessions = {}
        self.session_timeout = CONFIG['SESSION_TIMEOUT']

    def create_session(self, username: str, project_id: str = None) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())

        self.active_sessions[session_id] = {
            'username': username,
            'project_id': project_id,
            'created_at': datetime.datetime.now(),
            'last_activity': datetime.datetime.now()
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session if valid"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        # Check if session expired
        if (datetime.datetime.now() - session['last_activity']).seconds > self.session_timeout:
            del self.active_sessions[session_id]
            return None

        session['last_activity'] = datetime.datetime.now()
        return session

    def update_session(self, session_id: str, project_id: str):
        """Update session with project"""
        session = self.active_sessions.get(session_id)
        if session:
            session['project_id'] = project_id
            session['last_activity'] = datetime.datetime.now()


# Main Orchestrator Class
class AgentOrchestrator:
    def __init__(self, api_key: str):
        # Initialize data directory
        os.makedirs(CONFIG['DATA_DIR'], exist_ok=True)

        # Initialize databases
        self.vector_db = VectorDatabase(os.path.join(CONFIG['DATA_DIR'], 'vector_db'))
        self.database = ProjectDatabase(os.path.join(CONFIG['DATA_DIR'], 'projects.db'))

        # Initialize Claude client
        self.claude_client = ClaudeClient(api_key, self)

        # Initialize managers
        self.user_manager = UserManager(self.database)
        self.session_manager = SessionManager()

        # Initialize agents
        self.project_manager = ProjectManagerAgent(self)
        self.socratic_counselor = SocraticCounselorAgent(self)
        self.context_analyzer = ContextAnalyzerAgent(self)
        self.code_generator = CodeGeneratorAgent(self)
        self.system_monitor = SystemMonitorAgent(self)

        # Initialize knowledge base
        self._load_default_knowledge()

        print(f"{Fore.GREEN}✅ Enhanced Socratic RAG System initialized successfully!")

    def _load_default_knowledge(self):
        """Load default software development knowledge"""
        default_knowledge = [
            {
                'id': 'sdlc_1',
                'content': 'Software Development Life Cycle includes planning, analysis, design, implementation, '
                           'testing, and maintenance phases.',
                'category': 'methodology',
                'metadata': {'topic': 'sdlc', 'importance': 'high'}
            },
            {
                'id': 'arch_1',
                'content': 'Microservices architecture breaks applications into small, independent services that '
                           'communicate over APIs.',
                'category': 'architecture',
                'metadata': {'topic': 'microservices', 'importance': 'medium'}
            },
            {
                'id': 'test_1',
                'content': 'Test-driven development (TDD) involves writing tests before implementing functionality.',
                'category': 'testing',
                'metadata': {'topic': 'tdd', 'importance': 'medium'}
            },
            {
                'id': 'sec_1',
                'content': 'Always validate and sanitize user input to prevent injection attacks and data corruption.',
                'category': 'security',
                'metadata': {'topic': 'input_validation', 'importance': 'high'}
            }
        ]

        for knowledge_data in default_knowledge:
            entry = KnowledgeEntry(**knowledge_data)
            try:
                self.vector_db.add_knowledge(entry)
            except:
                pass  # Entry might already exist


# Enhanced CLI Interface
class EnhancedCLI:
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.current_session = None
        self.current_project = None

    def start(self):
        """Start the enhanced CLI interface"""
        self._print_header()

        # User authentication
        if not self._authenticate_user():
            return

        # Main menu loop
        while True:
            try:
                self._show_main_menu()
                choice = input(f"\n{Fore.CYAN}Enter your choice: {Style.RESET_ALL}").strip()

                if choice == '1':
                    self._create_new_project()
                elif choice == '2':
                    self._load_existing_project()
                elif choice == '3':
                    self._continue_project()
                elif choice == '4':
                    self._generate_script()
                elif choice == '5':
                    self._manage_collaborators()
                elif choice == '6':
                    self._view_system_stats()
                elif choice == '7':
                    self._export_project()
                elif choice == '0':
                    print("..τω Ασκληπιώ οφείλομεν αλετρυόνα, απόδοτε και μη αμελήσετε..")
                    print(f"{Fore.GREEN}Goodbye!")
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Interrupted. Saving session...")
                self._save_current_state()
                break
            except Exception as e:
                print(f"{Fore.RED}An error occurred: {e}")

    def _print_header(self):
        """Print enhanced header"""
        header = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║                  SOCRATIC RAG SYSTEM v7.0                ║
║              Multi-Agent Project Development             ║
║       "Ουδέν οίδα, ούτε διδάσκω τι, αλλά διαπορώ μόνον." ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.GREEN} Agents Active: {Fore.WHITE}ProjectManager | SocraticCounselor | ContextAnalyzer | CodeGenerator | SystemMonitor
{Fore.GREEN} Database: {Fore.WHITE}Vector Database + SQLite
{Fore.GREEN} Security: {Fore.WHITE}User Authentication Enabled
"""
        print(header)

    def _authenticate_user(self) -> bool:
        """Enhanced user authentication"""
        print(f"\n{Fore.YELLOW}🔐 User Authentication Required")
        print("=" * 40)

        while True:
            username = input(f"{Fore.CYAN}Username: {Style.RESET_ALL}").strip()
            if not username:
                continue

            user = self.orchestrator.user_manager.get_user(username)

            if not user:
                print(f"{Fore.YELLOW}User not found. Create new account? (y/n): {Style.RESET_ALL}", end="")
                if input().lower().startswith('y'):
                    passcode = getpass.getpass(f"{Fore.CYAN}Create passcode: {Style.RESET_ALL}")
                    if self.orchestrator.user_manager.create_user(username, passcode):
                        print(f"{Fore.GREEN} User created successfully!")
                        self.current_session = self.orchestrator.session_manager.create_session(username)
                        return True
                    else:
                        print(f"{Fore.RED}❌ Failed to create user.")
                continue

            passcode = getpass.getpass(f"{Fore.CYAN}Passcode: {Style.RESET_ALL}")

            if self.orchestrator.user_manager.authenticate_user(username, passcode):
                print(f"{Fore.GREEN} Authentication successful!")
                self.current_session = self.orchestrator.session_manager.create_session(username)
                return True
            else:
                print(f"{Fore.RED}❌ Invalid credentials. Try again.")

    def _show_main_menu(self):
        """Show enhanced main menu"""
        session = self.orchestrator.session_manager.get_session(self.current_session)
        username = session['username'] if session else "Unknown"

        # Get system stats
        stats = self.orchestrator.system_monitor.process({'action': 'get_stats'})

        menu = f"""
{Fore.BLUE}┌─ MAIN MENU ─────────────────────────────────────────────┐{Style.RESET_ALL}
{Fore.WHITE}│ User: {Fore.GREEN}{username:<20} {Fore.WHITE}│ Tokens: {Fore.YELLOW}{stats.get('total_tokens', 0):<10}{Fore.WHITE} │
{Fore.WHITE}│ Connection: {Fore.GREEN if stats.get('connection_status') else Fore.RED}●{Fore.WHITE} {'Online' if stats.get('connection_status') else 'Offline':<15} │ Cost: ${stats.get('total_cost', 0.0):.4f}      │{Style.RESET_ALL}
{Fore.BLUE}├─────────────────────────────────────────────────────────┤{Style.RESET_ALL}

{Fore.WHITE}1.{Fore.CYAN} 🆕 Create New Project
{Fore.WHITE}2.{Fore.CYAN} 📂 Load Existing Project  
{Fore.WHITE}3.{Fore.CYAN} ▶️  Continue Current Project
{Fore.WHITE}4.{Fore.CYAN} 🔨 Generate Script
{Fore.WHITE}5.{Fore.CYAN} 👥 Manage Collaborators
{Fore.WHITE}6.{Fore.CYAN} 📊 View System Statistics
{Fore.WHITE}7.{Fore.CYAN} 💾 Export Project
{Fore.WHITE}0.{Fore.RED} 🚪 Exit

{Fore.BLUE}└─────────────────────────────────────────────────────────┘{Style.RESET_ALL}
"""
        print(menu)

        # Show project status if available
        if self.current_project:
            print(
                f"{Fore.GREEN}📋 Current Project: {Fore.WHITE}{self.current_project.name} {Fore.YELLOW}({self.current_project.phase} phase)")

    def _create_new_project(self):
        """Create new project with enhanced flow"""
        print(f"\n{Fore.YELLOW}🆕 Creating New Project")
        print("=" * 30)

        project_name = input(f"{Fore.CYAN}Project name: {Style.RESET_ALL}").strip()
        if not project_name:
            print(f"{Fore.RED}Project name required.")
            return

        session = self.orchestrator.session_manager.get_session(self.current_session)

        # Create project through Project Manager Agent
        result = self.orchestrator.project_manager.process({
            'action': 'create_project',
            'project_name': project_name,
            'owner': session['username']
        })

        if result['status'] == 'success':
            self.current_project = result['project']
            self.orchestrator.session_manager.update_session(
                self.current_session,
                self.current_project.project_id
            )

            print(f"{Fore.GREEN}✅ Project '{project_name}' created successfully!")
            print(f"{Fore.BLUE}🆔 Project ID: {self.current_project.project_id}")

            # Start Socratic questioning
            self._start_socratic_session()
        else:
            print(f"{Fore.RED}❌ Failed to create project: {result.get('message')}")

    def _load_existing_project(self):
        """Load existing project"""
        print(f"\n{Fore.YELLOW}📂 Loading Existing Project")
        print("=" * 35)

        session = self.orchestrator.session_manager.get_session(self.current_session)

        # Get user's projects
        result = self.orchestrator.project_manager.process({
            'action': 'list_projects',
            'username': session['username']
        })

        if result['status'] == 'success' and result['projects']:
            print(f"{Fore.CYAN}Your Projects:")
            for i, project in enumerate(result['projects'], 1):
                print(
                    f"{Fore.WHITE}{i}. {Fore.GREEN}{project['name']} {Fore.YELLOW}({project['phase']}) {Fore.BLUE}[{project['updated_at']}]")

            try:
                choice = int(input(f"\n{Fore.CYAN}Select project (number): {Style.RESET_ALL}"))
                if 1 <= choice <= len(result['projects']):
                    selected_project = result['projects'][choice - 1]

                    # Load project details
                    load_result = self.orchestrator.project_manager.process({
                        'action': 'load_project',
                        'project_id': selected_project['project_id']
                    })

                    if load_result['status'] == 'success':
                        self.current_project = load_result['project']
                        self.orchestrator.session_manager.update_session(
                            self.current_session,
                            self.current_project.project_id
                        )
                        print(f"{Fore.GREEN}✅ Project loaded successfully!")
                    else:
                        print(f"{Fore.RED}❌ Failed to load project.")
                else:
                    print(f"{Fore.RED}Invalid selection.")
            except ValueError:
                print(f"{Fore.RED}Invalid input.")
        else:
            print(f"{Fore.YELLOW}No projects found. Create a new project first.")

    def _continue_project(self):
        """Continue current project conversation"""
        if not self.current_project:
            print(f"{Fore.RED}❌ No active project. Load or create a project first.")
            return

        print(f"\n{Fore.GREEN}▶️  Continuing Project: {self.current_project.name}")
        print(f"{Fore.BLUE}Phase: {self.current_project.phase}")
        print("=" * 50)

        self._start_socratic_session()

    def _start_socratic_session(self):
        """Start interactive Socratic questioning session"""
        print(f"\n{Fore.YELLOW}🤖 Starting Socratic Session...")
        print(f"{Fore.BLUE}Type 'summary' to see project overview")
        print(f"{Fore.BLUE}Type 'next' to advance to next phase")
        print(f"{Fore.BLUE}Type 'done' to finish session")
        print("-" * 50)

        while True:
            try:
                # Check token limits
                limit_check = self.orchestrator.system_monitor.process({'action': 'check_limits'})
                if limit_check.get('warnings'):
                    for warning in limit_check['warnings']:
                        print(f"{Fore.YELLOW}⚠️  {warning}")

                # Generate question
                question_result = self.orchestrator.socratic_counselor.process({
                    'action': 'generate_question',
                    'project': self.current_project
                })

                if question_result['status'] == 'success':
                    question = question_result['question']
                    print(f"\n{Fore.CYAN}🤖 Socratic Counselor: {Fore.WHITE}{question}")

                    # Get user response
                    user_response = input(f"\n{Fore.GREEN}You: {Style.RESET_ALL}").strip()

                    if user_response.lower() == 'done':
                        break
                    elif user_response.lower() == 'summary':
                        self._show_project_summary()
                        continue
                    elif user_response.lower() == 'next':
                        self._advance_phase()
                        continue
                    elif not user_response:
                        continue

                    # Process response
                    response_result = self.orchestrator.socratic_counselor.process({
                        'action': 'process_response',
                        'project': self.current_project,
                        'response': user_response
                    })

                    if response_result['status'] == 'success':
                        # Save updated project
                        self.orchestrator.project_manager.process({
                            'action': 'save_project',
                            'project': self.current_project
                        })

                        # Show insights if any
                        insights = response_result.get('insights', {})
                        if insights:
                            print(f"{Fore.BLUE}💡 Insights captured: {', '.join(insights.keys())}")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Session paused. Your progress is saved.")
                break
            except Exception as e:
                print(f"{Fore.RED}Error in session: {e}")

        # Save final state
        self._save_current_state()

    def _show_project_summary(self):
        """Show comprehensive project summary"""
        print(f"\n{Fore.YELLOW}📋 PROJECT SUMMARY")
        print("=" * 40)

        print(f"{Fore.CYAN}Name: {Fore.WHITE}{self.current_project.name}")
        print(f"{Fore.CYAN}Phase: {Fore.WHITE}{self.current_project.phase}")
        print(f"{Fore.CYAN}Owner: {Fore.WHITE}{self.current_project.owner}")

        if self.current_project.collaborators:
            print(f"{Fore.CYAN}Collaborators: {Fore.WHITE}{', '.join(self.current_project.collaborators)}")

        if self.current_project.goals:
            print(f"{Fore.CYAN}Goals: {Fore.WHITE}{self.current_project.goals}")

        if self.current_project.requirements:
            print(f"{Fore.CYAN}Requirements:")
            for req in self.current_project.requirements:
                print(f"  {Fore.WHITE}• {req}")

        if self.current_project.tech_stack:
            print(f"{Fore.CYAN}Tech Stack: {Fore.WHITE}{', '.join(self.current_project.tech_stack)}")

        if self.current_project.constraints:
            print(f"{Fore.CYAN}Constraints:")
            for constraint in self.current_project.constraints:
                print(f"  {Fore.WHITE}• {constraint}")

        print(f"{Fore.CYAN}Conversation Messages: {Fore.WHITE}{len(self.current_project.conversation_history)}")
        print(f"{Fore.CYAN}Last Updated: {Fore.WHITE}{self.current_project.updated_at}")

    def _advance_phase(self):
        """Advance to next project phase"""
        result = self.orchestrator.socratic_counselor.process({
            'action': 'advance_phase',
            'project': self.current_project
        })

        if result['status'] == 'success':
            new_phase = result['new_phase']
            print(f"{Fore.GREEN}✅ Advanced to {new_phase} phase!")

            # Save project
            self.orchestrator.project_manager.process({
                'action': 'save_project',
                'project': self.current_project
            })
        else:
            print(f"{Fore.RED}❌ Cannot advance phase.")

    def _generate_script(self):
        """Generate script for current project"""
        if not self.current_project:
            print(f"{Fore.RED}❌ No active project.")
            return

        print(f"\n{Fore.YELLOW}🔨 Generating Script...")
        print("=" * 30)

        # Show token warning if needed
        stats = self.orchestrator.system_monitor.process({'action': 'get_stats'})
        if stats.get('total_tokens', 0) > 40000:
            print(f"{Fore.YELLOW}⚠️  High token usage detected. This generation will use additional tokens.")

        result = self.orchestrator.code_generator.process({
            'action': 'generate_script',
            'project': self.current_project
        })

        if result['status'] == 'success':
            script = result['script']

            print(f"{Fore.GREEN}✅ Script generated successfully!")
            print(f"\n{Fore.CYAN}Generated Script:")
            print("=" * 50)
            print(f"{Fore.WHITE}{script}")
            print("=" * 50)

            # Ask if user wants to save to file
            save_choice = input(f"\n{Fore.CYAN}Save to file? (y/n): {Style.RESET_ALL}").lower()
            if save_choice.startswith('y'):
                filename = f"{self.current_project.name.replace(' ', '_')}_script.py"
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(script)
                    print(f"{Fore.GREEN}✅ Script saved as '{filename}'")
                except Exception as e:
                    print(f"{Fore.RED}❌ Error saving file: {e}")

            # Ask about documentation
            doc_choice = input(f"{Fore.CYAN}Generate documentation? (y/n): {Style.RESET_ALL}").lower()
            if doc_choice.startswith('y'):
                doc_result = self.orchestrator.code_generator.process({
                    'action': 'generate_documentation',
                    'project': self.current_project,
                    'script': script
                })

                if doc_result['status'] == 'success':
                    print(f"\n{Fore.CYAN}Generated Documentation:")
                    print("=" * 50)
                    print(f"{Fore.WHITE}{doc_result['documentation']}")

        else:
            print(f"{Fore.RED}❌ Failed to generate script: {result.get('message')}")

    def _manage_collaborators(self):
        """Manage project collaborators"""
        if not self.current_project:
            print(f"{Fore.RED}❌ No active project.")
            return

        print(f"\n{Fore.YELLOW}👥 Manage Collaborators")
        print("=" * 30)

        print(f"{Fore.CYAN}Current collaborators:")
        if self.current_project.collaborators:
            for collab in self.current_project.collaborators:
                print(f"  {Fore.WHITE}• {collab}")
        else:
            print(f"  {Fore.WHITE}None")

        print(f"\n{Fore.WHITE}1. Add collaborator")
        print(f"{Fore.WHITE}2. Remove collaborator")
        print(f"{Fore.WHITE}0. Back to main menu")

        choice = input(f"\n{Fore.CYAN}Choice: {Style.RESET_ALL}").strip()

        if choice == '1':
            username = input(f"{Fore.CYAN}Username to add: {Style.RESET_ALL}").strip()
            if username:
                result = self.orchestrator.project_manager.process({
                    'action': 'add_collaborator',
                    'project': self.current_project,
                    'username': username
                })

                if result['status'] == 'success':
                    print(f"{Fore.GREEN}✅ Added collaborator '{username}'")
                else:
                    print(f"{Fore.RED}❌ {result.get('message')}")

        elif choice == '2':
            if self.current_project.collaborators:
                username = input(f"{Fore.CYAN}Username to remove: {Style.RESET_ALL}").strip()
                if username in self.current_project.collaborators:
                    self.current_project.collaborators.remove(username)
                    self.orchestrator.project_manager.process({
                        'action': 'save_project',
                        'project': self.current_project
                    })
                    print(f"{Fore.GREEN}✅ Removed collaborator '{username}'")
                else:
                    print(f"{Fore.RED}❌ User not found in collaborators")
            else:
                print(f"{Fore.YELLOW}No collaborators to remove")

    def _view_system_stats(self):
        """View comprehensive system statistics"""
        print(f"\n{Fore.YELLOW}📊 System Statistics")
        print("=" * 30)

        # Get stats from System Monitor
        stats = self.orchestrator.system_monitor.process({'action': 'get_stats'})
        health = self.orchestrator.system_monitor.process({'action': 'check_health'})

        print(
            f"{Fore.CYAN}API Connection: {Fore.GREEN if health.get('connection') else Fore.RED}{'✅ Online' if health.get('connection') else '❌ Offline'}")
        print(f"{Fore.CYAN}Total API Calls: {Fore.WHITE}{stats.get('api_calls', 0)}")
        print(f"{Fore.CYAN}Total Tokens Used: {Fore.WHITE}{stats.get('total_tokens', 0):,}")
        print(f"{Fore.CYAN}Estimated Cost: {Fore.WHITE}${stats.get('total_cost', 0.0):.6f}")

        if health.get('last_check'):
            print(f"{Fore.CYAN}Last Health Check: {Fore.WHITE}{health['last_check']}")

        # Session info
        session = self.orchestrator.session_manager.get_session(self.current_session)
        if session:
            print(f"\n{Fore.CYAN}Session Duration: {Fore.WHITE}{datetime.datetime.now() - session['created_at']}")

        # Active projects count
        if session:
            projects = self.orchestrator.project_manager.process({
                'action': 'list_projects',
                'username': session['username']
            })

            if projects['status'] == 'success':
                print(f"{Fore.CYAN}Your Projects: {Fore.WHITE}{len(projects['projects'])}")

    def _export_project(self):
        """Export project data"""
        if not self.current_project:
            print(f"{Fore.RED}❌ No active project.")
            return

        print(f"\n{Fore.YELLOW}💾 Export Project")
        print("=" * 20)

        export_data = {
            'project_info': asdict(self.current_project),
            'export_date': datetime.datetime.now().isoformat(),
            'version': '7.0'
        }

        filename = f"{self.current_project.name.replace(' ', '_')}_export.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)

            print(f"{Fore.GREEN}✅ Project exported as '{filename}'")
            print(f"{Fore.BLUE}📝 Contains: Project data, conversation history, and metadata")

        except Exception as e:
            print(f"{Fore.RED}❌ Export failed: {e}")

    def _save_current_state(self):
        """Save current state"""
        if self.current_project:
            self.orchestrator.project_manager.process({
                'action': 'save_project',
                'project': self.current_project
            })
            print(f"{Fore.GREEN}💾 Project state saved.")


# Main execution
def main():
    """Main application entry point"""
    try:
        # Get API key
        api_key = os.getenv('API_KEY_CLAUDE')
        if not api_key:
            print(f"{Fore.RED}❌ API_KEY_CLAUDE environment variable not set.")
            print(f"{Fore.YELLOW}Please set it with your Anthropic API key:")
            print(f"{Fore.WHITE}export API_KEY_CLAUDE='your_key_here'")
            return

        # Initialize system
        orchestrator = AgentOrchestrator(api_key)

        # Start CLI
        cli = EnhancedCLI(orchestrator)
        cli.start()

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Application interrupted by user.")
    except Exception as e:
        print(f"{Fore.RED}❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
