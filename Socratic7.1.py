#!/usr/bin/env python3
# TODO An unexpected error occurred. Please check the logs.
# 2025-08-21 11:13:18,285 - httpx - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
# 2025-08-21 11:13:18,287 - __main__ - ERROR - Unexpected error in CLI: 'list' object has no attribute 'lower'
import os
import json
import hashlib
import datetime
import pickle
import uuid
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import sqlite3
import threading
import time
import numpy as np
from colorama import init, Fore, Back, Style
import logging
from contextlib import contextmanager

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

init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('socratic_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration with environment-specific overrides
CONFIG = {
    'MAX_CONTEXT_LENGTH': int(os.getenv('MAX_CONTEXT_LENGTH', '8000')),
    'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
    'CLAUDE_MODEL': os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022'),
    'MAX_RETRIES': int(os.getenv('MAX_RETRIES', '3')),
    'RETRY_DELAY': int(os.getenv('RETRY_DELAY', '1')),
    'TOKEN_WARNING_THRESHOLD': float(os.getenv('TOKEN_WARNING_THRESHOLD', '0.8')),
    'SESSION_TIMEOUT': int(os.getenv('SESSION_TIMEOUT', '3600')),
    'DATA_DIR': os.getenv('DATA_DIR', 'socratic_data'),
    'BATCH_SIZE': int(os.getenv('BATCH_SIZE', '5')),
    'CACHE_SIZE': int(os.getenv('CACHE_SIZE', '100'))
}


# Data Models (unchanged)
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


@dataclass
class ConflictInfo:
    conflict_id: str
    conflict_type: str
    old_value: str
    new_value: str
    old_author: str
    new_author: str
    old_timestamp: str
    new_timestamp: str
    severity: str
    suggestions: List[str]


# Connection Pool for Database
class DatabaseConnectionPool:
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool = []
        self._lock = threading.Lock()
        self._init_pool()

    def _init_pool(self):
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._pool.append(conn)

    @contextmanager
    def get_connection(self):
        with self._lock:
            if self._pool:
                conn = self._pool.pop()
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row

        try:
            yield conn
        finally:
            with self._lock:
                if len(self._pool) < self.pool_size:
                    self._pool.append(conn)
                else:
                    conn.close()


# Memory Cache Implementation
class MemoryCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None

    def set(self, key: str, value: Any):
        with self._lock:
            if len(self.cache) >= self.max_size:
                self._evict_oldest()

            self.cache[key] = value
            self.access_times[key] = time.time()

    def _evict_oldest(self):
        if self.access_times:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

    def clear(self):
        with self._lock:
            self.cache.clear()
            self.access_times.clear()


# Base Agent Class (enhanced with logging)
class Agent(ABC):
    def __init__(self, name: str, orchestrator: 'AgentOrchestrator'):
        self.name = name
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"Agent.{name}")

    @abstractmethod
    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = Fore.GREEN if level == "INFO" else Fore.RED if level == "ERROR" else Fore.YELLOW
        print(f"{color}[{timestamp}] {self.name}: {message}")

        # Also log to file
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARN":
            self.logger.warning(message)
        else:
            self.logger.info(message)


# Enhanced Session Manager Agent (combines ProjectManager functionality)
class SessionManagerAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("SessionManager", orchestrator)
        self.cache = MemoryCache(CONFIG['CACHE_SIZE'])

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
        elif action == 'list_collaborators':
            return self._list_collaborators(request)
        elif action == 'remove_collaborator':
            return self._remove_collaborator(request)
        elif action == 'authenticate_user':
            return self._authenticate_user(request)
        elif action == 'create_user':
            return self._create_user(request)
        elif action == 'save_session_state':
            return self._save_session_state(request)
        elif action == 'restore_session_state':
            return self._restore_session_state(request)

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
        self.cache.set(f"project_{project_id}", project)
        self.log(f"Created project '{project_name}' with ID {project_id}")

        return {'status': 'success', 'project': project}

    def _load_project(self, request: Dict) -> Dict:
        project_id = request.get('project_id')

        # Check cache first
        cached_project = self.cache.get(f"project_{project_id}")
        if cached_project:
            self.log(f"Loaded project '{cached_project.name}' from cache")
            return {'status': 'success', 'project': cached_project}

        project = self.orchestrator.database.load_project(project_id)
        if project:
            self.cache.set(f"project_{project_id}", project)
            self.log(f"Loaded project '{project.name}' from database")
            return {'status': 'success', 'project': project}
        else:
            return {'status': 'error', 'message': 'Project not found'}

    def _save_project(self, request: Dict) -> Dict:
        project = request.get('project')
        project.updated_at = datetime.datetime.now()
        self.orchestrator.database.save_project(project)
        self.cache.set(f"project_{project.project_id}", project)
        self.log(f"Saved project '{project.name}'")
        return {'status': 'success'}

    def _authenticate_user(self, request: Dict) -> Dict:
        username = request.get('username')
        passcode = request.get('passcode')

        # Check cache first
        cached_user = self.cache.get(f"user_{username}")
        user = cached_user if cached_user else self.orchestrator.database.load_user(username)

        if not user:
            return {'status': 'error', 'message': 'User not found'}

        passcode_hash = hashlib.sha256(passcode.encode()).hexdigest()
        if user.passcode_hash != passcode_hash:
            return {'status': 'error', 'message': 'Invalid passcode'}

        if not cached_user:
            self.cache.set(f"user_{username}", user)

        return {'status': 'success', 'user': user}

    def _create_user(self, request: Dict) -> Dict:
        username = request.get('username')
        passcode = request.get('passcode')

        if self.orchestrator.database.user_exists(username):
            return {'status': 'error', 'message': 'Username already exists'}

        passcode_hash = hashlib.sha256(passcode.encode()).hexdigest()
        user = User(
            username=username,
            passcode_hash=passcode_hash,
            created_at=datetime.datetime.now(),
            projects=[]
        )

        self.orchestrator.database.save_user(user)
        self.cache.set(f"user_{username}", user)

        return {'status': 'success', 'user': user}

    def _save_session_state(self, request: Dict) -> Dict:
        """Save current session state for resuming"""
        session_data = {
            'current_user': request.get('current_user'),
            'current_project_id': request.get('current_project_id'),
            'timestamp': datetime.datetime.now().isoformat()
        }

        session_file = os.path.join(CONFIG['DATA_DIR'], 'last_session.json')
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
            return {'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _restore_session_state(self, request: Dict) -> Dict:
        """Restore last session state"""
        session_file = os.path.join(CONFIG['DATA_DIR'], 'last_session.json')
        try:
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                return {'status': 'success', 'session_data': session_data}
            else:
                return {'status': 'error', 'message': 'No previous session found'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    # ... (include other methods from ProjectManagerAgent with caching enhancements)
    def _add_collaborator(self, request: Dict) -> Dict:
        project = request.get('project')
        username = request.get('username')

        if username not in project.collaborators:
            project.collaborators.append(username)
            self.orchestrator.database.save_project(project)
            self.cache.set(f"project_{project.project_id}", project)
            self.log(f"Added collaborator '{username}' to project '{project.name}'")
            return {'status': 'success'}
        else:
            return {'status': 'error', 'message': 'User already a collaborator'}

    def _list_projects(self, request: Dict) -> Dict:
        username = request.get('username')

        # Check cache first
        cache_key = f"user_projects_{username}"
        cached_projects = self.cache.get(cache_key)
        if cached_projects:
            return {'status': 'success', 'projects': cached_projects}

        projects = self.orchestrator.database.get_user_projects(username)
        self.cache.set(cache_key, projects)
        return {'status': 'success', 'projects': projects}

    def _list_collaborators(self, request: Dict) -> Dict:
        project = request.get('project')

        collaborators_info = []
        collaborators_info.append({
            'username': project.owner,
            'role': 'owner'
        })

        for collaborator in project.collaborators:
            collaborators_info.append({
                'username': collaborator,
                'role': 'collaborator'
            })

        return {
            'status': 'success',
            'collaborators': collaborators_info,
            'total_count': len(collaborators_info)
        }

    def _remove_collaborator(self, request: Dict) -> Dict:
        project = request.get('project')
        username = request.get('username')
        requester = request.get('requester')

        if requester != project.owner:
            return {'status': 'error', 'message': 'Only project owner can remove collaborators'}

        if username == project.owner:
            return {'status': 'error', 'message': 'Cannot remove project owner'}

        if username in project.collaborators:
            project.collaborators.remove(username)
            self.orchestrator.database.save_project(project)
            self.cache.set(f"project_{project.project_id}", project)
            self.log(f"Removed collaborator '{username}' from project '{project.name}'")
            return {'status': 'success'}
        else:
            return {'status': 'error', 'message': 'User is not a collaborator'}


# Enhanced Conversation Engine (combines SocraticCounselor and ConflictDetector)
class ConversationEngineAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("ConversationEngine", orchestrator)
        self.use_dynamic_questions = True
        self.max_questions_per_phase = 5
        self.response_cache = MemoryCache(50)  # Cache for API responses

        # Conflict detection rules
        self.conflict_rules = {
            'databases': ['mysql', 'postgresql', 'sqlite', 'mongodb', 'redis'],
            'frontend_frameworks': ['react', 'vue', 'angular', 'svelte'],
            'backend_frameworks': ['django', 'flask', 'fastapi', 'express'],
            'languages': ['python', 'javascript', 'java', 'go', 'rust'],
            'deployment': ['aws', 'azure', 'gcp', 'heroku', 'vercel'],
            'mobile': ['react native', 'flutter', 'native ios', 'native android']
        }

        # Static questions fallback
        self.static_questions = {
            'discovery': [
                "What specific problem does your project solve?",
                "Who is your target audience or user base?",
                "What are the core features you envision?",
                "Are there similar solutions that exist? How will yours differ?",
                "What are your success criteria for this project?"
            ],
            'analysis': [
                "What technical challenges do you anticipate?",
                "What are your performance requirements?",
                "How will you handle user authentication and security?",
                "What third-party integrations might you need?",
                "How will you test and validate your solution?"
            ],
            'design': [
                "How will you structure your application architecture?",
                "What design patterns will you use?",
                "How will you organize your code and modules?",
                "What development workflow will you follow?",
                "How will you handle error cases and edge scenarios?"
            ],
            'implementation': [
                "What will be your first implementation milestone?",
                "How will you handle deployment and DevOps?",
                "What monitoring and logging will you implement?",
                "How will you document your code and API?",
                "What's your plan for maintenance and updates?"
            ]
        }

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        if action == 'generate_question':
            return self._generate_question(request)
        elif action == 'process_response':
            return self._process_response(request)
        elif action == 'advance_phase':
            return self._advance_phase(request)
        elif action == 'toggle_dynamic_questions':
            self.use_dynamic_questions = not self.use_dynamic_questions
            return {'status': 'success', 'dynamic_mode': self.use_dynamic_questions}
        elif action == 'detect_conflicts':
            return self._detect_conflicts(request)
        elif action == 'batch_process_insights':
            return self._batch_process_insights(request)

        return {'status': 'error', 'message': 'Unknown action'}

    def _generate_question(self, request: Dict) -> Dict:
        project = request.get('project')
        context = self.orchestrator.context_analyzer.get_context_summary(project)

        # Count questions already asked in this phase
        phase_questions = [msg for msg in project.conversation_history
                           if msg.get('type') == 'assistant' and msg.get('phase') == project.phase]

        # Check cache for similar context
        cache_key = f"question_{project.phase}_{len(phase_questions)}_{hash(context[:100])}"
        cached_question = self.response_cache.get(cache_key)
        if cached_question:
            project.conversation_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'type': 'assistant',
                'content': cached_question,
                'phase': project.phase,
                'question_number': len(phase_questions) + 1
            })
            return {'status': 'success', 'question': cached_question}

        if self.use_dynamic_questions:
            question = self._generate_dynamic_question(project, context, len(phase_questions))
        else:
            question = self._generate_static_question(project, len(phase_questions))

        # Cache the question
        self.response_cache.set(cache_key, question)

        # Store the question in conversation history
        project.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'assistant',
            'content': question,
            'phase': project.phase,
            'question_number': len(phase_questions) + 1
        })

        return {'status': 'success', 'question': question}

    def _generate_dynamic_question(self, project: ProjectContext, context: str, question_count: int) -> str:
        """Generate contextual questions using Claude with retry logic"""
        recent_conversation = self._get_recent_conversation(project)
        relevant_knowledge = self._get_relevant_knowledge(context)

        prompt = self._build_question_prompt(project, context, recent_conversation, relevant_knowledge, question_count)

        # Exponential backoff retry logic
        for attempt in range(CONFIG['MAX_RETRIES']):
            try:
                question = self.orchestrator.claude_client.generate_socratic_question(prompt)
                self.log(f"Generated dynamic question for {project.phase} phase")
                return question
            except Exception as e:
                wait_time = CONFIG['RETRY_DELAY'] * (2 ** attempt)
                self.log(f"Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s", "WARN")
                if attempt < CONFIG['MAX_RETRIES'] - 1:
                    time.sleep(wait_time)

        # Final fallback to static question
        self.log("All dynamic question attempts failed, using static fallback", "WARN")
        return self._generate_static_question(project, question_count)

    def _get_recent_conversation(self, project: ProjectContext) -> str:
        """Get recent conversation for context"""
        if not project.conversation_history:
            return ""

        recent_messages = project.conversation_history[-4:]
        conversation = ""
        for msg in recent_messages:
            role = "Assistant" if msg['type'] == 'assistant' else "User"
            conversation += f"{role}: {msg['content']}\n"
        return conversation

    def _get_relevant_knowledge(self, context: str) -> str:
        """Get relevant knowledge from vector database"""
        if not context:
            return ""

        knowledge_results = self.orchestrator.vector_db.search_similar(context, top_k=3)
        if knowledge_results:
            return "\n".join([result['content'][:200] + "..." for result in knowledge_results])
        return ""

    def _build_question_prompt(self, project: ProjectContext, context: str,
                               recent_conversation: str, relevant_knowledge: str, question_count: int) -> str:
        """Build optimized prompt for dynamic question generation"""

        phase_descriptions = {
            'discovery': "exploring the problem space, understanding user needs, and defining project goals",
            'analysis': "analyzing technical requirements, identifying challenges, and planning solutions",
            'design': "designing architecture, choosing patterns, and planning implementation structure",
            'implementation': "planning development steps, deployment strategy, and maintenance approach"
        }

        phase_focus = {
            'discovery': "problem definition, user needs, market research, competitive analysis",
            'analysis': "technical feasibility, performance requirements, security considerations, integrations",
            'design': "architecture patterns, code organization, development workflow, error handling",
            'implementation': "development milestones, deployment pipeline, monitoring, documentation"
        }

        # Compressed prompt to reduce token usage
        return f"""Socratic tutor for software project.

Project: {project.name} | Phase: {project.phase} | Q#{question_count + 1}
Goals: {project.goals or 'TBD'}
Stack: {', '.join(project.tech_stack[:3]) if project.tech_stack else 'TBD'}

Context: {context[:500]}
Recent: {recent_conversation[-300:] if recent_conversation else 'None'}

Focus: {phase_focus.get(project.phase, '')}

Generate ONE insightful question that builds on discussion and helps deeper thinking about {project.phase} phase.
Be conversational, specific to their goals/stack, thought-provoking but not overwhelming.

Question only:"""

    def _generate_static_question(self, project: ProjectContext, question_count: int) -> str:
        """Generate questions from static predefined lists"""
        questions = self.static_questions.get(project.phase, [])

        if question_count < len(questions):
            return questions[question_count]
        else:
            fallbacks = {
                'discovery': "What other aspects of the problem space should we explore?",
                'analysis': "What technical considerations haven't we discussed yet?",
                'design': "What design decisions are you still uncertain about?",
                'implementation': "What implementation details would you like to work through?"
            }
            return fallbacks.get(project.phase, "What would you like to explore further?")

    def _process_response(self, request: Dict) -> Dict:
        project = request.get('project')
        user_response = request.get('response')
        current_user = request.get('current_user')

        # Add to conversation history
        project.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'user',
            'content': user_response,
            'phase': project.phase,
            'author': current_user
        })

        # Extract insights using Claude with retry logic
        insights = self._extract_insights_with_retry(user_response, project)

        # Real-time conflict detection
        if insights:
            conflict_result = self._detect_conflicts({
                'project': project,
                'new_insights': insights,
                'current_user': current_user
            })

            if conflict_result['status'] == 'success' and conflict_result['conflicts']:
                conflicts_resolved = self._handle_conflicts_realtime(conflict_result['conflicts'], project)
                if not conflicts_resolved:
                    return {'status': 'success', 'insights': insights, 'conflicts_pending': True}

        # Update context only if no conflicts or conflicts were resolved
        self._update_project_context(project, insights)

        return {'status': 'success', 'insights': insights}

    def _extract_insights_with_retry(self, user_response: str, project: ProjectContext) -> Dict:
        """Extract insights with exponential backoff retry"""
        for attempt in range(CONFIG['MAX_RETRIES']):
            try:
                return self.orchestrator.claude_client.extract_insights(user_response, project)
            except Exception as e:
                wait_time = CONFIG['RETRY_DELAY'] * (2 ** attempt)
                self.log(f"Insight extraction attempt {attempt + 1} failed: {e}, retrying in {wait_time}s", "WARN")
                if attempt < CONFIG['MAX_RETRIES'] - 1:
                    time.sleep(wait_time)

        # Return empty insights if all attempts fail
        self.log("All insight extraction attempts failed", "ERROR")
        return {}

    def _batch_process_insights(self, request: Dict) -> Dict:
        """Process multiple insights at once to reduce API calls"""
        insights_batch = request.get('insights_batch', [])
        project = request.get('project')

        if not insights_batch:
            return {'status': 'success', 'processed': 0}

        # Combine insights into single API call
        combined_insights = {}
        for insights in insights_batch:
            for key, value in insights.items():
                if key not in combined_insights:
                    combined_insights[key] = []
                if isinstance(value, list):
                    combined_insights[key].extend(value)
                else:
                    combined_insights[key].append(value)

        # Update project context with combined insights
        self._update_project_context(project, combined_insights)

        return {'status': 'success', 'processed': len(insights_batch)}

    # ... (include conflict detection methods from ConflictDetectorAgent)
    def _detect_conflicts(self, request: Dict) -> Dict:
        project = request.get('project')
        new_insights = request.get('new_insights')
        current_user = request.get('current_user')

        conflicts = []
        if not new_insights or not isinstance(new_insights, dict):
            return {'status': 'success', 'conflicts': []}

        conflicts.extend(self._check_tech_stack_conflicts(project, new_insights, current_user))
        conflicts.extend(self._check_requirements_conflicts(project, new_insights, current_user))
        conflicts.extend(self._check_goals_conflicts(project, new_insights, current_user))
        conflicts.extend(self._check_constraints_conflicts(project, new_insights, current_user))

        return {'status': 'success', 'conflicts': conflicts}

    def _check_tech_stack_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        conflicts = []
        new_tech = new_insights.get('tech_stack', [])
        if not isinstance(new_tech, list):
            new_tech = [new_tech] if new_tech else []

        for new_item in new_tech:
            if not new_item:
                continue

            new_item_lower = new_item.lower()
            for existing_item in project.tech_stack:
                existing_lower = existing_item.lower()
                conflict_category = self._find_conflict_category(new_item_lower, existing_lower)

                if conflict_category:
                    original_author = self._find_spec_author(project, 'tech_stack', existing_item)
                    conflict = ConflictInfo(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type='tech_stack',
                        old_value=existing_item,
                        new_value=new_item,
                        old_author=original_author,
                        new_author=current_user,
                        old_timestamp=project.created_at.isoformat(),
                        new_timestamp=datetime.datetime.now().isoformat(),
                        severity='high' if conflict_category in ['databases', 'languages'] else 'medium',
                        suggestions=self._generate_tech_suggestions(conflict_category, existing_item, new_item)
                    )
                    conflicts.append(conflict)

        return conflicts

    def _find_conflict_category(self, item1: str, item2: str) -> Optional[str]:
        for category, items in self.conflict_rules.items():
            if any(item1 in tech.lower() for tech in items) and any(item2 in tech.lower() for tech in items):
                return category
        return None

    def _find_spec_author(self, project: ProjectContext, spec_type: str, spec_value: str) -> str:
        """Find the author who originally specified this value"""
        for msg in project.conversation_history:
            if msg.get('type') == 'user' and spec_value.lower() in msg.get('content', '').lower():
                return msg.get('author', project.owner)
        return project.owner

    def _generate_tech_suggestions(self, category: str, old_tech: str, new_tech: str) -> List[str]:
        """Generate suggestions for resolving tech stack conflicts"""
        suggestions = []

        if category == 'databases':
            suggestions.extend([
                f"Consider using {old_tech} as primary database and {new_tech} for specific use cases",
                f"Evaluate performance requirements to choose between {old_tech} and {new_tech}",
                f"Use {old_tech} for structured data and {new_tech} for different data patterns"
            ])
        elif category == 'frontend_frameworks':
            suggestions.extend([
                f"Standardize on either {old_tech} or {new_tech} for consistency",
                f"Consider team expertise when choosing between {old_tech} and {new_tech}",
                f"Evaluate project requirements to determine if {old_tech} or {new_tech} fits better"
            ])
        elif category == 'languages':
            suggestions.extend([
                f"Use {old_tech} for backend and {new_tech} for specific modules",
                f"Consider migration path from {old_tech} to {new_tech} if needed",
                f"Evaluate team skills and project needs for {old_tech} vs {new_tech}"
            ])
        else:
            suggestions.extend([
                f"Discuss trade-offs between {old_tech} and {new_tech}",
                f"Consider combining both {old_tech} and {new_tech} if compatible",
                f"Evaluate which approach ({old_tech} or {new_tech}) better serves project goals"
            ])

        return suggestions

    def _check_requirements_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        """Check for conflicts in requirements"""
        conflicts = []
        new_requirements = new_insights.get('requirements', [])
        if not isinstance(new_requirements, list):
            new_requirements = [new_requirements] if new_requirements else []

        for new_req in new_requirements:
            if not new_req:
                continue

            for existing_req in project.requirements:
                if self._requirements_conflict(new_req, existing_req):
                    original_author = self._find_spec_author(project, 'requirements', existing_req)
                    conflict = ConflictInfo(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type='requirements',
                        old_value=existing_req,
                        new_value=new_req,
                        old_author=original_author,
                        new_author=current_user,
                        old_timestamp=project.created_at.isoformat(),
                        new_timestamp=datetime.datetime.now().isoformat(),
                        severity='medium',
                        suggestions=[
                            f"Clarify priority between '{existing_req}' and '{new_req}'",
                            f"Consider if both requirements can coexist",
                            f"Refine requirements to resolve contradiction"
                        ]
                    )
                    conflicts.append(conflict)

        return conflicts

    def _requirements_conflict(self, req1: str, req2: str) -> bool:
        """Check if two requirements conflict"""
        conflict_keywords = [
            ('fast', 'slow'), ('quick', 'thorough'), ('simple', 'complex'),
            ('minimal', 'comprehensive'), ('lightweight', 'feature-rich'),
            ('basic', 'advanced'), ('free', 'premium'), ('offline', 'online')
        ]

        req1_lower = req1.lower()
        req2_lower = req2.lower()

        for word1, word2 in conflict_keywords:
            if word1 in req1_lower and word2 in req2_lower:
                return True
            if word2 in req1_lower and word1 in req2_lower:
                return True

        return False

    def _check_goals_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        """Check for conflicts in project goals"""
        conflicts = []
        new_goals = new_insights.get('goals', '')

        if new_goals and project.goals and self._goals_conflict(project.goals, new_goals):
            original_author = self._find_spec_author(project, 'goals', project.goals)
            conflict = ConflictInfo(
                conflict_id=str(uuid.uuid4()),
                conflict_type='goals',
                old_value=project.goals,
                new_value=new_goals,
                old_author=original_author,
                new_author=current_user,
                old_timestamp=project.created_at.isoformat(),
                new_timestamp=datetime.datetime.now().isoformat(),
                severity='high',
                suggestions=[
                    "Clarify the primary objective of the project",
                    "Consider if goals can be combined or refined",
                    "Prioritize goals by importance and timeline"
                ]
            )
            conflicts.append(conflict)

        return conflicts

    def _goals_conflict(self, goal1: str, goal2: str) -> bool:
        """Check if two goals conflict"""
        conflicting_pairs = [
            ('profit', 'free'), ('commercial', 'open source'),
            ('speed', 'security'), ('simple', 'comprehensive'),
            ('individual', 'team'), ('prototype', 'production')
        ]

        goal1_lower = goal1.lower()
        goal2_lower = goal2.lower()

        for word1, word2 in conflicting_pairs:
            if word1 in goal1_lower and word2 in goal2_lower:
                return True
            if word2 in goal1_lower and word1 in goal2_lower:
                return True

        return False

    def _check_constraints_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        """Check for conflicts in project constraints"""
        conflicts = []
        new_constraints = new_insights.get('constraints', [])
        if not isinstance(new_constraints, list):
            new_constraints = [new_constraints] if new_constraints else []

        for new_constraint in new_constraints:
            if not new_constraint:
                continue

            for existing_constraint in project.constraints:
                if self._constraints_conflict(existing_constraint, new_constraint):
                    original_author = self._find_spec_author(project, 'constraints', existing_constraint)
                    conflict = ConflictInfo(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type='constraints',
                        old_value=existing_constraint,
                        new_value=new_constraint,
                        old_author=original_author,
                        new_author=current_user,
                        old_timestamp=project.created_at.isoformat(),
                        new_timestamp=datetime.datetime.now().isoformat(),
                        severity='medium',
                        suggestions=[
                            f"Evaluate feasibility of both '{existing_constraint}' and '{new_constraint}'",
                            "Consider trade-offs between constraints",
                            "Prioritize constraints by business impact"
                        ]
                    )
                    conflicts.append(conflict)

        return conflicts

    def _constraints_conflict(self, constraint1: str, constraint2: str) -> bool:
        """Check if two constraints conflict"""
        conflicting_patterns = [
            ('budget', 'no budget'), ('time', 'no deadline'),
            ('team', 'solo'), ('cloud', 'on-premise'),
            ('mobile', 'desktop only'), ('real-time', 'batch processing')
        ]

        c1_lower = constraint1.lower()
        c2_lower = constraint2.lower()

        for pattern1, pattern2 in conflicting_patterns:
            if pattern1 in c1_lower and pattern2 in c2_lower:
                return True
            if pattern2 in c1_lower and pattern1 in c2_lower:
                return True

        return False

    def _handle_conflicts_realtime(self, conflicts: List[ConflictInfo], project: ProjectContext) -> bool:
        """Handle conflicts in real-time with user interaction"""
        if not conflicts:
            return True

        print(f"\n{Fore.RED}🚨 Conflicts Detected! 🚨")
        print(f"{Style.BRIGHT}The following conflicts need resolution:")

        for i, conflict in enumerate(conflicts, 1):
            print(f"\n{Fore.YELLOW}Conflict {i}: {conflict.conflict_type.title()}")
            print(f"  Previous: {conflict.old_value} (by {conflict.old_author})")
            print(f"  Current:  {conflict.new_value} (by {conflict.new_author})")
            print(f"  Severity: {conflict.severity}")

            print(f"\n{Fore.CYAN}Suggestions:")
            for j, suggestion in enumerate(conflict.suggestions, 1):
                print(f"  {j}. {suggestion}")

        print(f"\n{Style.BRIGHT}How would you like to proceed?")
        print("1. Keep original values")
        print("2. Use new values")
        print("3. Merge/combine values")
        print("4. Discuss with team")
        print("5. Skip for now")

        try:
            choice = input(f"\n{Fore.GREEN}Choice (1-5): ").strip()
            if choice == '1':
                self.log("Keeping original values", "INFO")
                return True
            elif choice == '2':
                self.log("Using new values", "INFO")
                return True
            elif choice == '3':
                self.log("Values will need manual merging", "INFO")
                return False
            elif choice == '4':
                self.log("Conflicts marked for team discussion", "INFO")
                return False
            else:
                self.log("Conflicts skipped", "WARN")
                return False
        except KeyboardInterrupt:
            self.log("Conflict resolution interrupted", "WARN")
            return False

    def _update_project_context(self, project: ProjectContext, insights: Dict):
        """Update project context with extracted insights"""
        if not insights:
            return

        # Update goals
        if 'goals' in insights and insights['goals']:
            project.goals = insights['goals']

        # Update requirements
        if 'requirements' in insights:
            new_requirements = insights['requirements']
            if isinstance(new_requirements, list):
                for req in new_requirements:
                    if req and req not in project.requirements:
                        project.requirements.append(req)
            elif new_requirements and new_requirements not in project.requirements:
                project.requirements.append(new_requirements)

        # Update tech stack
        if 'tech_stack' in insights:
            new_tech = insights['tech_stack']
            if isinstance(new_tech, list):
                for tech in new_tech:
                    if tech and tech not in project.tech_stack:
                        project.tech_stack.append(tech)
            elif new_tech and new_tech not in project.tech_stack:
                project.tech_stack.append(new_tech)

        # Update constraints
        if 'constraints' in insights:
            new_constraints = insights['constraints']
            if isinstance(new_constraints, list):
                for constraint in new_constraints:
                    if constraint and constraint not in project.constraints:
                        project.constraints.append(constraint)
            elif new_constraints and new_constraints not in project.constraints:
                project.constraints.append(new_constraints)

        # Update other fields
        if 'deployment_target' in insights and insights['deployment_target']:
            project.deployment_target = insights['deployment_target']

        if 'language_preferences' in insights and insights['language_preferences']:
            project.language_preferences = insights['language_preferences']

        self.log(f"Updated project context for '{project.name}'")

    def _advance_phase(self, request: Dict) -> Dict:
        """Advance project to next phase"""
        project = request.get('project')

        phases = ['discovery', 'analysis', 'design', 'implementation']
        current_index = phases.index(project.phase) if project.phase in phases else 0

        if current_index < len(phases) - 1:
            project.phase = phases[current_index + 1]
            project.updated_at = datetime.datetime.now()
            self.log(f"Advanced project '{project.name}' to {project.phase} phase")
            return {'status': 'success', 'new_phase': project.phase}
        else:
            return {'status': 'info', 'message': 'Project is already in final phase'}


class ContextAnalyzerAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("ContextAnalyzer", orchestrator)

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        if action == 'get_context_summary':
            project = request.get('project')
            return {'status': 'success', 'summary': self.get_context_summary(project)}
        elif action == 'analyze_completeness':
            project = request.get('project')
            return {'status': 'success', 'completeness': self.analyze_completeness(project)}
        elif action == 'suggest_next_steps':
            project = request.get('project')
            return {'status': 'success', 'next_steps': self.suggest_next_steps(project)}

        return {'status': 'error', 'message': 'Unknown action'}

    def get_context_summary(self, project: ProjectContext) -> str:
        """Generate a comprehensive context summary for the project"""
        summary_parts = []

        # Basic project info
        summary_parts.append(f"Project: {project.name}")
        summary_parts.append(f"Owner: {project.owner}")
        summary_parts.append(f"Phase: {project.phase}")

        # Goals and requirements
        if project.goals:
            summary_parts.append(f"Goals: {project.goals}")

        if project.requirements:
            summary_parts.append(f"Requirements: {', '.join(project.requirements[:3])}")

        # Technical details
        if project.tech_stack:
            summary_parts.append(f"Tech Stack: {', '.join(project.tech_stack[:3])}")

        if project.constraints:
            summary_parts.append(f"Constraints: {', '.join(project.constraints[:2])}")

        # Recent conversation context
        if project.conversation_history:
            recent_messages = project.conversation_history[-3:]
            context_msgs = []
            for msg in recent_messages:
                if msg['type'] == 'user':
                    context_msgs.append(msg['content'][:100])
            if context_msgs:
                summary_parts.append(f"Recent discussion: {' | '.join(context_msgs)}")

        return ' | '.join(summary_parts)

    def analyze_completeness(self, project: ProjectContext) -> Dict[str, float]:
        """Analyze how complete each aspect of the project is"""
        completeness = {}

        # Goals completeness (0-1)
        completeness['goals'] = 1.0 if project.goals and len(project.goals) > 20 else 0.0

        # Requirements completeness (0-1)
        completeness['requirements'] = min(len(project.requirements) / 5.0, 1.0)

        # Tech stack completeness (0-1)
        completeness['tech_stack'] = min(len(project.tech_stack) / 3.0, 1.0)

        # Constraints completeness (0-1)
        completeness['constraints'] = min(len(project.constraints) / 3.0, 1.0)

        # Conversation depth (0-1)
        user_messages = [msg for msg in project.conversation_history if msg['type'] == 'user']
        completeness['discussion'] = min(len(user_messages) / 10.0, 1.0)

        # Overall completeness
        completeness['overall'] = sum(completeness.values()) / len(completeness)

        return completeness

    def suggest_next_steps(self, project: ProjectContext) -> List[str]:
        """Suggest next steps based on project state"""
        completeness = self.analyze_completeness(project)
        suggestions = []

        if completeness['goals'] < 0.5:
            suggestions.append("Define clearer project goals and objectives")

        if completeness['requirements'] < 0.7:
            suggestions.append("Gather more detailed requirements")

        if completeness['tech_stack'] < 0.5:
            suggestions.append("Research and select appropriate technologies")

        if completeness['constraints'] < 0.5:
            suggestions.append("Identify project constraints and limitations")

        if completeness['discussion'] < 0.3:
            suggestions.append("Continue exploring the problem space")

        # Phase-specific suggestions
        phase_suggestions = {
            'discovery': [
                "Interview potential users or stakeholders",
                "Research existing solutions and competitors",
                "Define success metrics and KPIs"
            ],
            'analysis': [
                "Create technical specifications",
                "Plan testing and validation strategy",
                "Identify potential risks and mitigation plans"
            ],
            'design': [
                "Create system architecture diagrams",
                "Design database schema and API endpoints",
                "Plan development workflow and coding standards"
            ],
            'implementation': [
                "Set up development environment",
                "Create project repository and documentation",
                "Plan deployment and monitoring strategy"
            ]
        }

        if project.phase in phase_suggestions:
            suggestions.extend(phase_suggestions[project.phase][:2])

        return suggestions[:5]  # Limit to 5 suggestions


# Vector Database Handler
class VectorDatabaseHandler:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.collection_name = "socratic_knowledge"

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=os.path.join(data_dir, "chroma_db"),
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(self.collection_name)

        logger.info("Vector database initialized successfully")

    def add_knowledge(self, entries: List[KnowledgeEntry]):
        """Add knowledge entries to vector database"""
        if not self.embedding_model:
            logger.error("Embedding model not available")
            return False

        try:
            texts = [entry.content for entry in entries]
            embeddings = self.embedding_model.encode(texts).tolist()

            ids = [entry.id for entry in entries]
            metadatas = [entry.metadata for entry in entries]

            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(entries)} knowledge entries to vector database")
            return True

        except Exception as e:
            logger.error(f"Failed to add knowledge entries: {e}")
            return False

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar content in vector database"""
        if not self.embedding_model:
            logger.warning("Embedding model not available for search")
            return []

        try:
            query_embedding = self.embedding_model.encode([query]).tolist()

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'][0] else 0.0
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search vector database: {e}")
            return []

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry from vector database"""
        try:
            self.collection.delete(ids=[entry_id])
            logger.info(f"Deleted entry {entry_id} from vector database")
            return True
        except Exception as e:
            logger.error(f"Failed to delete entry {entry_id}: {e}")
            return False

    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector database collection"""
        try:
            count = self.collection.count()
            return {
                'total_entries': count,
                'collection_name': self.collection_name,
                'embedding_model': CONFIG['EMBEDDING_MODEL']
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'total_entries': 0, 'error': str(e)}


# Database Handler
class DatabaseHandler:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.db_path = os.path.join(data_dir, 'socratic_system.db')
        self.connection_pool = DatabaseConnectionPool(self.db_path)

        self._init_database()
        logger.info("Database handler initialized successfully")

    def _init_database(self):
        """Initialize database tables"""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    passcode_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    projects TEXT DEFAULT '[]'
                )
            ''')

            # Projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (owner) REFERENCES users (username)
                )
            ''')

            # Token usage tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    username TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_estimate REAL,
                    timestamp TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id),
                    FOREIGN KEY (username) REFERENCES users (username)
                )
            ''')

            # Knowledge entries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            ''')

            conn.commit()

    def save_user(self, user: User) -> bool:
        """Save user to database"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO users (username, passcode_hash, created_at, projects)
                    VALUES (?, ?, ?, ?)
                ''', (user.username, user.passcode_hash, user.created_at.isoformat(), json.dumps(user.projects)))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save user {user.username}: {e}")
            return False

    def load_user(self, username: str) -> Optional[User]:
        """Load user from database"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
                row = cursor.fetchone()

                if row:
                    return User(
                        username=row['username'],
                        passcode_hash=row['passcode_hash'],
                        created_at=datetime.datetime.fromisoformat(row['created_at']),
                        projects=json.loads(row['projects'])
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to load user {username}: {e}")
            return None

    def user_exists(self, username: str) -> bool:
        """Check if user exists"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM users WHERE username = ?', (username,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check user existence {username}: {e}")
            return False

    def save_project(self, project: ProjectContext) -> bool:
        """Save project to database"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                project_data = json.dumps(asdict(project), default=str)

                cursor.execute('''
                    INSERT OR REPLACE INTO projects (project_id, name, owner, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    project.project_id,
                    project.name,
                    project.owner,
                    project_data,
                    project.created_at.isoformat(),
                    project.updated_at.isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save project {project.project_id}: {e}")
            return False

    def load_project(self, project_id: str) -> Optional[ProjectContext]:
        """Load project from database"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT data FROM projects WHERE project_id = ?', (project_id,))
                row = cursor.fetchone()

                if row:
                    project_data = json.loads(row['data'])
                    # Convert datetime strings back to datetime objects
                    project_data['created_at'] = datetime.datetime.fromisoformat(project_data['created_at'])
                    project_data['updated_at'] = datetime.datetime.fromisoformat(project_data['updated_at'])
                    return ProjectContext(**project_data)
                return None
        except Exception as e:
            logger.error(f"Failed to load project {project_id}: {e}")
            return None

    def get_user_projects(self, username: str) -> List[Dict]:
        """Get all projects for a user"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT project_id, name, created_at, updated_at 
                    FROM projects 
                    WHERE owner = ? OR json_extract(data, '$.collaborators') LIKE ?
                    ORDER BY updated_at DESC
                ''', (username, f'%{username}%'))

                projects = []
                for row in cursor.fetchall():
                    projects.append({
                        'project_id': row['project_id'],
                        'name': row['name'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    })
                return projects
        except Exception as e:
            logger.error(f"Failed to get projects for user {username}: {e}")
            return []

    def save_token_usage(self, usage: TokenUsage, project_id: str = None, username: str = None) -> bool:
        """Save token usage statistics"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO token_usage (project_id, username, input_tokens, output_tokens, 
                                           total_tokens, cost_estimate, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    project_id, username, usage.input_tokens, usage.output_tokens,
                    usage.total_tokens, usage.cost_estimate, usage.timestamp.isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save token usage: {e}")
            return False

    def get_token_usage_stats(self, username: str = None, days: int = 30) -> Dict:
        """Get token usage statistics"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                since_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()

                if username:
                    cursor.execute('''
                        SELECT SUM(input_tokens) as total_input, SUM(output_tokens) as total_output,
                               SUM(total_tokens) as total_all, SUM(cost_estimate) as total_cost,
                               COUNT(*) as request_count
                        FROM token_usage 
                        WHERE username = ? AND timestamp > ?
                    ''', (username, since_date))
                else:
                    cursor.execute('''
                        SELECT SUM(input_tokens) as total_input, SUM(output_tokens) as total_output,
                               SUM(total_tokens) as total_all, SUM(cost_estimate) as total_cost,
                               COUNT(*) as request_count
                        FROM token_usage 
                        WHERE timestamp > ?
                    ''', (since_date,))

                row = cursor.fetchone()
                if row:
                    return {
                        'total_input_tokens': row['total_input'] or 0,
                        'total_output_tokens': row['total_output'] or 0,
                        'total_tokens': row['total_all'] or 0,
                        'estimated_cost': row['total_cost'] or 0.0,
                        'request_count': row['request_count'] or 0,
                        'period_days': days
                    }
                return {'total_tokens': 0, 'estimated_cost':0.0, 'request_count': 0, 'period_days': days}
        except Exception as e:
            logger.error(f"Failed to get token usage stats: {e}")
            return {'total_tokens': 0, 'estimated_cost': 0.0, 'request_count': 0, 'period_days': days}

    def save_knowledge_entry(self, entry: KnowledgeEntry) -> bool:
        """Save knowledge entry to database"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO knowledge_entries (id, content, category, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    entry.id, entry.content, entry.category,
                    json.dumps(entry.metadata), datetime.datetime.now().isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save knowledge entry {entry.id}: {e}")
            return False

    def load_knowledge_entries(self, category: str = None) -> List[KnowledgeEntry]:
        """Load knowledge entries from database"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                if category:
                    cursor.execute('SELECT * FROM knowledge_entries WHERE category = ?', (category,))
                else:
                    cursor.execute('SELECT * FROM knowledge_entries')

                entries = []
                for row in cursor.fetchall():
                    entries.append(KnowledgeEntry(
                        id=row['id'],
                        content=row['content'],
                        category=row['category'],
                        metadata=json.loads(row['metadata'])
                    ))
                return entries
        except Exception as e:
            logger.error(f"Failed to load knowledge entries: {e}")
            return []


class ClaudeClient:
    def __init__(self):
        api_key = os.getenv('API_KEY_CLAUDE')
        if not api_key:
            logger.error("API_KEY_CLAUDE not found in environment variables")
            raise ValueError("Missing Anthropic API key")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = CONFIG['CLAUDE_MODEL']
        logger.info(f"Claude client initialized with model: {self.model}")

    def generate_socratic_question(self, prompt: str) -> str:
        """Generate a Socratic question using Claude"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            return message.content[0].text.strip()

        except Exception as e:
            logger.error(f"Failed to generate Socratic question: {e}")
            raise

    def extract_insights(self, user_response: str, project: ProjectContext) -> Dict:
        """Extract insights from user response"""
        prompt = f"""Analyze this user response and extract structured insights for a software project.

Project Context:
- Name: {project.name}
- Phase: {project.phase}
- Current Goals: {project.goals or 'Not defined'}
- Current Tech Stack: {', '.join(project.tech_stack) if project.tech_stack else 'Not defined'}

User Response: "{user_response}"

Extract the following if mentioned (return empty if not found):
1. goals: Main project objectives
2. requirements: Functional or non-functional requirements
3. tech_stack: Technologies, frameworks, tools mentioned
4. constraints: Limitations, restrictions, or constraints
5. deployment_target: Where the project will be deployed
6. language_preferences: Programming languages mentioned

Return as JSON only:"""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            response_text = message.content[0].text.strip()

            # Try to parse JSON response
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: extract JSON from response if wrapped in text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    logger.warning("Failed to parse insights JSON, returning empty dict")
                    return {}

        except Exception as e:
            logger.error(f"Failed to extract insights: {e}")
            return {}


# Agent Orchestrator - The main coordinator
class AgentOrchestrator:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or CONFIG['DATA_DIR']
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize core components
        self.database = DatabaseHandler(self.data_dir)
        self.vector_db = VectorDatabaseHandler(self.data_dir)
        self.claude_client = ClaudeClient()

        # Initialize agents
        self.session_manager = SessionManagerAgent(self)
        self.conversation_engine = ConversationEngineAgent(self)
        self.context_analyzer = ContextAnalyzerAgent(self)

        # Current session state
        self.current_user = None
        self.current_project = None

        logger.info("Agent orchestrator initialized successfully")

    def authenticate_user(self, username: str, passcode: str) -> bool:
        """Authenticate user and set as current user"""
        result = self.session_manager.process({
            'action': 'authenticate_user',
            'username': username,
            'passcode': passcode
        })

        if result['status'] == 'success':
            self.current_user = result['user']
            return True
        return False

    def create_user(self, username: str, passcode: str) -> bool:
        """Create new user account"""
        result = self.session_manager.process({
            'action': 'create_user',
            'username': username,
            'passcode': passcode
        })

        if result['status'] == 'success':
            self.current_user = result['user']
            return True
        return False

    def create_project(self, project_name: str) -> bool:
        """Create new project for current user"""
        if not self.current_user:
            return False

        result = self.session_manager.process({
            'action': 'create_project',
            'project_name': project_name,
            'owner': self.current_user.username
        })

        if result['status'] == 'success':
            self.current_project = result['project']
            return True
        return False

    def load_project(self, project_id: str) -> bool:
        """Load existing project"""
        result = self.session_manager.process({
            'action': 'load_project',
            'project_id': project_id
        })

        if result['status'] == 'success':
            self.current_project = result['project']
            return True
        return False

    def get_user_projects(self) -> List[Dict]:
        """Get all projects for current user"""
        if not self.current_user:
            return []

        result = self.session_manager.process({
            'action': 'list_projects',
            'username': self.current_user.username
        })

        return result.get('projects', []) if result['status'] == 'success' else []

    def start_socratic_session(self) -> str:
        """Start a Socratic questioning session"""
        if not self.current_project:
            return "No project selected. Please create or load a project first."

        result = self.conversation_engine.process({
            'action': 'generate_question',
            'project': self.current_project
        })

        if result['status'] == 'success':
            return result['question']
        else:
            return "Sorry, I couldn't generate a question right now. Please try again."

    def process_user_response(self, response: str) -> Dict:
        """Process user response and return insights"""
        if not self.current_project or not self.current_user:
            return {'status': 'error', 'message': 'No active session'}

        # Process the response
        result = self.conversation_engine.process({
            'action': 'process_response',
            'project': self.current_project,
            'response': response,
            'current_user': self.current_user.username
        })

        # Save project after processing
        if result['status'] == 'success':
            self.session_manager.process({
                'action': 'save_project',
                'project': self.current_project
            })

        return result

    def advance_project_phase(self) -> str:
        """Advance project to next phase"""
        if not self.current_project:
            return "No project selected."

        result = self.conversation_engine.process({
            'action': 'advance_phase',
            'project': self.current_project
        })

        if result['status'] == 'success':
            self.session_manager.process({
                'action': 'save_project',
                'project': self.current_project
            })
            return f"Project advanced to {result['new_phase']} phase!"
        else:
            return result.get('message', 'Could not advance phase.')

    def get_project_status(self) -> Dict:
        """Get current project status and completeness"""
        if not self.current_project:
            return {'status': 'error', 'message': 'No project selected'}

        completeness = self.context_analyzer.analyze_completeness(self.current_project)
        next_steps = self.context_analyzer.suggest_next_steps(self.current_project)

        return {
            'status': 'success',
            'project_name': self.current_project.name,
            'phase': self.current_project.phase,
            'completeness': completeness,
            'next_steps': next_steps,
            'conversation_count': len([msg for msg in self.current_project.conversation_history
                                       if msg['type'] == 'user'])
        }

    def save_session_state(self):
        """Save current session state"""
        self.session_manager.process({
            'action': 'save_session_state',
            'current_user': self.current_user.username if self.current_user else None,
            'current_project_id': self.current_project.project_id if self.current_project else None
        })

    def restore_session_state(self) -> bool:
        """Restore previous session state"""
        result = self.session_manager.process({
            'action': 'restore_session_state'
        })

        if result['status'] == 'success':
            session_data = result['session_data']

            # Restore user
            if session_data.get('current_user'):
                user = self.database.load_user(session_data['current_user'])
                if user:
                    self.current_user = user

            # Restore project
            if session_data.get('current_project_id'):
                if self.load_project(session_data['current_project_id']):
                    return True

        return False


# Main CLI Interface
class SocraticSystemCLI:
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.running = True

        # Try to restore previous session
        self.orchestrator.restore_session_state()

    def display_banner(self):
        """Display system banner"""
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{'=' * 60}")
        print(f"🧠 SOCRATIC SOFTWARE DEVELOPMENT SYSTEM v7.1 🧠")
        print(f"{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Intelligent project guidance through Socratic questioning{Style.RESET_ALL}\n")

    def display_menu(self):
        """Display main menu"""
        print(f"\n{Style.BRIGHT}{Fore.YELLOW}Main Menu:")
        print(f"{Fore.WHITE}1. 👤 User Management")
        print(f"2. 📁 Project Management")
        print(f"3. 🤔 Start Socratic Session")
        print(f"4. 📊 Project Status")
        print(f"5. ⚙️  System Settings")
        print(f"6. 🚪 Exit")

        if self.orchestrator.current_user:
            print(f"\n{Fore.GREEN}Current User: {self.orchestrator.current_user.username}")
        if self.orchestrator.current_project:
            print(f"{Fore.BLUE}Current Project: {self.orchestrator.current_project.name} "
                  f"({self.orchestrator.current_project.phase})")

    def handle_user_management(self):
        """Handle user authentication and management"""
        while True:
            print(f"\n{Style.BRIGHT}{Fore.YELLOW}User Management:")
            print(f"{Fore.WHITE}1. Login")
            print(f"2. Create New Account")
            print(f"3. Back to Main Menu")

            choice = input(f"\n{Fore.GREEN}Choice (1-3): ").strip()

            if choice == '1':
                self.handle_login()
                break
            elif choice == '2':
                self.handle_create_account()
                break
            elif choice == '3':
                break
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.")

    def handle_login(self):
        """Handle user login"""
        print(f"\n{Style.BRIGHT}{Fore.CYAN}User Login")
        username = input("Username: ").strip()
        if not username:
            print(f"{Fore.RED}Username cannot be empty.")
            return

        passcode = input(f"{Fore.WHITE}Passcode: ").strip()
        if not passcode:
            print(f"{Fore.RED}Passcode cannot be empty.")
            return

        if self.orchestrator.authenticate_user(username, passcode):
            print(f"{Fore.GREEN}✓ Login successful! Welcome, {username}!")
        else:
            print(f"{Fore.RED}✗ Login failed. Please check your credentials.")

    def handle_create_account(self):
        """Handle new account creation"""
        print(f"\n{Style.BRIGHT}{Fore.CYAN}Create New Account")
        username = input("Username: ").strip()
        print(1)
        if not username:
            print(2)
            print(f"{Fore.RED}Username cannot be empty.")
            return
        print(3)
        passcode = input(f"{Fore.WHITE}Passcode: ").strip()
        print(4)
        if not passcode:
            print(f"{Fore.RED}Passcode cannot be empty.")
            return

        confirm_passcode = input(f"{Fore.WHITE}Passcode: ").strip()
        if passcode != confirm_passcode:
            print(f"{Fore.RED}Passcodes do not match.")
            return

        if self.orchestrator.create_user(username, passcode):
            print(f"{Fore.GREEN}✓ Account created successfully! Welcome, {username}!")
        else:
            print(f"{Fore.RED}✗ Failed to create account. Username may already exist.")

    def handle_project_management(self):
        """Handle project management"""
        if not self.orchestrator.current_user:
            print(f"{Fore.RED}Please login first.")
            return

        while True:
            print(f"\n{Style.BRIGHT}{Fore.YELLOW}Project Management:")
            print(f"{Fore.WHITE}1. Create New Project")
            print(f"2. Load Existing Project")
            print(f"3. List My Projects")
            print(f"4. Back to Main Menu")

            choice = input(f"\n{Fore.GREEN}Choice (1-4): ").strip()

            if choice == '1':
                self.handle_create_project()
                break
            elif choice == '2':
                self.handle_load_project()
                break
            elif choice == '3':
                self.handle_list_projects()
            elif choice == '4':
                break
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.")

    def handle_create_project(self):
        """Handle project creation"""
        project_name = input(f"\n{Fore.CYAN}Project Name: ").strip()
        if not project_name:
            print(f"{Fore.RED}Project name cannot be empty.")
            return

        if self.orchestrator.create_project(project_name):
            print(f"{Fore.GREEN}✓ Project '{project_name}' created successfully!")
        else:
            print(f"{Fore.RED}✗ Failed to create project.")

    def handle_load_project(self):
        """Handle project loading"""
        projects = self.orchestrator.get_user_projects()
        if not projects:
            print(f"{Fore.YELLOW}No projects found.")
            return

        print(f"\n{Style.BRIGHT}{Fore.CYAN}Your Projects:")
        for i, project in enumerate(projects, 1):
            print(f"{i}. {project['name']} (Updated: {project['updated_at']})")

        try:
            choice = int(input(f"\n{Fore.GREEN}Select project (1-{len(projects)}): "))
            if 1 <= choice <= len(projects):
                project_id = projects[choice - 1]['project_id']
                if self.orchestrator.load_project(project_id):
                    print(f"{Fore.GREEN}✓ Project loaded successfully!")
                else:
                    print(f"{Fore.RED}✗ Failed to load project.")
            else:
                print(f"{Fore.RED}Invalid choice.")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number.")

    def handle_list_projects(self):
        """Handle project listing"""
        projects = self.orchestrator.get_user_projects()
        if not projects:
            print(f"{Fore.YELLOW}No projects found.")
            return

        print(f"\n{Style.BRIGHT}{Fore.CYAN}Your Projects:")
        for project in projects:
            print(f"• {project['name']}")
            print(f"  ID: {project['project_id']}")
            print(f"  Created: {project['created_at']}")
            print(f"  Updated: {project['updated_at']}\n")

    def handle_socratic_session(self):
        """Handle Socratic questioning session"""
        if not self.orchestrator.current_user:
            print(f"{Fore.RED}Please login first.")
            return

        if not self.orchestrator.current_project:
            print(f"{Fore.RED}Please create or load a project first.")
            return

        print(f"\n{Style.BRIGHT}{Fore.CYAN}🤔 Socratic Session Started")
        print(f"{Fore.WHITE}Project: {self.orchestrator.current_project.name}")
        print(f"Phase: {self.orchestrator.current_project.phase.title()}")
        print(f"\n{Fore.YELLOW}Commands: 'next' (advance phase), 'status' (project status), 'quit' (exit session)\n")

        while True:
            # Generate question
            question = self.orchestrator.start_socratic_session()
            print(f"{Fore.CYAN}🤔 {question}")

            # Get user response
            response = input(f"\n{Fore.GREEN}Your response: ").strip()

            if not response:
                continue

            # Handle special commands
            if response.lower() == 'quit':
                print(f"{Fore.YELLOW}Session ended.")
                break
            elif response.lower() == 'next':
                result = self.orchestrator.advance_project_phase()
                print(f"{Fore.BLUE}{result}")
                continue
            elif response.lower() == 'status':
                self.display_project_status()
                continue

            # Process response
            result = self.orchestrator.process_user_response(response)

            if result['status'] == 'success':
                if result.get('conflicts_pending'):
                    print(f"{Fore.YELLOW}⚠️ Conflicts detected and need resolution.")

                insights = result.get('insights', {})
                if insights:
                    print(f"\n{Fore.GREEN}📝 Insights captured:")
                    for key, value in insights.items():
                        if value:
                            if isinstance(value, list):
                                print(f"  • {key.title()}: {', '.join(value)}")
                            else:
                                print(f"  • {key.title()}: {value}")
            else:
                print(f"{Fore.RED}Error processing response: {result.get('message', 'Unknown error')}")

            print()

    def display_project_status(self):
        """Display current project status"""
        status = self.orchestrator.get_project_status()

        if status['status'] == 'error':
            print(f"{Fore.RED}{status['message']}")
            return

        print(f"\n{Style.BRIGHT}{Fore.CYAN}📊 Project Status")
        print(f"{Fore.WHITE}Project: {status['project_name']}")
        print(f"Phase: {status['phase'].title()}")
        print(f"Conversations: {status['conversation_count']}")

        print(f"\n{Style.BRIGHT}Completeness:")
        for aspect, score in status['completeness'].items():
            if aspect == 'overall':
                continue
            percentage = int(score * 100)
            bar = '█' * (percentage // 10) + '░' * (10 - percentage // 10)
            color = Fore.GREEN if score > 0.7 else Fore.YELLOW if score > 0.3 else Fore.RED
            print(f"{color}{aspect.title()}: {bar} {percentage}%")

        overall = int(status['completeness']['overall'] * 100)
        print(f"\n{Style.BRIGHT}{Fore.CYAN}Overall: {overall}%")

        if status['next_steps']:
            print(f"\n{Style.BRIGHT}{Fore.YELLOW}Suggested Next Steps:")
            for i, step in enumerate(status['next_steps'], 1):
                print(f"  {i}. {step}")

    def handle_system_settings(self):
        """Handle system settings"""
        while True:
            print(f"\n{Style.BRIGHT}{Fore.YELLOW}System Settings:")
            print(
                f"{Fore.WHITE}1. Toggle Dynamic Questions (Currently: {'ON' if self.orchestrator.conversation_engine.use_dynamic_questions else 'OFF'})")
            print(f"2. View Token Usage")
            print(f"3. Vector Database Stats")
            print(f"4. Clear Cache")
            print(f"5. Back to Main Menu")

            choice = input(f"\n{Fore.GREEN}Choice (1-5): ").strip()

            if choice == '1':
                result = self.orchestrator.conversation_engine.process({'action': 'toggle_dynamic_questions'})
                mode = "ON" if result['dynamic_mode'] else "OFF"
                print(f"{Fore.BLUE}Dynamic questions mode: {mode}")

            elif choice == '2':
                self.display_token_usage()

            elif choice == '3':
                self.display_vector_db_stats()

            elif choice == '4':
                self.orchestrator.session_manager.cache.clear()
                self.orchestrator.conversation_engine.response_cache.clear()
                print(f"{Fore.GREEN}✓ Cache cleared successfully!")

            elif choice == '5':
                break

            else:
                print(f"{Fore.RED}Invalid choice. Please try again.")

    def display_token_usage(self):
        """Display token usage statistics"""
        username = self.orchestrator.current_user.username if self.orchestrator.current_user else None
        stats = self.orchestrator.database.get_token_usage_stats(username)

        print(f"\n{Style.BRIGHT}{Fore.CYAN}📈 Token Usage (Last {stats['period_days']} days)")
        print(f"{Fore.WHITE}Total Requests: {stats['request_count']}")
        print(f"Input Tokens: {stats['total_input_tokens']:,}")
        print(f"Output Tokens: {stats['total_output_tokens']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Estimated Cost: ${stats['estimated_cost']:.4f}")

    def display_vector_db_stats(self):
        """Display vector database statistics"""
        stats = self.orchestrator.vector_db.get_collection_stats()

        print(f"\n{Style.BRIGHT}{Fore.CYAN}🗄️ Vector Database Stats")
        print(f"{Fore.WHITE}Total Entries: {stats.get('total_entries', 0):,}")
        print(f"Collection: {stats.get('collection_name', 'N/A')}")
        print(f"Embedding Model: {stats.get('embedding_model', 'N/A')}")

        if 'error' in stats:
            print(f"{Fore.RED}Error: {stats['error']}")

    def run(self):
        """Run the main CLI loop"""
        self.display_banner()

        try:
            while self.running:
                self.display_menu()
                choice = input(f"\n{Fore.GREEN}Choice (1-6): ").strip()

                if choice == '1':
                    self.handle_user_management()
                elif choice == '2':
                    self.handle_project_management()
                elif choice == '3':
                    self.handle_socratic_session()
                elif choice == '4':
                    self.display_project_status()
                elif choice == '5':
                    self.handle_system_settings()
                elif choice == '6':
                    print(f"\n{Fore.CYAN}Saving session state...")
                    self.orchestrator.save_session_state()
                    print(f"{Fore.GREEN}✓ Thank you for using the Socratic System! Goodbye! 👋")
                    self.running = False
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Session interrupted. Saving state...")
            self.orchestrator.save_session_state()
            print(f"{Fore.GREEN}✓ State saved. Goodbye! 👋")
        except Exception as e:
            logger.error(f"Unexpected error in CLI: {e}")
            print(f"\n{Fore.RED}An unexpected error occurred. Please check the logs.")
        finally:
            self.orchestrator.save_session_state()


# Main execution
if __name__ == "__main__":
    # Check for required environment variable
    if not os.getenv('API_KEY_CLAUDE'):
        print(f"{Fore.RED}Error: API_KEY_CLAUDE environment variable not set.")
        print(f"{Fore.YELLOW}Please set your Anthropic API key:")
        print(f"{Fore.WHITE}export API_KEY_CLAUDE='your-api-key-here'")
        exit(1)

    # Initialize and run the CLI
    try:
        cli = SocraticSystemCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Fatal error starting system: {e}")
        print(f"{Fore.RED}Failed to start Socratic System. Check logs for details.")
        exit(1)
