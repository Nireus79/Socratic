#!/usr/bin/env python3

import os
import json
import hashlib
import getpass
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
                return