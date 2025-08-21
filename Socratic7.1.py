#!/usr/bin/env python3
# Fixed version addressing the 'list' object has no attribute 'lower' error
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

    # FIXED: Conflict detection methods with proper type handling
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

        # FIXED: Ensure new_tech is always a list of strings
        if not isinstance(new_tech, list):
            new_tech = [new_tech] if new_tech else []

        # Filter out non-string items and convert to strings
        new_tech = [str(tech).strip() for tech in new_tech if tech and str(tech).strip()]

        for new_item in new_tech:
            if not new_item:
                continue

            new_item_lower = new_item.lower()
            for existing_item in project.tech_stack:
                if not existing_item:
                    continue

                existing_lower = str(existing_item).lower()  # Ensure it's a string
                conflict_category = self._find_conflict_category(new_item_lower, existing_lower)

                if conflict_category:
                    conflict_id = str(uuid.uuid4())
                    conflicts.append(ConflictInfo(
                        conflict_id=conflict_id,
                        conflict_type='tech_stack_incompatible',
                        old_value=existing_item,
                        new_value=new_item,
                        old_author='unknown',  # Would need to track in real implementation
                        new_author=current_user,
                        old_timestamp='unknown',
                        new_timestamp=datetime.datetime.now().isoformat(),
                        severity='medium',
                        suggestions=[
                            f"Consider choosing between {existing_item} and {new_item}",
                            f"Evaluate the benefits of each {conflict_category} option",
                            "Discuss with team members before making the final decision"
                        ]
                    ))

        return conflicts

    def _check_requirements_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        conflicts = []
        new_requirements = new_insights.get('requirements', [])

        if not isinstance(new_requirements, list):
            new_requirements = [new_requirements] if new_requirements else []

        new_requirements = [str(req).strip() for req in new_requirements if req and str(req).strip()]

        for new_req in new_requirements:
            new_req_lower = new_req.lower()
            for existing_req in project.requirements:
                existing_req_lower = str(existing_req).lower()

                # Check for contradictory requirements
                if self._are_contradictory_requirements(new_req_lower, existing_req_lower):
                    conflict_id = str(uuid.uuid4())
                    conflicts.append(ConflictInfo(
                        conflict_id=conflict_id,
                        conflict_type='requirements_contradictory',
                        old_value=existing_req,
                        new_value=new_req,
                        old_author='unknown',
                        new_author=current_user,
                        old_timestamp='unknown',
                        new_timestamp=datetime.datetime.now().isoformat(),
                        severity='high',
                        suggestions=[
                            "Review both requirements for compatibility",
                            "Consider if both can be satisfied simultaneously",
                            "Prioritize requirements based on business value"
                        ]
                    ))

        return conflicts

    def _check_goals_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        conflicts = []
        new_goals = new_insights.get('goals', '')

        if new_goals and project.goals:
            if self._are_conflicting_goals(project.goals.lower(), str(new_goals).lower()):
                conflict_id = str(uuid.uuid4())
                conflicts.append(ConflictInfo(
                    conflict_id=conflict_id,
                    conflict_type='goals_misaligned',
                    old_value=project.goals,
                    new_value=str(new_goals),
                    old_author='unknown',
                    new_author=current_user,
                    old_timestamp='unknown',
                    new_timestamp=datetime.datetime.now().isoformat(),
                    severity='high',
                    suggestions=[
                        "Clarify the primary project objective",
                        "Ensure all team members share the same vision",
                        "Consider breaking into separate projects if goals are too different"
                    ]
                ))

        return conflicts

    def _check_constraints_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        conflicts = []
        new_constraints = new_insights.get('constraints', [])

        if not isinstance(new_constraints, list):
            new_constraints = [new_constraints] if new_constraints else []

        new_constraints = [str(const).strip() for const in new_constraints if const and str(const).strip()]

        for new_const in new_constraints:
            new_const_lower = new_const.lower()
            for existing_const in project.constraints:
                existing_const_lower = str(existing_const).lower()

                if self._are_contradictory_constraints(new_const_lower, existing_const_lower):
                    conflict_id = str(uuid.uuid4())
                    conflicts.append(ConflictInfo(
                        conflict_id=conflict_id,
                        conflict_type='constraints_contradictory',
                        old_value=existing_const,
                        new_value=new_const,
                        old_author='unknown',
                        new_author=current_user,
                        old_timestamp='unknown',
                        new_timestamp=datetime.datetime.now().isoformat(),
                        severity='medium',
                        suggestions=[
                            "Evaluate if both constraints can be satisfied",
                            "Consider constraint priority and flexibility",
                            "Discuss trade-offs with stakeholders"
                        ]
                    ))

        return conflicts

    def _find_conflict_category(self, item1: str, item2: str) -> Optional[str]:
        """Find if two items belong to the same category and might conflict"""
        for category, items in self.conflict_rules.items():
            if item1 in items and item2 in items and item1 != item2:
                return category
        return None

    def _are_contradictory_requirements(self, req1: str, req2: str) -> bool:
        """Check if two requirements contradict each other"""
        contradictions = [
            ('real-time', 'batch'),
            ('synchronous', 'asynchronous'),
            ('high performance', 'low resource'),
            ('simple', 'feature-rich'),
            ('secure', 'open access'),
        ]

        for term1, term2 in contradictions:
            if (term1 in req1 and term2 in req2) or (term2 in req1 and term1 in req2):
                return True
        return False

    def _are_conflicting_goals(self, goal1: str, goal2: str) -> bool:
        """Check if two goals conflict"""
        conflicting_keywords = [
            ('profit', 'free'),
            ('simple', 'comprehensive'),
            ('fast development', 'perfect quality'),
            ('minimal', 'feature-complete'),
        ]

        for term1, term2 in conflicting_keywords:
            if (term1 in goal1 and term2 in goal2) or (term2 in goal1 and term1 in goal2):
                return True
        return False

    def _are_contradictory_constraints(self, const1: str, const2: str) -> bool:
        """Check if two constraints contradict each other"""
        contradictions = [
            ('no budget', 'enterprise features'),
            ('no external dependencies', 'use cloud services'),
            ('offline', 'real-time sync'),
            ('minimal ui', 'rich interface'),
        ]

        for term1, term2 in contradictions:
            if (term1 in const1 and term2 in const2) or (term2 in const1 and term1 in const2):
                return True
        return False

    def _handle_conflicts_realtime(self, conflicts: List[ConflictInfo], project: ProjectContext) -> bool:
        """Handle conflicts in real-time - simplified version"""
        if not conflicts:
            return True

        self.log(f"Detected {len(conflicts)} conflicts", "WARN")
        for conflict in conflicts:
            self.log(f"Conflict: {conflict.conflict_type} - {conflict.old_value} vs {conflict.new_value}", "WARN")

        # For now, just log conflicts. In a full implementation, this would
        # prompt users for resolution or apply automatic resolution rules
        return False  # Indicates conflicts need manual resolution

    def _update_project_context(self, project: ProjectContext, insights: Dict):
        """Update project context with extracted insights"""
        if not insights:
            return

        # Update goals
        if 'goals' in insights and insights['goals']:
            if not project.goals:
                project.goals = str(insights['goals'])
            else:
                # Append or merge goals
                new_goals = str(insights['goals'])
                if new_goals not in project.goals:
                    project.goals += f" {new_goals}"

        # Update tech stack
        if 'tech_stack' in insights:
            tech_items = insights['tech_stack']
            if not isinstance(tech_items, list):
                tech_items = [tech_items] if tech_items else []

            for item in tech_items:
                if item and str(item).strip() not in project.tech_stack:
                    project.tech_stack.append(str(item).strip())

        # Update requirements
        if 'requirements' in insights:
            req_items = insights['requirements']
            if not isinstance(req_items, list):
                req_items = [req_items] if req_items else []

            for item in req_items:
                if item and str(item).strip() not in project.requirements:
                    project.requirements.append(str(item).strip())

        # Update constraints
        if 'constraints' in insights:
            const_items = insights['constraints']
            if not isinstance(const_items, list):
                const_items = [const_items] if const_items else []

            for item in const_items:
                if item and str(item).strip() not in project.constraints:
                    project.constraints.append(str(item).strip())

    def _advance_phase(self, request: Dict) -> Dict:
        project = request.get('project')

        phases = ['discovery', 'analysis', 'design', 'implementation']
        current_index = phases.index(project.phase) if project.phase in phases else 0

        if current_index < len(phases) - 1:
            project.phase = phases[current_index + 1]
            self.log(f"Advanced to {project.phase} phase")
            return {'status': 'success', 'new_phase': project.phase}
        else:
            return {'status': 'error', 'message': 'Already in final phase'}


class DatabaseManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, 'socratic_system.db')
        self.pool = DatabaseConnectionPool(self.db_path, pool_size=5)
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        os.makedirs(self.data_dir, exist_ok=True)

        with self.pool.get_connection() as conn:
            # Users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    passcode_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    projects TEXT DEFAULT '[]'
                )
            ''')

            # Projects table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')

            # Knowledge base table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')

            conn.commit()

    def save_user(self, user: User):
        """Save user to database"""
        with self.pool.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO users (username, passcode_hash, created_at, projects)
                VALUES (?, ?, ?, ?)
            ''', (user.username, user.passcode_hash, user.created_at.isoformat(),
                  json.dumps(user.projects)))
            conn.commit()

    def load_user(self, username: str) -> Optional[User]:
        """Load user from database"""
        with self.pool.get_connection() as conn:
            row = conn.execute('''
                SELECT username, passcode_hash, created_at, projects
                FROM users WHERE username = ?
            ''', (username,)).fetchone()

            if row:
                return User(
                    username=row['username'],
                    passcode_hash=row['passcode_hash'],
                    created_at=datetime.datetime.fromisoformat(row['created_at']),
                    projects=json.loads(row['projects'] or '[]')
                )
            return None

    def user_exists(self, username: str) -> bool:
        """Check if user exists"""
        with self.pool.get_connection() as conn:
            result = conn.execute('SELECT 1 FROM users WHERE username = ?', (username,)).fetchone()
            return result is not None

    def save_project(self, project: ProjectContext):
        """Save project to database"""
        with self.pool.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO projects (project_id, name, owner, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (project.project_id, project.name, project.owner,
                  json.dumps(asdict(project)),
                  project.created_at.isoformat(),
                  project.updated_at.isoformat()))
            conn.commit()

    def load_project(self, project_id: str) -> Optional[ProjectContext]:
        """Load project from database"""
        with self.pool.get_connection() as conn:
            row = conn.execute('''
                SELECT data FROM projects WHERE project_id = ?
            ''', (project_id,)).fetchone()

            if row:
                data = json.loads(row['data'])
                data['created_at'] = datetime.datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.datetime.fromisoformat(data['updated_at'])
                return ProjectContext(**data)
            return None

    def get_user_projects(self, username: str) -> List[Dict]:
        """Get all projects for a user"""
        with self.pool.get_connection() as conn:
            rows = conn.execute('''
                SELECT project_id, name, created_at, updated_at
                FROM projects WHERE owner = ? OR data LIKE ?
            ''', (username, f'%"collaborators":%"{username}"%')).fetchall()

            projects = []
            for row in rows:
                projects.append({
                    'project_id': row['project_id'],
                    'name': row['name'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                })
            return projects


# Vector Database for Semantic Search
class VectorDatabase:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.chroma_dir = os.path.join(data_dir, 'chroma_db')
        self.model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="socratic_knowledge",
            metadata={"description": "Knowledge base for Socratic system"}
        )

    def add_knowledge(self, entry: KnowledgeEntry):
        """Add knowledge entry to vector database"""
        if not entry.embedding:
            entry.embedding = self.model.encode(entry.content).tolist()

        self.collection.add(
            documents=[entry.content],
            metadatas=[entry.metadata],
            ids=[entry.id],
            embeddings=[entry.embedding]
        )

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar knowledge entries"""
        query_embedding = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        knowledge_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                knowledge_results.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0
                })

        return knowledge_results


# Context Analyzer
class ContextAnalyzer:
    def __init__(self, max_length: int = 8000):
        self.max_length = max_length

    def get_context_summary(self, project: ProjectContext) -> str:
        """Generate a context summary for the project"""
        context_parts = []

        if project.goals:
            context_parts.append(f"Goals: {project.goals}")

        if project.tech_stack:
            context_parts.append(f"Tech Stack: {', '.join(project.tech_stack[:5])}")

        if project.requirements:
            context_parts.append(f"Requirements: {', '.join(project.requirements[:3])}")

        if project.constraints:
            context_parts.append(f"Constraints: {', '.join(project.constraints[:3])}")

        context_parts.append(f"Phase: {project.phase}")
        context_parts.append(f"Team: {project.team_structure}")

        # Add recent conversation
        if project.conversation_history:
            recent_messages = project.conversation_history[-3:]
            context_parts.append("Recent discussion:")
            for msg in recent_messages:
                role = "Q" if msg['type'] == 'assistant' else "A"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                context_parts.append(f"{role}: {content}")

        full_context = " | ".join(context_parts)

        # Truncate if too long
        if len(full_context) > self.max_length:
            full_context = full_context[:self.max_length - 3] + "..."

        return full_context


# Claude API Client
class ClaudeClient:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = CONFIG['CLAUDE_MODEL']

    def generate_socratic_question(self, prompt: str) -> str:
        """Generate a Socratic question using Claude"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            raise

    def extract_insights(self, user_response: str, project: ProjectContext) -> Dict:
        """Extract insights from user response"""
        prompt = f"""Analyze this user response and extract structured insights:

Project Phase: {project.phase}
User Response: "{user_response}"

Extract and return ONLY a JSON object with these fields (use empty lists/strings if nothing found):
{{
    "goals": "extracted goals or empty string",
    "tech_stack": ["tech", "stack", "items"],
    "requirements": ["functional", "requirements"],
    "constraints": ["project", "constraints"],
    "concerns": ["user", "concerns"],
    "decisions": ["key", "decisions"]
}}

JSON only:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            response_text = response.content[0].text.strip()
            # Try to extract JSON from response
            if response_text.startswith('{') and response_text.endswith('}'):
                return json.loads(response_text)
            else:
                # Fallback parsing
                return self._fallback_insight_extraction(user_response)

        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return self._fallback_insight_extraction(user_response)

    def _fallback_insight_extraction(self, user_response: str) -> Dict:
        """Simple fallback insight extraction"""
        insights = {
            "goals": "",
            "tech_stack": [],
            "requirements": [],
            "constraints": [],
            "concerns": [],
            "decisions": []
        }

        response_lower = user_response.lower()

        # Simple keyword-based extraction
        tech_keywords = ['python', 'javascript', 'react', 'django', 'flask', 'mysql', 'postgresql']
        for tech in tech_keywords:
            if tech in response_lower:
                insights['tech_stack'].append(tech)

        # Extract requirements patterns
        if 'need' in response_lower or 'require' in response_lower:
            insights['requirements'].append(user_response[:100])

        # Extract constraints
        if 'budget' in response_lower or 'time' in response_lower or 'cannot' in response_lower:
            insights['constraints'].append(user_response[:100])

        return insights


# Main Agent Orchestrator
class AgentOrchestrator:
    def __init__(self, api_key: str, data_dir: str = None):
        self.data_dir = data_dir or CONFIG['DATA_DIR']

        # Initialize components
        self.database = DatabaseManager(self.data_dir)
        self.vector_db = VectorDatabase(self.data_dir)
        self.context_analyzer = ContextAnalyzer(CONFIG['MAX_CONTEXT_LENGTH'])
        self.claude_client = ClaudeClient(api_key)

        # Initialize agents
        self.session_manager = SessionManagerAgent(self)
        self.conversation_engine = ConversationEngineAgent(self)

        # System state
        self.current_user = None
        self.current_project = None

        self.log("Agent Orchestrator initialized successfully")

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = Fore.CYAN
        print(f"{color}[{timestamp}] Orchestrator: {message}")

        if level == "ERROR":
            logger.error(message)
        elif level == "WARN":
            logger.warning(message)
        else:
            logger.info(message)


# CLI Interface
class SocraticCLI:
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.running = True

    def start(self):
        """Start the CLI interface"""
        print(f"{Fore.YELLOW}{'=' * 60}")
        print(f"{Fore.YELLOW}    🎓 SOCRATIC SOFTWARE DEVELOPMENT SYSTEM 🎓")
        print(f"{Fore.YELLOW}{'=' * 60}")
        print(f"{Fore.GREEN}Welcome to your AI-powered Socratic tutor for software projects!")
        print(f"{Fore.WHITE}Type 'help' for commands, 'quit' to exit\n")

        # Try to restore session
        self._try_restore_session()

        while self.running:
            try:
                if not self.orchestrator.current_user:
                    self._handle_authentication()
                elif not self.orchestrator.current_project:
                    self._handle_project_selection()
                else:
                    self._handle_conversation()
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Goodbye!")
                self._save_session()
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}")
                logger.error(f"CLI error: {e}")

    def _try_restore_session(self):
        """Try to restore the last session"""
        result = self.orchestrator.session_manager.process({
            'action': 'restore_session_state'
        })

        if result['status'] == 'success':
            session_data = result['session_data']
            username = session_data.get('current_user')
            project_id = session_data.get('current_project_id')

            if username:
                # Try to restore user
                auth_result = self.orchestrator.session_manager.process({
                    'action': 'authenticate_user',
                    'username': username,
                    'passcode': input(f"Welcome back {username}! Enter passcode: ")
                })

                if auth_result['status'] == 'success':
                    self.orchestrator.current_user = username
                    print(f"{Fore.GREEN}Session restored for {username}")

                    if project_id:
                        # Try to restore project
                        project_result = self.orchestrator.session_manager.process({
                            'action': 'load_project',
                            'project_id': project_id
                        })

                        if project_result['status'] == 'success':
                            self.orchestrator.current_project = project_result['project']
                            print(f"{Fore.GREEN}Restored project: {self.orchestrator.current_project.name}")

    def _save_session(self):
        """Save current session state"""
        if self.orchestrator.current_user:
            self.orchestrator.session_manager.process({
                'action': 'save_session_state',
                'current_user': self.orchestrator.current_user,
                'current_project_id': self.orchestrator.current_project.project_id if self.orchestrator.current_project else None
            })

    def _handle_authentication(self):
        """Handle user authentication"""
        print(f"\n{Fore.CYAN}=== AUTHENTICATION ===")
        choice = input("1) Login  2) Register  3) Quit\nChoice: ").strip()

        if choice == '1':
            self._login()
        elif choice == '2':
            self._register()
        elif choice == '3':
            self.running = False
        else:
            print(f"{Fore.RED}Invalid choice")

    def _login(self):
        """Handle user login"""
        username = input("Username: ").strip()
        passcode = input("Passcode: ").strip()

        result = self.orchestrator.session_manager.process({
            'action': 'authenticate_user',
            'username': username,
            'passcode': passcode
        })

        if result['status'] == 'success':
            self.orchestrator.current_user = username
            print(f"{Fore.GREEN}Welcome, {username}!")
        else:
            print(f"{Fore.RED}Authentication failed: {result['message']}")

    def _register(self):
        """Handle user registration"""
        username = input("Choose username: ").strip()
        passcode = input("Choose passcode: ").strip()
        confirm = input("Confirm passcode: ").strip()

        if passcode != confirm:
            print(f"{Fore.RED}Passcodes don't match")
            return

        result = self.orchestrator.session_manager.process({
            'action': 'create_user',
            'username': username,
            'passcode': passcode
        })

        if result['status'] == 'success':
            self.orchestrator.current_user = username
            print(f"{Fore.GREEN}Account created! Welcome, {username}!")
        else:
            print(f"{Fore.RED}Registration failed: {result['message']}")

    def _handle_project_selection(self):
        """Handle project selection or creation"""
        print(f"\n{Fore.CYAN}=== PROJECT SELECTION ===")

        # List existing projects
        result = self.orchestrator.session_manager.process({
            'action': 'list_projects',
            'username': self.orchestrator.current_user
        })

        if result['status'] == 'success' and result['projects']:
            print(f"{Fore.WHITE}Your projects:")
            for i, project in enumerate(result['projects'], 1):
                print(f"{i}) {project['name']} (ID: {project['project_id'][:8]}...)")

        print(f"\n{Fore.WHITE}Options:")
        print("1) Create new project")
        print("2) Load existing project")
        print("3) Logout")

        choice = input("Choice: ").strip()

        if choice == '1':
            self._create_project()
        elif choice == '2':
            self._load_project()
        elif choice == '3':
            self.orchestrator.current_user = None
        else:
            print(f"{Fore.RED}Invalid choice")

    def _create_project(self):
        """Create a new project"""
        project_name = input("Project name: ").strip()

        if not project_name:
            print(f"{Fore.RED}Project name cannot be empty")
            return

        result = self.orchestrator.session_manager.process({
            'action': 'create_project',
            'project_name': project_name,
            'owner': self.orchestrator.current_user
        })

        if result['status'] == 'success':
            self.orchestrator.current_project = result['project']
            print(f"{Fore.GREEN}Created project: {project_name}")
            print(f"{Fore.WHITE}Starting in discovery phase...")
        else:
            print(f"{Fore.RED}Failed to create project: {result['message']}")

    def _load_project(self):
        """Load an existing project"""
        project_id = input("Enter project ID (or first 8 characters): ").strip()

        # If short ID provided, try to find full ID
        if len(project_id) == 8:
            result = self.orchestrator.session_manager.process({
                'action': 'list_projects',
                'username': self.orchestrator.current_user
            })

            if result['status'] == 'success':
                for project in result['projects']:
                    if project['project_id'].startswith(project_id):
                        project_id = project['project_id']
                        break

        result = self.orchestrator.session_manager.process({
            'action': 'load_project',
            'project_id': project_id
        })

        if result['status'] == 'success':
            self.orchestrator.current_project = result['project']
            print(f"{Fore.GREEN}Loaded project: {self.orchestrator.current_project.name}")
            print(f"{Fore.WHITE}Current phase: {self.orchestrator.current_project.phase}")
        else:
            print(f"{Fore.RED}Failed to load project: {result['message']}")

    def _handle_conversation(self):
        """Handle the main conversation loop"""
        project = self.orchestrator.current_project

        print(f"\n{Fore.CYAN}=== PROJECT: {project.name.upper()} ===")
        print(f"{Fore.WHITE}Phase: {project.phase} | Owner: {project.owner}")

        if project.goals:
            print(f"{Fore.WHITE}Goals: {project.goals}")

        if project.tech_stack:
            print(f"{Fore.WHITE}Tech Stack: {', '.join(project.tech_stack[:3])}")

        print(
            f"\n{Fore.WHITE}Commands: 'next' (advance phase), 'status' (project info), 'save', 'switch' (project), 'help', 'quit'")
        print(f"{Fore.YELLOW}{'─' * 60}")

        # Generate and ask a question
        question_result = self.orchestrator.conversation_engine.process({
            'action': 'generate_question',
            'project': project
        })

        if question_result['status'] == 'success':
            print(f"\n{Fore.BLUE}🤔 {question_result['question']}")
        else:
            print(f"{Fore.RED}Error generating question: {question_result.get('message', 'Unknown error')}")
            return

        # Get user response
        print(f"\n{Fore.WHITE}Your response (or command):")
        user_input = input("> ").strip()

        # Handle commands
        if user_input.lower() == 'quit':
            self._save_session()
            self.running = False
            return
        elif user_input.lower() == 'help':
            self._show_help()
            return
        elif user_input.lower() == 'status':
            self._show_project_status()
            return
        elif user_input.lower() == 'save':
            self._save_project()
            return
        elif user_input.lower() == 'switch':
            self.orchestrator.current_project = None
            return
        elif user_input.lower() == 'next':
            self._advance_phase()
            return
        elif user_input.lower() == 'dynamic':
            self._toggle_dynamic_mode()
            return
        elif user_input.lower() == 'conflicts':
            self._show_conflicts()
            return
        elif not user_input:
            print(f"{Fore.YELLOW}Please provide a response or command.")
            return

        # Process the response
        response_result = self.orchestrator.conversation_engine.process({
            'action': 'process_response',
            'project': project,
            'response': user_input,
            'current_user': self.orchestrator.current_user
        })

        if response_result['status'] == 'success':
            insights = response_result.get('insights', {})
            if insights:
                print(f"{Fore.GREEN}✓ Insights captured and integrated into project context")

                # Show conflicts if any were detected
                if response_result.get('conflicts_pending'):
                    print(f"{Fore.YELLOW}⚠ Potential conflicts detected - review project status")

            # Auto-save project after each response
            self.orchestrator.session_manager.process({
                'action': 'save_project',
                'project': project
            })
        else:
            print(f"{Fore.RED}Error processing response: {response_result.get('message', 'Unknown error')}")

    def _show_help(self):
        """Show help information"""
        print(f"\n{Fore.CYAN}=== HELP ===")
        print(f"{Fore.WHITE}Commands:")
        print("  help     - Show this help")
        print("  status   - Show detailed project status")
        print("  save     - Save current project")
        print("  switch   - Switch to different project")
        print("  next     - Advance to next phase")
        print("  dynamic  - Toggle dynamic/static questions")
        print("  conflicts- Show detected conflicts")
        print("  quit     - Exit system")
        print(f"\n{Fore.WHITE}Phases: discovery → analysis → design → implementation")
        print(f"{Fore.WHITE}Just type your responses to questions to continue the conversation.")

    def _show_project_status(self):
        """Show detailed project status"""
        project = self.orchestrator.current_project

        print(f"\n{Fore.CYAN}=== PROJECT STATUS ===")
        print(f"{Fore.WHITE}Name: {project.name}")
        print(f"ID: {project.project_id}")
        print(f"Owner: {project.owner}")
        print(f"Phase: {project.phase}")
        print(f"Created: {project.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"Updated: {project.updated_at.strftime('%Y-%m-%d %H:%M')}")

        if project.collaborators:
            print(f"Collaborators: {', '.join(project.collaborators)}")

        if project.goals:
            print(f"\n{Fore.YELLOW}Goals:")
            print(f"  {project.goals}")

        if project.tech_stack:
            print(f"\n{Fore.YELLOW}Tech Stack:")
            for tech in project.tech_stack:
                print(f"  • {tech}")

        if project.requirements:
            print(f"\n{Fore.YELLOW}Requirements:")
            for req in project.requirements:
                print(f"  • {req}")

        if project.constraints:
            print(f"\n{Fore.YELLOW}Constraints:")
            for constraint in project.constraints:
                print(f"  • {constraint}")

        # Show conversation statistics
        total_messages = len(project.conversation_history)
        user_messages = len([msg for msg in project.conversation_history if msg['type'] == 'user'])
        print(f"\n{Fore.WHITE}Conversation: {user_messages} responses, {total_messages} total messages")

    def _save_project(self):
        """Save current project"""
        result = self.orchestrator.session_manager.process({
            'action': 'save_project',
            'project': self.orchestrator.current_project
        })

        if result['status'] == 'success':
            print(f"{Fore.GREEN}✓ Project saved successfully")
        else:
            print(f"{Fore.RED}Error saving project: {result.get('message', 'Unknown error')}")

    def _advance_phase(self):
        """Advance to next phase"""
        result = self.orchestrator.conversation_engine.process({
            'action': 'advance_phase',
            'project': self.orchestrator.current_project
        })

        if result['status'] == 'success':
            print(f"{Fore.GREEN}✓ Advanced to {result['new_phase']} phase")
            self._save_project()
        else:
            print(f"{Fore.YELLOW}{result.get('message', 'Cannot advance phase')}")

    def _toggle_dynamic_mode(self):
        """Toggle between dynamic and static question modes"""
        result = self.orchestrator.conversation_engine.process({
            'action': 'toggle_dynamic_questions'
        })

        if result['status'] == 'success':
            mode = "dynamic (AI-powered)" if result['dynamic_mode'] else "static (predefined)"
            print(f"{Fore.GREEN}✓ Question mode: {mode}")
        else:
            print(f"{Fore.RED}Error toggling mode")

    def _show_conflicts(self):
        """Show any detected conflicts"""
        project = self.orchestrator.current_project

        # Run conflict detection on current project state
        result = self.orchestrator.conversation_engine.process({
            'action': 'detect_conflicts',
            'project': project,
            'new_insights': {},  # Check existing state
            'current_user': self.orchestrator.current_user
        })

        if result['status'] == 'success':
            conflicts = result.get('conflicts', [])
            if conflicts:
                print(f"\n{Fore.YELLOW}⚠ DETECTED CONFLICTS ({len(conflicts)}):")
                for i, conflict in enumerate(conflicts, 1):
                    print(f"\n{i}. {conflict.conflict_type.replace('_', ' ').title()}")
                    print(f"   Old: {conflict.old_value}")
                    print(f"   New: {conflict.new_value}")
                    print(f"   Severity: {conflict.severity}")
                    if conflict.suggestions:
                        print(f"   Suggestions:")
                        for suggestion in conflict.suggestions[:2]:
                            print(f"     • {suggestion}")
            else:
                print(f"{Fore.GREEN}✓ No conflicts detected")
        else:
            print(f"{Fore.RED}Error checking conflicts")


def main():
    """Main entry point"""
    # Get API key from environment or user input
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print(f"{Fore.YELLOW}ANTHROPIC_API_KEY not found in environment.")
        api_key = input("Enter your Anthropic API key: ").strip()

        if not api_key:
            print(f"{Fore.RED}API key is required to run the system.")
            return

    try:
        # Initialize the orchestrator
        orchestrator = AgentOrchestrator(api_key)

        # Start CLI
        cli = SocraticCLI(orchestrator)
        cli.start()

    except Exception as e:
        print(f"{Fore.RED}Failed to start system: {e}")
        logger.error(f"System startup error: {e}")


if __name__ == "__main__":
    main()

