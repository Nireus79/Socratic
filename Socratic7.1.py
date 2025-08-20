#!/usr/bin/env python3

import os
import json
import hashlib
import getpass
import datetime
import pickle
import uuid
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from contextlib import contextmanager
import sqlite3
import logging
from functools import wraps, lru_cache
import numpy as np
from colorama import init, Fore, Back, Style

# Third-party imports with better error handling
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


# Enhanced Configuration
@dataclass
class SystemConfig:
    # Core settings
    MAX_CONTEXT_LENGTH: int = 8000
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
    CLAUDE_MODEL: str = 'claude-3-5-sonnet-20241022'
    DATA_DIR: str = 'socratic_data'

    # Performance settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    CONNECTION_POOL_SIZE: int = 5
    CACHE_SIZE: int = 100
    MAX_CONVERSATION_HISTORY: int = 50
    BATCH_PROCESS_SIZE: int = 5

    # API settings
    TOKEN_WARNING_THRESHOLD: float = 0.8
    API_TIMEOUT: int = 30
    MAX_CONCURRENT_REQUESTS: int = 3

    # Session settings
    SESSION_TIMEOUT: int = 3600
    AUTO_SAVE_INTERVAL: int = 300
    MAX_TOKEN_USAGE_HISTORY: int = 100


CONFIG = SystemConfig()


# Enhanced Data Models
@dataclass
class User:
    username: str
    passcode_hash: str
    created_at: datetime.datetime
    projects: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    last_login: Optional[datetime.datetime] = None


@dataclass
class ProjectContext:
    project_id: str
    name: str
    owner: str
    collaborators: List[str] = field(default_factory=list)
    goals: str = ""
    requirements: List[str] = field(default_factory=list)
    tech_stack: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    team_structure: str = "individual"
    language_preferences: str = "python"
    deployment_target: str = "local"
    code_style: str = "documented"
    phase: str = "discovery"
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=CONFIG.MAX_CONVERSATION_HISTORY))
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    version: int = 1


@dataclass
class CachedResponse:
    key: str
    response: str
    timestamp: datetime.datetime
    expiry: datetime.datetime
    usage_count: int = 0


@dataclass
class SystemMetrics:
    total_tokens: int = 0
    total_cost: float = 0.0
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time: float = 0.0
    error_count: int = 0
    uptime_start: datetime.datetime = field(default_factory=datetime.datetime.now)


# Enhanced Logging Setup
def setup_logging():
    """Setup structured logging"""
    log_dir = os.path.join(CONFIG.DATA_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'socratic.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# Performance Decorators
def measure_time(func):
    """Decorator to measure function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.2f}s")
        return result

    return wrapper


def retry_with_backoff(max_retries=3, base_delay=1.0):
    """Decorator for exponential backoff retry logic"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


# Enhanced Database Connection Pool
class DatabasePool:
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._connections = deque()
        self._lock = threading.Lock()
        self._init_pool()

    def _init_pool(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._connections.append(conn)

    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager"""
        with self._lock:
            if self._connections:
                conn = self._connections.popleft()
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row

        try:
            yield conn
        finally:
            with self._lock:
                if len(self._connections) < self.pool_size:
                    self._connections.append(conn)
                else:
                    conn.close()


# Enhanced Cache Manager
class IntelligentCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, CachedResponse] = {}
        self._access_times: Dict[str, datetime.datetime] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        """Get cached response if valid"""
        with self._lock:
            if key in self._cache:
                cached = self._cache[key]
                if datetime.datetime.now() < cached.expiry:
                    cached.usage_count += 1
                    self._access_times[key] = datetime.datetime.now()
                    return cached.response
                else:
                    # Expired
                    del self._cache[key]
                    del self._access_times[key]
        return None

    def put(self, key: str, response: str, ttl_seconds: int = 3600):
        """Cache response with TTL"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._evict_oldest()

            expiry = datetime.datetime.now() + datetime.timedelta(seconds=ttl_seconds)
            self._cache[key] = CachedResponse(key, response, datetime.datetime.now(), expiry)
            self._access_times[key] = datetime.datetime.now()

    def _evict_oldest(self):
        """Evict least recently used item"""
        if not self._access_times:
            return

        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[oldest_key]
        del self._access_times[oldest_key]


# Consolidated Agent Architecture
class BaseAgent(ABC):
    def __init__(self, name: str, orchestrator: 'SystemOrchestrator'):
        self.name = name
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.metrics = defaultdict(int)

    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = Fore.GREEN if level == "INFO" else Fore.RED if level == "ERROR" else Fore.YELLOW
        print(f"{color}[{timestamp}] {self.name}: {message}")
        self.logger.info(f"{self.name}: {message}")


class SessionManager(BaseAgent):
    """Unified session, project, and user management"""

    def __init__(self, orchestrator):
        super().__init__("SessionManager", orchestrator)
        self.active_sessions: Dict[str, Dict] = {}
        self.project_cache: Dict[str, ProjectContext] = {}

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        handlers = {
            'authenticate_user': self._authenticate_user,
            'create_user': self._create_user,
            'create_project': self._create_project,
            'load_project': self._load_project,
            'save_project': self._save_project,
            'list_projects': self._list_projects,
            'manage_collaborators': self._manage_collaborators,
            'get_session_state': self._get_session_state,
            'cleanup_sessions': self._cleanup_sessions
        }

        handler = handlers.get(action)
        if handler:
            return await handler(request)

        return {'status': 'error', 'message': f'Unknown action: {action}'}

    @measure_time
    async def _authenticate_user(self, request: Dict) -> Dict:
        """Enhanced user authentication with session management"""
        username = request.get('username')
        passcode = request.get('passcode')

        with self.orchestrator.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT data FROM users WHERE username = ?', (username,))
            result = cursor.fetchone()

            if result:
                user_data = pickle.loads(result[0])
                user = User(**user_data)

                # Verify passcode
                passcode_hash = hashlib.sha256(passcode.encode()).hexdigest()
                if user.passcode_hash == passcode_hash:
                    # Create session
                    session_id = str(uuid.uuid4())
                    session_data = {
                        'user': user,
                        'created_at': datetime.datetime.now(),
                        'last_activity': datetime.datetime.now()
                    }
                    self.active_sessions[session_id] = session_data

                    # Update last login
                    user.last_login = datetime.datetime.now()
                    self._save_user_data(user)

                    self.log(f"User {username} authenticated successfully")
                    return {
                        'status': 'success',
                        'session_id': session_id,
                        'user': user
                    }

        return {'status': 'error', 'message': 'Invalid credentials'}

    @measure_time
    async def _create_project(self, request: Dict) -> Dict:
        """Create new project with enhanced validation"""
        project_name = request.get('project_name')
        owner = request.get('owner')

        # Validate project name
        if not project_name or len(project_name.strip()) < 2:
            return {'status': 'error', 'message': 'Project name must be at least 2 characters'}

        project_id = str(uuid.uuid4())
        project = ProjectContext(
            project_id=project_id,
            name=project_name.strip(),
            owner=owner
        )

        # Save to database and cache
        await self._save_project({'project': project})
        self.project_cache[project_id] = project

        self.log(f"Created project '{project_name}' with ID {project_id}")
        return {'status': 'success', 'project': project}

    @lru_cache(maxsize=50)
    def _get_cached_project(self, project_id: str) -> Optional[ProjectContext]:
        """Get project from cache or database"""
        if project_id in self.project_cache:
            return self.project_cache[project_id]

        with self.orchestrator.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT data FROM projects WHERE project_id = ?', (project_id,))
            result = cursor.fetchone()

            if result:
                project_data = pickle.loads(result[0])
                project = ProjectContext(**project_data)
                self.project_cache[project_id] = project
                return project

        return None

    async def _save_project(self, request: Dict) -> Dict:
        """Save project with optimistic locking"""
        project = request.get('project')
        project.updated_at = datetime.datetime.now()
        project.version += 1

        with self.orchestrator.db_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Check for version conflicts
            cursor.execute('SELECT version FROM projects WHERE project_id = ?', (project.project_id,))
            result = cursor.fetchone()

            if result and result[0] > project.version - 1:
                return {'status': 'error', 'message': 'Project was modified by another user'}

            # Save project
            data = pickle.dumps(asdict(project))
            cursor.execute('''
                INSERT OR REPLACE INTO projects (project_id, data, updated_at, version)
                VALUES (?, ?, ?, ?)
            ''', (project.project_id, data, project.updated_at.isoformat(), project.version))

            conn.commit()

        # Update cache
        self.project_cache[project.project_id] = project

        return {'status': 'success'}

    def _save_user_data(self, user: User):
        """Save user data to database"""
        with self.orchestrator.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            data = pickle.dumps(asdict(user))
            cursor.execute('''
                INSERT OR REPLACE INTO users (username, passcode_hash, data, created_at)
                VALUES (?, ?, ?, ?)
            ''', (user.username, user.passcode_hash, data, user.created_at.isoformat()))
            conn.commit()


class ConversationEngine(BaseAgent):
    """Enhanced conversation management with conflict detection"""

    def __init__(self, orchestrator):
        super().__init__("ConversationEngine", orchestrator)
        self.response_cache = IntelligentCache(max_size=50)
        self.question_cache = IntelligentCache(max_size=30)

        # Enhanced question templates with better variety
        self.question_templates = {
            'discovery': [
                "What specific pain point or inefficiency does your project address?",
                "Who would benefit most from using your solution, and why?",
                "What makes your approach different from existing alternatives?",
                "What would success look like for this project in concrete terms?",
                "What assumptions are you making about your users or market?"
            ],
            'analysis': [
                "What technical challenges worry you most about this project?",
                "How will you measure and ensure the performance of your solution?",
                "What happens if your project needs to scale beyond initial expectations?",
                "What external dependencies might become bottlenecks or risks?",
                "How will you validate that your technical approach solves the core problem?"
            ],
            'design': [
                "How will different parts of your system communicate and share data?",
                "What design patterns would make your code most maintainable?",
                "How will you handle errors and edge cases throughout your system?",
                "What would make it easy for other developers to understand your code?",
                "How will you structure your code to accommodate future changes?"
            ],
            'implementation': [
                "What's the smallest version of your project that would still be valuable?",
                "How will you know when each part of your system is working correctly?",
                "What's your strategy for deploying and monitoring your solution?",
                "How will you handle updates and maintenance after launch?",
                "What documentation will others need to use or extend your work?"
            ]
        }

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        handlers = {
            'generate_question': self._generate_question,
            'process_response': self._process_response,
            'detect_conflicts': self._detect_conflicts,
            'advance_phase': self._advance_phase,
            'get_suggestions': self._get_suggestions,
            'batch_process_responses': self._batch_process_responses
        }

        handler = handlers.get(action)
        if handler:
            return await handler(request)

        return {'status': 'error', 'message': f'Unknown action: {action}'}

    @measure_time
    async def _generate_question(self, request: Dict) -> Dict:
        """Generate contextual question with caching"""
        project = request.get('project')
        use_dynamic = request.get('use_dynamic', True)

        # Create cache key
        context_key = f"{project.project_id}_{project.phase}_{len(project.conversation_history)}"

        # Check cache first
        cached_question = self.question_cache.get(context_key)
        if cached_question:
            self.orchestrator.metrics.cache_hits += 1
            return {'status': 'success', 'question': cached_question, 'cached': True}

        self.orchestrator.metrics.cache_misses += 1

        # Generate new question
        if use_dynamic and self.orchestrator.claude_client:
            try:
                question = await self._generate_dynamic_question(project)
            except Exception as e:
                self.log(f"Dynamic question generation failed: {e}", "WARN")
                question = self._get_static_question(project)
        else:
            question = self._get_static_question(project)

        # Cache the question
        self.question_cache.put(context_key, question, ttl_seconds=1800)

        # Add to conversation history
        project.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'assistant',
            'content': question,
            'phase': project.phase
        })

        return {'status': 'success', 'question': question, 'cached': False}

    async def _generate_dynamic_question(self, project: ProjectContext) -> str:
        """Generate dynamic question using Claude with context optimization"""

        # Build optimized context
        context_parts = []
        if project.goals:
            context_parts.append(f"Goals: {project.goals}")
        if project.tech_stack:
            context_parts.append(f"Tech: {', '.join(project.tech_stack[-3:])}")  # Last 3 items
        if project.requirements:
            context_parts.append(f"Requirements: {', '.join(project.requirements[-3:])}")

        # Get relevant conversation snippets
        recent_conversation = []
        if project.conversation_history:
            for msg in list(project.conversation_history)[-4:]:
                role = "A" if msg['type'] == 'assistant' else "U"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                recent_conversation.append(f"{role}: {content}")

        # Build compact prompt
        prompt = f"""Generate a Socratic question for a {project.phase} phase project.

Context: {'; '.join(context_parts)}

Recent: {' | '.join(recent_conversation[-2:])}

Create ONE question that:
- Builds on the conversation
- Challenges assumptions
- Is specific to {project.phase}
- Encourages deep thinking

Return only the question."""

        return await self.orchestrator.claude_client.generate_response(prompt, max_tokens=150)

    def _get_static_question(self, project: ProjectContext) -> str:
        """Get static question from templates"""
        questions = self.question_templates.get(project.phase, [])
        if not questions:
            return "What would you like to explore further about your project?"

        # Count phase questions already asked
        phase_questions = sum(1 for msg in project.conversation_history
                              if msg.get('type') == 'assistant' and msg.get('phase') == project.phase)

        if phase_questions < len(questions):
            return questions[phase_questions]
        else:
            # Cycle through or use fallback
            return questions[phase_questions % len(questions)]

    @measure_time
    async def _process_response(self, request: Dict) -> Dict:
        """Process user response with batch insights extraction"""
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

        # Extract insights
        insights = await self._extract_insights(user_response, project)

        # Detect conflicts if insights exist
        conflicts = []
        if insights:
            conflict_result = await self._detect_conflicts({
                'project': project,
                'new_insights': insights,
                'current_user': current_user
            })
            conflicts = conflict_result.get('conflicts', [])

        # Update project context if no conflicts
        if not conflicts and insights:
            self._update_project_context(project, insights)

        return {
            'status': 'success',
            'insights': insights,
            'conflicts': conflicts,
            'conflicts_pending': bool(conflicts)
        }

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def _extract_insights(self, response: str, project: ProjectContext) -> Dict:
        """Extract insights with caching and error handling"""
        if not response or len(response.strip()) < 3:
            return {}

        # Create cache key
        cache_key = hashlib.md5(f"{response}_{project.phase}".encode()).hexdigest()

        # Check cache
        cached_insights = self.response_cache.get(cache_key)
        if cached_insights:
            self.orchestrator.metrics.cache_hits += 1
            return json.loads(cached_insights)

        self.orchestrator.metrics.cache_misses += 1

        # Build compact extraction prompt
        prompt = f"""Extract project insights from this response in the {project.phase} phase:

"{response}"

Current project context:
- Goals: {project.goals or 'None'}
- Tech: {', '.join(project.tech_stack[-3:]) if project.tech_stack else 'None'}

Return JSON with any mentioned:
{{"goals": "specific goals", "requirements": ["req1", "req2"], "tech_stack": ["tech1"], "constraints": ["constraint1"]}}

If nothing relevant, return {{}}.
"""

        try:
            response_text = await self.orchestrator.claude_client.generate_response(
                prompt, max_tokens=500, temperature=0.2
            )

            # Parse JSON
            insights = self._parse_insights_json(response_text)

            # Cache successful result
            self.response_cache.put(cache_key, json.dumps(insights), ttl_seconds=3600)

            return insights

        except Exception as e:
            self.log(f"Insight extraction failed: {e}", "ERROR")
            return {}

    def _parse_insights_json(self, response_text: str) -> Dict:
        """Safely parse insights JSON from Claude response"""
        try:
            # Clean response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()

            # Find JSON boundaries
            start = response_text.find('{')
            end = response_text.rfind('}') + 1

            if 0 <= start < end:
                json_text = response_text[start:end]
                return json.loads(json_text)

        except (json.JSONDecodeError, ValueError) as e:
            self.log(f"JSON parsing failed: {e}", "WARN")

        return {}

    def _update_project_context(self, project: ProjectContext, insights: Dict):
        """Update project context with validated insights"""
        if not insights:
            return

        try:
            # Update goals
            if insights.get('goals') and isinstance(insights['goals'], str):
                project.goals = insights['goals'].strip()

            # Update lists with deduplication
            for list_field in ['requirements', 'tech_stack', 'constraints']:
                if insights.get(list_field):
                    items = insights[list_field]
                    if isinstance(items, list):
                        current_list = getattr(project, list_field)
                        new_items = [item.strip() for item in items
                                     if isinstance(item, str) and item.strip()
                                     and item.strip().lower() not in [x.lower() for x in current_list]]
                        current_list.extend(new_items)

        except Exception as e:
            self.log(f"Context update failed: {e}", "ERROR")


class ContentGenerator(BaseAgent):
    """Enhanced content generation with templates and optimization"""

    def __init__(self, orchestrator):
        super().__init__("ContentGenerator", orchestrator)
        self.generation_cache = IntelligentCache(max_size=20)
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        if action == 'generate_code':
            return await self._generate_code(request)
        elif action == 'generate_documentation':
            return await self._generate_documentation(request)
        elif action == 'generate_project_report':
            return await self._generate_project_report(request)

        return {'status': 'error', 'message': f'Unknown action: {action}'}

    @measure_time
    async def _generate_code(self, request: Dict) -> Dict:
        """Generate code with caching and optimization"""
        project = request.get('project')

        # Create cache key
        context_key = self._create_context_key(project)
        cached_code = self.generation_cache.get(f"code_{context_key}")

        if cached_code:
            return {'status': 'success', 'code': cached_code, 'cached': True}

        # Build generation context
        context = self._build_generation_context(project)

        try:
            # Generate code asynchronously
            loop = asyncio.get_event_loop()
            code = await loop.run_in_executor(
                self.executor,
                self._generate_code_sync,
                context
            )

            # Cache result
            self.generation_cache.put(f"code_{context_key}", code, ttl_seconds=7200)

            return {'status': 'success', 'code': code, 'cached': False}

        except Exception as e:
            self.log(f"Code generation failed: {e}", "ERROR")
            return {'status': 'error', 'message': str(e)}

    def _generate_code_sync(self, context: str) -> str:
        """Synchronous code generation for executor"""
        prompt = f"""Generate production-ready code based on this project:

{context}

Requirements:
- Include proper error handling and logging
- Add comprehensive docstrings and comments
- Follow best practices for the chosen technology
- Include basic tests or validation
- Make it maintainable and extensible

Provide a complete, working implementation."""

        return self.orchestrator.claude_client.generate_response_sync(
            prompt, max_tokens=4000, temperature=0.3
        )

    def _create_context_key(self, project: ProjectContext) -> str:
        """Create cache key from project context"""
        key_parts = [
            project.project_id,
            str(hash(project.goals)),
            str(hash(''.join(project.tech_stack))),
            str(hash(''.join(project.requirements))),
            str(project.version)
        ]
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()

    def _build_generation_context(self, project: ProjectContext) -> str:
        """Build comprehensive context for code generation"""
        context_parts = [
            f"Project: {project.name}",
            f"Phase: {project.phase}",
            f"Goals: {project.goals}",
            f"Primary Tech Stack: {', '.join(project.tech_stack[:5])}",
            f"Key Requirements: {', '.join(project.requirements[:5])}",
            f"Constraints: {', '.join(project.constraints[:3])}",
            f"Deployment Target: {project.deployment_target}",
            f"Code Style: {project.code_style}"
        ]

        # Add relevant conversation insights
        if project.conversation_history:
            recent_user_responses = [
                msg['content'] for msg in list(project.conversation_history)[-10:]
                if msg.get('type') == 'user' and len(msg['content']) > 20
            ]
            if recent_user_responses:
                context_parts.append(f"