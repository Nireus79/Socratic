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
import asyncio

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
                context_parts.append(f"Recent Context: {' | '.join(recent_user_responses[-3:])}")

        return '\n'.join(context_parts)


class KnowledgeManager(BaseAgent):
    """Enhanced knowledge management with vector search"""

    def __init__(self, orchestrator):
        super().__init__("KnowledgeManager", orchestrator)
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize ChromaDB and embedding model"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(CONFIG.EMBEDDING_MODEL)

            # Initialize ChromaDB
            chroma_db_path = os.path.join(CONFIG.DATA_DIR, 'chroma_db')
            os.makedirs(chroma_db_path, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)

            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection("socratic_knowledge")
            except:
                self.collection = self.chroma_client.create_collection("socratic_knowledge")

            self.log("Knowledge base initialized successfully")

        except Exception as e:
            self.log(f"Knowledge base initialization failed: {e}", "ERROR")
            self.embedding_model = None
            self.chroma_client = None

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        handlers = {
            'store_knowledge': self._store_knowledge,
            'retrieve_knowledge': self._retrieve_knowledge,
            'update_project_knowledge': self._update_project_knowledge,
            'search_similar_projects': self._search_similar_projects
        }

        handler = handlers.get(action)
        if handler:
            return await handler(request)

        return {'status': 'error', 'message': f'Unknown action: {action}'}

    @measure_time
    async def _store_knowledge(self, request: Dict) -> Dict:
        """Store knowledge with embeddings"""
        if not self.collection:
            return {'status': 'error', 'message': 'Knowledge base not available'}

        knowledge_id = request.get('id', str(uuid.uuid4()))
        content = request.get('content')
        metadata = request.get('metadata', {})

        try:
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()

            # Store in ChromaDB
            self.collection.upsert(
                ids=[knowledge_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )

            return {'status': 'success', 'id': knowledge_id}

        except Exception as e:
            self.log(f"Knowledge storage failed: {e}", "ERROR")
            return {'status': 'error', 'message': str(e)}

    @measure_time
    async def _retrieve_knowledge(self, request: Dict) -> Dict:
        """Retrieve relevant knowledge using similarity search"""
        if not self.collection:
            return {'status': 'success', 'results': []}

        query = request.get('query')
        limit = request.get('limit', 5)

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Search similar content
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )

            # Format results
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    })

            return {'status': 'success', 'results': formatted_results}

        except Exception as e:
            self.log(f"Knowledge retrieval failed: {e}", "ERROR")
            return {'status': 'success', 'results': []}


class ClaudeClient:
    """Enhanced Claude API client with rate limiting and error handling"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.token_usage: deque = deque(maxlen=CONFIG.MAX_TOKEN_USAGE_HISTORY)
        self.request_semaphore = asyncio.Semaphore(CONFIG.MAX_CONCURRENT_REQUESTS)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests

    @retry_with_backoff(max_retries=CONFIG.MAX_RETRIES, base_delay=CONFIG.RETRY_DELAY)
    async def generate_response(self, prompt: str, max_tokens: int = 1000,
                                temperature: float = 0.7, system_prompt: str = None) -> str:
        """Generate response with async support and rate limiting"""
        async with self.request_semaphore:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)

            try:
                messages = [{"role": "user", "content": prompt}]

                kwargs = {
                    "model": CONFIG.CLAUDE_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

                if system_prompt:
                    kwargs["system"] = system_prompt

                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(**kwargs)
                )

                # Track usage
                if hasattr(response, 'usage'):
                    self.token_usage.append({
                        'timestamp': datetime.datetime.now(),
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens
                    })

                self.last_request_time = time.time()
                return response.content[0].text

            except Exception as e:
                logger.error(f"Claude API error: {e}")
                raise

    def generate_response_sync(self, prompt: str, max_tokens: int = 1000,
                               temperature: float = 0.7) -> str:
        """Synchronous version for use in executors"""
        try:
            response = self.client.messages.create(
                model=CONFIG.CLAUDE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API sync error: {e}")
            raise

    def get_token_usage_stats(self) -> Dict:
        """Get token usage statistics"""
        if not self.token_usage:
            return {'total_input': 0, 'total_output': 0, 'requests': 0}

        total_input = sum(usage['input_tokens'] for usage in self.token_usage)
        total_output = sum(usage['output_tokens'] for usage in self.token_usage)

        return {
            'total_input': total_input,
            'total_output': total_output,
            'requests': len(self.token_usage),
            'avg_input': total_input / len(self.token_usage) if self.token_usage else 0,
            'avg_output': total_output / len(self.token_usage) if self.token_usage else 0
        }


class SystemOrchestrator:
    """Main orchestrator for the Socratic development system"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or CONFIG.DATA_DIR
        self.metrics = SystemMetrics()
        self.claude_client: Optional[ClaudeClient] = None
        self.db_pool: Optional[DatabasePool] = None

        # Initialize agents
        self.session_manager = SessionManager(self)
        self.conversation_engine = ConversationEngine(self)
        self.content_generator = ContentGenerator(self)
        self.knowledge_manager = KnowledgeManager(self)

        # Auto-save thread
        self._auto_save_thread = None
        self._shutdown_flag = threading.Event()

        self._initialize_system()

    def _initialize_system(self):
        """Initialize the complete system"""
        try:
            # Create data directory
            os.makedirs(self.data_dir, exist_ok=True)

            # Initialize database
            self._initialize_database()

            # Initialize Claude client if API key available
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.claude_client = ClaudeClient(api_key)
                logger.info("Claude client initialized")
            else:
                logger.warning("ANTHROPIC_API_KEY not found. Dynamic features disabled.")

            # Start auto-save thread
            self._start_auto_save()

            logger.info("System initialization completed successfully")

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise

    def _initialize_database(self):
        """Initialize SQLite database with connection pooling"""
        db_path = os.path.join(self.data_dir, 'socratic.db')
        self.db_pool = DatabasePool(db_path, CONFIG.CONNECTION_POOL_SIZE)

        # Create tables
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    passcode_hash TEXT NOT NULL,
                    data BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')

            # Projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER DEFAULT 1
                )
            ''')

            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    data BLOB NOT NULL
                )
            ''')

            # Conversations table for backup
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    conversation_data BLOB NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id)
                )
            ''')

            conn.commit()

        logger.info("Database initialized successfully")

    def _start_auto_save(self):
        """Start auto-save thread for periodic data persistence"""

        def auto_save_worker():
            while not self._shutdown_flag.wait(CONFIG.AUTO_SAVE_INTERVAL):
                try:
                    self._perform_auto_save()
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")

        self._auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self._auto_save_thread.start()

    def _perform_auto_save(self):
        """Perform automatic data persistence"""
        # Save metrics
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            metrics_data = pickle.dumps(asdict(self.metrics))
            cursor.execute(
                'INSERT INTO metrics (timestamp, data) VALUES (?, ?)',
                (datetime.datetime.now().isoformat(), metrics_data)
            )
            conn.commit()

    async def process_request(self, agent_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route requests to appropriate agents"""
        agents = {
            'session': self.session_manager,
            'conversation': self.conversation_engine,
            'content': self.content_generator,
            'knowledge': self.knowledge_manager
        }

        agent = agents.get(agent_name)
        if agent:
            return await agent.process(request)

        return {'status': 'error', 'message': f'Unknown agent: {agent_name}'}

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'uptime': datetime.datetime.now() - self.metrics.uptime_start,
            'active_sessions': len(self.session_manager.active_sessions),
            'cached_projects': len(self.session_manager.project_cache),
            'metrics': asdict(self.metrics),
            'claude_available': self.claude_client is not None,
            'knowledge_base_available': self.knowledge_manager.collection is not None,
            'token_usage': self.claude_client.get_token_usage_stats() if self.claude_client else {}
        }

    def shutdown(self):
        """Graceful shutdown of the system"""
        logger.info("Shutting down system...")

        # Signal shutdown
        self._shutdown_flag.set()

        # Wait for auto-save thread
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            self._auto_save_thread.join(timeout=5)

        # Final save
        self._perform_auto_save()

        # Close content generator executor
        if hasattr(self.content_generator, 'executor'):
            self.content_generator.executor.shutdown(wait=True)

        logger.info("System shutdown complete")


class SocraticInterface:
    """Enhanced command-line interface with better UX"""

    def __init__(self):
        self.orchestrator = SystemOrchestrator()
        self.current_session = None
        self.current_project = None

        # Interface state
        self.show_system_info = True
        self.auto_advance_phases = False
        self.conversation_limit = 10

    def print_header(self):
        """Print enhanced system header"""
        print(f"{Fore.CYAN}{'=' * 60}")
        print(f"{Style.BRIGHT}🧠 Socratic Development Assistant v7.1")
        print(f"{Style.NORMAL}Advanced AI-Powered Project Development")
        print(f"{'=' * 60}{Style.RESET_ALL}")

        if self.show_system_info:
            status = self.orchestrator.get_system_status()
            print(f"{Fore.GREEN}📊 System Status:")
            print(f"  • Claude API: {'✓ Available' if status['claude_available'] else '✗ Unavailable'}")
            print(f"  • Knowledge Base: {'✓ Ready' if status['knowledge_base_available'] else '✗ Not Ready'}")
            print(f"  • Active Sessions: {status['active_sessions']}")
            print(f"  • Uptime: {str(status['uptime']).split('.')[0]}")

            if status['token_usage']:
                usage = status['token_usage']
                print(f"  • API Usage: {usage['requests']} requests, {usage['total_input']} input tokens")
            print()

    async def run(self):
        """Enhanced main interface loop"""
        try:
            self.print_header()

            # Authentication
            session_result = await self._handle_authentication()
            if not session_result:
                return

            # Main loop
            while True:
                try:
                    if not self.current_project:
                        await self._handle_project_selection()
                    else:
                        await self._handle_conversation()

                except KeyboardInterrupt:
                    print(
                        f"\n{Fore.YELLOW}⏸️  Session paused. Type 'exit' to quit or continue conversation.{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
                    logger.error(f"Interface error: {e}")

        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}")
        finally:
            self.orchestrator.shutdown()

    async def _handle_authentication(self) -> bool:
        """Enhanced authentication with registration option"""
        print(f"{Fore.YELLOW}🔐 Authentication Required{Style.RESET_ALL}")

        while True:
            username = input("Username: ").strip()
            if not username:
                continue

            choice = input("(L)ogin or (R)egister? ").strip().lower()

            if choice.startswith('r'):
                # Registration
                if len(username) < 2:
                    print(f"{Fore.RED}Username must be at least 2 characters{Style.RESET_ALL}")
                    continue

                passcode = getpass.getpass("Create passcode: ")
                confirm_passcode = getpass.getpass("Confirm passcode: ")

                if passcode != confirm_passcode:
                    print(f"{Fore.RED}Passcodes don't match{Style.RESET_ALL}")
                    continue

                # Create user
                result = await self.orchestrator.process_request('session', {
                    'action': 'create_user',
                    'username': username,
                    'passcode': passcode
                })

                if result['status'] == 'success':
                    print(f"{Fore.GREEN}✓ User registered successfully{Style.RESET_ALL}")
                    # Continue to login
                    choice = 'l'
                else:
                    print(f"{Fore.RED}Registration failed: {result.get('message', 'Unknown error')}{Style.RESET_ALL}")
                    continue

            if choice.startswith('l'):
                # Login
                passcode = getpass.getpass("Passcode: ")

                result = await self.orchestrator.process_request('session', {
                    'action': 'authenticate_user',
                    'username': username,
                    'passcode': passcode
                })

                if result['status'] == 'success':
                    self.current_session = result
                    user = result['user']
                    print(f"{Fore.GREEN}✓ Welcome back, {user['username']}!{Style.RESET_ALL}")
                    if user.get('last_login'):
                        print(f"Last login: {user['last_login']}")
                    print()
                    return True
                else:
                    print(
                        f"{Fore.RED}Authentication failed: {result.get('message', 'Invalid credentials')}{Style.RESET_ALL}")

    async def _handle_project_selection(self):
        """Enhanced project selection with management options"""
        print(f"{Fore.CYAN}📁 Project Management{Style.RESET_ALL}")

        # List existing projects
        projects_result = await self.orchestrator.process_request('session', {
            'action': 'list_projects',
            'username': self.current_session['user']['username']
        })

        projects = projects_result.get('projects', [])

        if projects:
            print("Your projects:")
            for i, project in enumerate(projects, 1):
                phase_color = self._get_phase_color(project.get('phase', 'discovery'))
                print(f"  {i}. {Fore.WHITE}{project['name']}{Style.RESET_ALL} "
                      f"({phase_color}{project.get('phase', 'discovery')}{Style.RESET_ALL})")

        print("\nOptions:")
        print("  (N)ew project")
        if projects:
            print("  (1-9) Select project")
        print("  (E)xit")

        choice = input("\nChoice: ").strip().lower()

        if choice == 'n':
            await self._create_new_project()
        elif choice == 'e':
            exit(0)
        elif choice.isdigit() and 1 <= int(choice) <= len(projects):
            selected_project = projects[int(choice) - 1]
            await self._load_project(selected_project['project_id'])
        else:
            print(f"{Fore.YELLOW}Invalid choice{Style.RESET_ALL}")

    async def _create_new_project(self):
        """Create new project with enhanced setup"""
        print(f"\n{Fore.CYAN}🚀 Create New Project{Style.RESET_ALL}")

        name = input("Project name: ").strip()
        if not name:
            print(f"{Fore.RED}Project name is required{Style.RESET_ALL}")
            return

        result = await self.orchestrator.process_request('session', {
            'action': 'create_project',
            'project_name': name,
            'owner': self.current_session['user']['username']
        })

        if result['status'] == 'success':
            self.current_project = result['project']
            print(f"{Fore.GREEN}✓ Project '{name}' created successfully!{Style.RESET_ALL}")

            # Optional quick setup
            if input("\nWould you like to do quick setup? (y/n): ").strip().lower() == 'y':
                await self._quick_project_setup()
        else:
            print(f"{Fore.RED}Failed to create project: {result.get('message')}{Style.RESET_ALL}")

    async def _quick_project_setup(self):
        """Quick project configuration"""
        project = self.current_project

        print(f"{Fore.YELLOW}Quick Setup for {project['name']}{Style.RESET_ALL}")

        # Goals
        goals = input("Project goals (optional): ").strip()
        if goals:
            project['goals'] = goals

        # Tech stack
        tech = input("Primary technology (e.g., python, javascript): ").strip()
        if tech:
            project['tech_stack'] = [tech]

        # Save project
        await self.orchestrator.process_request('session', {
            'action': 'save_project',
            'project': project
        })

        print(f"{Fore.GREEN}✓ Quick setup completed{Style.RESET_ALL}\n")

    async def _load_project(self, project_id: str):
        """Load existing project"""
        result = await self.orchestrator.process_request('session', {
            'action': 'load_project',
            'project_id': project_id
        })

        if result['status'] == 'success':
            self.current_project = result['project']
            project = self.current_project
            print(f"{Fore.GREEN}✓ Loaded project: {project['name']}{Style.RESET_ALL}")

            # Show project summary
            print(f"  Phase: {self._get_phase_color(project['phase'])}{project['phase']}{Style.RESET_ALL}")
            if project.get('goals'):
                print(f"  Goals: {project['goals'][:100]}...")
            if project.get('tech_stack'):
                print(f"  Tech: {', '.join(project['tech_stack'][:3])}")
            print()
        else:
            print(f"{Fore.RED}Failed to load project{Style.RESET_ALL}")

    async def _handle_conversation(self):
        """Enhanced conversation handling"""
        project = self.current_project

        # Show conversation header
        self._print_conversation_header()

        # Handle conversation commands
        user_input = input(f"{Fore.BLUE}> {Style.RESET_ALL}").strip()

        if not user_input:
            return

        # Handle commands
        if user_input.startswith('/'):
            await self._handle_command(user_input)
            return

        # Process conversation turn
        await self._process_conversation_turn(user_input)

    def _print_conversation_header(self):
        """Print conversation status header"""
        project = self.current_project
        phase_color = self._get_phase_color(project['phase'])

        print(f"\n{Fore.CYAN}💬 {project['name']} - {phase_color}{project['phase'].title()} Phase{Style.RESET_ALL}")

        # Show progress
        conversation_count = len([msg for msg in project.get('conversation_history', [])
                                  if msg.get('type') == 'user'])
        print(f"Conversation turns: {conversation_count}")

        if conversation_count == 0:
            print(f"{Fore.YELLOW}💡 This is your first conversation in this project!{Style.RESET_ALL}")

        print("Commands: /help /save /phase /generate /exit")
        print()

    async def _handle_command(self, command: str):
        """Handle special commands"""
        cmd = command.lower().split()[0]

        if cmd == '/help':
            self._show_help()
        elif cmd == '/save':
            await self._save_project()
        elif cmd == '/phase':
            await self._manage_phases()
        elif cmd == '/generate':
            await self._generate_content()
        elif cmd == '/status':
            self._show_project_status()
        elif cmd == '/settings':
            await self._manage_settings()
        elif cmd == '/exit':
            if input("Save project before exit? (y/n): ").strip().lower() == 'y':
                await self._save_project()
            self.current_project = None
        else:
            print(f"{Fore.RED}Unknown command: {cmd}{Style.RESET_ALL}")

    def _show_help(self):
        """Show help information"""
        print(f"{Fore.CYAN}Available Commands:{Style.RESET_ALL}")
        commands = [
            "/help - Show this help",
            "/save - Save current project",
            "/phase - Manage project phases",
            "/generate - Generate code/docs",
            "/status - Show project status",
            "/settings - Manage settings",
            "/exit - Return to project selection"
        ]
        for cmd in commands:
            print(f"  {cmd}")
        print()

    async def _process_conversation_turn(self, user_input: str):
        """Process a complete conversation turn"""
        project = self.current_project

        # Generate question if needed
        if len(project.get('conversation_history', [])) == 0 or \
                project.get('conversation_history', [])[-1].get('type') == 'user':

            print(f"{Fore.YELLOW}🤔 Generating question...{Style.RESET_ALL}")

            question_result = await self.orchestrator.process_request('conversation', {
                'action': 'generate_question',
                'project': project,
                'use_dynamic': True
            })

            if question_result['status'] == 'success':
                question = question_result['question']
                cached = question_result.get('cached', False)
                cache_indicator = " 📋" if cached else ""

                print(f"\n{Fore.GREEN}🧠 Socratic Question{cache_indicator}:{Style.RESET_ALL}")
                print(f"{question}\n")

                # Get user response
                response = input(f"{Fore.BLUE}Your response: {Style.RESET_ALL}").strip()
                if not response:
                    return

                user_input = response

        # Process user response
        print(f"{Fore.YELLOW}🔍 Analyzing response...{Style.RESET_ALL}")

        response_result = await self.orchestrator.process_request('conversation', {
            'action': 'process_response',
            'project': project,
            'response': user_input,
            'current_user': self.current_session['user']['username']
        })

        if response_result['status'] == 'success':
            insights = response_result.get('insights', {})
            conflicts = response_result.get('conflicts', [])

            # Show insights
            if insights:
                print(f"{Fore.GREEN}✨ Insights extracted:{Style.RESET_ALL}")
                for key, value in insights.items():
                    if value:
                        if isinstance(value, list):
                            print(f"  • {key.title()}: {', '.join(value)}")
                        else:
                            print(f"  • {key.title()}: {value}")

            # Handle conflicts
            if conflicts:
                print(f"{Fore.RED}⚠️  Conflicts detected:{Style.RESET_ALL}")
                for conflict in conflicts:
                    print(f"  • {conflict}")

                if input("Resolve conflicts now? (y/n): ").strip().lower() == 'y':
                    await self._resolve_conflicts(conflicts)

            # Auto-save
            await self._save_project(silent=True)

            # Phase advancement check
            if self.auto_advance_phases:
                await self._check_phase_advancement()

    def _get_phase_color(self, phase: str) -> str:
        """Get color for phase display"""
        colors = {
            'discovery': Fore.BLUE,
            'analysis': Fore.YELLOW,
            'design': Fore.MAGENTA,
            'implementation': Fore.GREEN
        }
        return colors.get(phase, Fore.WHITE)

    async def _save_project(self, silent: bool = False):
        """Save current project"""
        if not self.current_project:
            return

        result = await self.orchestrator.process_request('session', {
            'action': 'save_project',
            'project': self.current_project
        })

        if not silent:
            if result['status'] == 'success':
                print(f"{Fore.GREEN}✓ Project saved{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Save failed: {result.get('message')}{Style.RESET_ALL}")

    async def _manage_phases(self):
        """Manage project phases"""
        project = self.current_project
        current_phase = project.get('phase', 'discovery')
        phases = ['discovery', 'analysis', 'design', 'implementation']

        print(f"{Fore.CYAN}Phase Management{Style.RESET_ALL}")
        print(f"Current phase: {self._get_phase_color(current_phase)}{current_phase}{Style.RESET_ALL}")

        print("\nAvailable phases:")
        for i, phase in enumerate(phases, 1):
            color = self._get_phase_color(phase)
            marker = "→" if phase == current_phase else " "
            print(f"  {marker} {i}. {color}{phase.title()}{Style.RESET_ALL}")

        choice = input("\nSelect phase (1-4) or press Enter to cancel: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(phases):
            new_phase = phases[int(choice) - 1]
            project['phase'] = new_phase
            print(f"{Fore.GREEN}✓ Phase changed to {new_phase}{Style.RESET_ALL}")
            await self._save_project(silent=True)

    async def _generate_content(self):
        """Generate project content"""
        print(f"{Fore.CYAN}Content Generation{Style.RESET_ALL}")
        print("1. Generate code")
        print("2. Generate documentation")
        print("3. Generate project report")

        choice = input("Choose option (1-3): ").strip()

        if choice == '1':
            await self._generate_code_content()
        elif choice == '2':
            await self._generate_docs_content()
        elif choice == '3':
            await self._generate_report_content()
        else:
            print(f"{Fore.YELLOW}Invalid choice{Style.RESET_ALL}")

    async def _generate_code_content(self):
        """Generate code for the project"""
        print(f"{Fore.YELLOW}🔧 Generating code...{Style.RESET_ALL}")

        result = await self.orchestrator.process_request('content', {
            'action': 'generate_code',
            'project': self.current_project
        })

        if result['status'] == 'success':
            code = result['code']
            cached = result.get('cached', False)
            cache_indicator = " (cached)" if cached else ""

            print(f"{Fore.GREEN}✓ Code generated{cache_indicator}:{Style.RESET_ALL}\n")
            print("=" * 60)
            print(code)
            print("=" * 60)
        else:
            print(f"{Fore.RED}Code generation failed: {result.get('message')}{Style.RESET_ALL}")

    async def _generate_docs_content(self):
        """Generate documentation"""
        print(f"{Fore.GREEN}📝 Documentation generation not yet implemented{Style.RESET_ALL}")

    async def _generate_report_content(self):
        """Generate project report"""
        print(f"{Fore.GREEN}📊 Report generation not yet implemented{Style.RESET_ALL}")

    def _show_project_status(self):
        """Show detailed project status"""
        project = self.current_project
        print(f"{Fore.CYAN}📊 Project Status: {project['name']}{Style.RESET_ALL}")

        print(f"  Phase: {self._get_phase_color(project['phase'])}{project['phase']}{Style.RESET_ALL}")
        print(f"  Owner: {project['owner']}")
        print(f"  Created: {project.get('created_at', 'Unknown')}")
        print(f"  Updated: {project.get('updated_at', 'Unknown')}")
        print(f"  Version: {project.get('version', 1)}")

        if project.get('goals'):
            print(f"  Goals: {project['goals']}")

        if project.get('tech_stack'):
            print(f"  Tech Stack: {', '.join(project['tech_stack'])}")

        if project.get('requirements'):
            print(f"  Requirements: {len(project['requirements'])} items")

        conversation_count = len([msg for msg in project.get('conversation_history', [])
                                  if msg.get('type') == 'user'])
        print(f"  Conversation Turns: {conversation_count}")

    async def _manage_settings(self):
        """Manage application settings"""
        print(f"{Fore.CYAN}⚙️  Settings{Style.RESET_ALL}")
        print(f"1. Show system info: {'✓' if self.show_system_info else '✗'}")
        print(f"2. Auto-advance phases: {'✓' if self.auto_advance_phases else '✗'}")
        print(f"3. Conversation limit: {self.conversation_limit}")

        choice = input("Toggle setting (1-3) or press Enter: ").strip()

        if choice == '1':
            self.show_system_info = not self.show_system_info
            print(f"System info display: {'enabled' if self.show_system_info else 'disabled'}")
        elif choice == '2':
            self.auto_advance_phases = not self.auto_advance_phases
            print(f"Auto-advance phases: {'enabled' if self.auto_advance_phases else 'disabled'}")
        elif choice == '3':
            try:
                new_limit = int(input("New conversation limit: "))
                self.conversation_limit = max(1, min(50, new_limit))
                print(f"Conversation limit set to: {self.conversation_limit}")
            except ValueError:
                print("Invalid number")

    async def _resolve_conflicts(self, conflicts):
        """Resolve project conflicts"""
        print(f"{Fore.YELLOW}Conflict resolution not yet implemented{Style.RESET_ALL}")

    async def _check_phase_advancement(self):
        """Check if project should advance to next phase"""
        project = self.current_project
        conversation_count = len([msg for msg in project.get('conversation_history', [])
                                  if msg.get('type') == 'user'])

        if conversation_count >= self.conversation_limit:
            result = await self.orchestrator.process_request('conversation', {
                'action': 'advance_phase',
                'project': project
            })

            if result['status'] == 'success':
                new_phase = result['new_phase']
                project['phase'] = new_phase
                print(f"{Fore.GREEN}🎉 Advanced to {new_phase} phase!{Style.RESET_ALL}")
                await self._save_project(silent=True)


if __name__ == "__main__":
    interface = SocraticInterface()
    asyncio.run(interface.run())
