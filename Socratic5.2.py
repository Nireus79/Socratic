#!/usr/bin/env python3
"""
Optimized Multi-Agent Socratic RAG System
A high-performance project development system with enhanced architecture and optimizations
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Protocol
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import anthropic
from datetime import datetime
import logging
from pathlib import Path
import sqlite3
import uuid
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
import aiofiles
import concurrent.futures
from threading import Lock
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_CONTEXT_LENGTH = 8000
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
BATCH_SIZE = 10
MAX_RETRIES = 3
CACHE_SIZE = 128


class ProjectPhase(Enum):
    """Project phases with clear transitions"""
    DISCOVERY = "discovery"
    PLANNING = "planning"
    GENERATION = "generation"
    VALIDATION = "validation"
    COMPLETE = "complete"

    def next_phase(self) -> Optional['ProjectPhase']:
        """Get the next phase in the workflow"""
        transitions = {
            self.DISCOVERY: self.PLANNING,
            self.PLANNING: self.GENERATION,
            self.GENERATION: self.VALIDATION,
            self.VALIDATION: self.COMPLETE,
            self.COMPLETE: None
        }
        return transitions.get(self)


@dataclass
class ProjectContext:
    """Optimized project context with validation and serialization"""
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

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Efficiently update context from dictionary"""
        for key, value in updates.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, list) and isinstance(value, list):
                    # Merge lists, avoiding duplicates
                    merged = current_value + [item for item in value if item not in current_value]
                    setattr(self, key, merged)
                else:
                    setattr(self, key, value)
        self.last_updated = datetime.now().isoformat()

    def to_summary(self) -> Dict[str, Any]:
        """Create a summary view of the context"""
        return {
            "requirements_count": len(self.functional_requirements),
            "tech_stack_count": len(self.technical_stack),
            "ui_components_count": len(self.ui_components),
            "has_architecture": bool(self.architecture_pattern),
            "has_deployment": bool(self.deployment_target),
            "completeness_score": self.completeness_score,
            "last_updated": self.last_updated
        }


@dataclass
class AgentResponse:
    """Optimized agent response with validation"""
    agent_id: str
    content: str
    context_updates: Dict[str, Any] = field(default_factory=dict)
    next_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    requires_followup: bool = False
    processing_time: float = 0.0

    def __post_init__(self):
        """Validate response data"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if len(self.next_questions) > 5:
            self.next_questions = self.next_questions[:5]


class AgentProtocol(Protocol):
    """Protocol for agent implementations"""
    agent_id: str
    expertise_areas: List[str]

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        """Process user input and return structured response"""
        ...


class DatabaseManager:
    """Optimized database manager with connection pooling and caching"""

    def __init__(self, db_path: str = "socratic_rag.db"):
        self.db_path = Path(db_path)
        self._lock = Lock()
        self._cache = {}
        self._init_database()

    def _init_database(self):
        """Initialize database with optimized schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")

            # Users table with indexes
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

            # Projects table with indexes
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_updated ON projects(updated_at)")

            # Conversation history with partitioning
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_project ON conversation_history(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_timestamp ON conversation_history(timestamp)")

            # Generated code with versioning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generated_code (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    file_hash TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_project_version ON generated_code(project_id, version)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
        finally:
            conn.close()

    @lru_cache(maxsize=CACHE_SIZE)
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username with caching"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()

            if row:
                return {
                    'user_id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'created_at': row[3],
                    'last_login': row[4],
                    'is_active': row[5]
                }
        return None

    def create_user(self, username: str, email: str) -> Dict[str, Any]:
        """Create a new user with validation"""
        user_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO users (user_id, username, email, created_at, last_login)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, username, email, now, now))
                conn.commit()

                # Clear cache
                self.get_user.cache_clear()

                return {
                    'user_id': user_id,
                    'username': username,
                    'email': email,
                    'created_at': now,
                    'last_login': now,
                    'is_active': True
                }
            except sqlite3.IntegrityError:
                raise ValueError(f"User with username '{username}' or email '{email}' already exists")

    def update_project_optimized(self, project_id: str, updates: Dict[str, Any]) -> None:
        """Optimized project update with selective updates"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build dynamic update query
            set_clauses = []
            values = []

            for key, value in updates.items():
                if key in ['name', 'description', 'current_phase', 'context_json', 'collaborators_json']:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)

            if set_clauses:
                set_clauses.append("updated_at = ?")
                values.append(datetime.now().isoformat())
                values.append(project_id)

                query = f"UPDATE projects SET {', '.join(set_clauses)} WHERE project_id = ?"
                cursor.execute(query, values)
                conn.commit()

    def batch_save_conversation(self, conversations: List[Dict[str, Any]]) -> None:
        """Batch save conversations for better performance"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO conversation_history (project_id, user_id, timestamp, user_input, system_response)
                VALUES (?, ?, ?, ?, ?)
            """, [(c['project_id'], c['user_id'], c['timestamp'], c['user_input'], c['system_response'])
                  for c in conversations])
            conn.commit()


class BaseAgent(ABC):
    """Optimized base agent with caching and performance improvements"""

    def __init__(self, agent_id: str, client: anthropic.Anthropic):
        self.agent_id = agent_id
        self.client = client
        self.expertise_areas: List[str] = []
        self._response_cache: Dict[str, str] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    @abstractmethod
    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        """Process user input and return structured response"""
        pass

    async def _generate_response_async(self, prompt: str, max_tokens: int = 1000) -> str:
        """Async wrapper for Claude API calls"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._generate_response, prompt, max_tokens)

    def _generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Claude with caching and error handling"""
        # Create cache key
        cache_key = f"{hash(prompt)}_{max_tokens}"

        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )

                result = response.content[0].text

                # Cache successful responses
                if len(self._response_cache) < CACHE_SIZE:
                    self._response_cache[cache_key] = result

                return result

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {self.agent_id}: {e}")
                if attempt == MAX_RETRIES - 1:
                    return f"I apologize, but I encountered an error processing your request after {MAX_RETRIES} attempts."
                asyncio.sleep(2 ** attempt)  # Exponential backoff

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from response with better error handling"""
        try:
            # Try to find JSON block
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from {self.agent_id} response")

        # Return safe default
        return {
            "analysis": response_text,
            "confidence": 0.5,
            "follow_up_questions": []
        }


class RequirementsAgent(BaseAgent):
    """Optimized requirements agent with structured processing"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("requirements_agent", client)
        self.expertise_areas = ["functional_requirements", "user_stories", "acceptance_criteria"]

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()

        prompt = self._build_prompt(user_input, context)
        response_text = await self._generate_response_async(prompt)
        response_data = self._extract_json_from_response(response_text)

        context_updates = {
            "functional_requirements": response_data.get("functional_requirements", []),
            "non_functional_requirements": response_data.get("non_functional_requirements", [])
        }

        processing_time = asyncio.get_event_loop().time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", ""),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5),
            processing_time=processing_time
        )

    def _build_prompt(self, user_input: str, context: ProjectContext) -> str:
        """Build optimized prompt with context awareness"""
        context_summary = context.to_summary()

        return f"""
        You are a Business Analysis Agent. Analyze this user input for requirements.

        User input: "{user_input}"

        Current context summary:
        - Requirements: {context_summary['requirements_count']} functional requirements
        - Completeness: {context_summary['completeness_score']:.2f}

        Extract:
        1. Functional requirements (specific features/capabilities)
        2. Non-functional requirements (performance, security, usability)
        3. 1-2 specific follow-up questions
        4. Confidence level (0-1)

        Respond with valid JSON:
        {{
            "functional_requirements": ["req1", "req2"],
            "non_functional_requirements": ["nfr1"],
            "follow_up_questions": ["question1", "question2"],
            "confidence": 0.8,
            "analysis": "Brief analysis"
        }}
        """


class TechnicalAgent(BaseAgent):
    """Optimized technical agent with architecture focus"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("technical_agent", client)
        self.expertise_areas = ["technical_stack", "architecture", "database", "apis"]

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()

        prompt = self._build_technical_prompt(user_input, context)
        response_text = await self._generate_response_async(prompt)
        response_data = self._extract_json_from_response(response_text)

        context_updates = {
            "technical_stack": response_data.get("technical_stack", []),
            "architecture_pattern": response_data.get("architecture_pattern", ""),
            "database_requirements": response_data.get("database_requirements", []),
            "api_specifications": response_data.get("api_specifications", [])
        }

        processing_time = asyncio.get_event_loop().time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", ""),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5),
            processing_time=processing_time
        )

    def _build_technical_prompt(self, user_input: str, context: ProjectContext) -> str:
        """Build technical analysis prompt"""
        return f"""
        You are a Technical Architecture Agent. Analyze for technical decisions.

        User input: "{user_input}"

        Current technical context:
        - Stack: {', '.join(context.technical_stack[:3])}
        - Architecture: {context.architecture_pattern or 'Not specified'}

        Identify:
        1. Technology preferences (languages, frameworks, databases)
        2. Architecture patterns (MVC, microservices, etc.)
        3. Database and API requirements
        4. Technical questions to clarify

        Respond with valid JSON:
        {{
            "technical_stack": ["tech1", "tech2"],
            "architecture_pattern": "pattern",
            "database_requirements": ["req1"],
            "api_specifications": ["spec1"],
            "follow_up_questions": ["question1"],
            "confidence": 0.8,
            "analysis": "Technical analysis"
        }}
        """


class OptimizedOrchestrator:
    """Optimized orchestrator with improved performance and error handling"""

    def __init__(self, api_key: str, db_path: str = "socratic_rag.db"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.db_manager = DatabaseManager(db_path)

        # Initialize agents
        self.agents = {
            "requirements": RequirementsAgent(self.client),
            "technical": TechnicalAgent(self.client),
        }

        self.current_user = None
        self.current_project = None
        self._conversation_buffer = []

    async def authenticate_user(self, username: str, email: str = None) -> bool:
        """Authenticate user with caching"""
        user = self.db_manager.get_user(username)

        if user:
            self.current_user = user
            return True
        elif email:
            try:
                self.current_user = self.db_manager.create_user(username, email)
                return True
            except ValueError:
                return False
        return False

    async def process_user_input_optimized(self, user_input: str) -> str:
        """Optimized input processing with parallel agent execution"""
        if not self.current_user or not self.current_project:
            return "Please authenticate and select a project first."

        current_phase = ProjectPhase(self.current_project.get('current_phase', 'discovery'))

        if current_phase == ProjectPhase.DISCOVERY:
            return await self._handle_discovery_optimized(user_input)
        elif current_phase == ProjectPhase.PLANNING:
            return await self._handle_planning_optimized(user_input)
        else:
            return f"Current phase: {current_phase.value}. Use appropriate commands."

    async def _handle_discovery_optimized(self, user_input: str) -> str:
        """Optimized discovery phase with parallel processing"""
        # Create context object
        context = ProjectContext(**json.loads(self.current_project.get('context_json', '{}')))

        # Process with relevant agents in parallel
        agent_names = ["requirements", "technical"]
        tasks = []

        for agent_name in agent_names:
            if agent_name in self.agents:
                task = self.agents[agent_name].process_input(user_input, context)
                tasks.append((agent_name, task))

        # Execute agents concurrently
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        responses = []
        all_questions = []
        context_updates = {}

        for (agent_name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_name} failed: {result}")
                continue

            responses.append(f"**{agent_name.title()}**: {result.content}")
            all_questions.extend(result.next_questions)
            context_updates.update(result.context_updates)

        # Update context efficiently
        context.update_from_dict(context_updates)

        # Calculate completeness
        completeness_score = self._calculate_completeness_optimized(context)
        context.completeness_score = completeness_score

        # Check for phase transition
        if completeness_score > DEFAULT_CONFIDENCE_THRESHOLD:
            responses.append("\n**🎯 Ready for Planning Phase!**")
            next_phase = ProjectPhase.PLANNING.value
        else:
            next_phase = ProjectPhase.DISCOVERY.value

        # Update project
        self.db_manager.update_project_optimized(self.current_project['project_id'], {
            'current_phase': next_phase,
            'context_json': json.dumps(asdict(context))
        })

        # Build response
        response_text = "\n\n".join(responses)
        if all_questions:
            response_text += "\n\n**Next Questions:**\n" + "\n".join(f"• {q}" for q in all_questions[:3])

        # Buffer conversation for batch saving
        self._conversation_buffer.append({
            'project_id': self.current_project['project_id'],
            'user_id': self.current_user['user_id'],
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'system_response': response_text
        })

        # Batch save if buffer is full
        if len(self._conversation_buffer) >= BATCH_SIZE:
            self.db_manager.batch_save_conversation(self._conversation_buffer)
            self._conversation_buffer.clear()

        return response_text

    async def _handle_planning_optimized(self, user_input: str) -> str:
        """Optimized planning phase"""
        if user_input.lower().strip() == "plan":
            return "Planning phase implementation - create project specification"
        else:
            return "In planning phase. Use 'plan' to create the project specification."

    def _calculate_completeness_optimized(self, context: ProjectContext) -> float:
        """Optimized completeness calculation"""
        weights = {
            'functional_requirements': 0.4,
            'technical_stack': 0.3,
            'ui_components': 0.2,
            'deployment_target': 0.1
        }

        scores = {}

        # Requirements score
        scores['functional_requirements'] = min(len(context.functional_requirements) / 5, 1.0)

        # Technical score
        tech_score = min(len(context.technical_stack) / 3, 1.0)
        if context.architecture_pattern:
            tech_score = min(tech_score + 0.3, 1.0)
        scores['technical_stack'] = tech_score

        # UI score
        scores['ui_components'] = min(len(context.ui_components) / 3, 1.0)

        # Deployment score
        scores['deployment_target'] = 1.0 if context.deployment_target else 0.0

        # Weighted average
        total_score = sum(scores[key] * weights[key] for key in weights)
        return total_score

    async def get_project_status_optimized(self) -> Dict[str, Any]:
        """Get optimized project status"""
        if not self.current_project:
            return {"error": "No project selected"}

        context = ProjectContext(**json.loads(self.current_project.get('context_json', '{}')))

        return {
            "project_id": self.current_project['project_id'],
            "name": self.current_project['name'],
            "phase": self.current_project['current_phase'],
            "completeness_score": self._calculate_completeness_optimized(context),
            "context_summary": context.to_summary(),
            "last_updated": context.last_updated
        }

    async def cleanup(self):
        """Cleanup resources"""
        # Save any remaining conversations
        if self._conversation_buffer:
            self.db_manager.batch_save_conversation(self._conversation_buffer)
            self._conversation_buffer.clear()

        # Close agent executors
        for agent in self.agents.values():
            if hasattr(agent, '_executor'):
                agent._executor.shutdown(wait=True)


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""

    async def wrapper(*args, **kwargs):
        start_time = asyncio.get_event_loop().time()
        try:
            result = await func(*args, **kwargs)
            end_time = asyncio.get_event_loop().time()
            logger.info(f"{func.__name__} completed in {end_time - start_time:.2f}s")
            return result
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            logger.error(f"{func.__name__} failed after {end_time - start_time:.2f}s: {e}")
            raise

    return wrapper


# Example usage
async def main():
    """Optimized main function with proper resource management"""
    orchestrator = OptimizedOrchestrator(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        db_path="optimized_socratic_rag.db"
    )

    try:
        print("🚀 Optimized Socratic RAG System")
        print("=" * 50)

        # Simple authentication for demo
        if await orchestrator.authenticate_user("demo_user", "demo@example.com"):
            print("✅ Authenticated successfully")
        else:
            print("❌ Authentication failed")
            return

        # Create demo project
        orchestrator.current_project = {
            'project_id': 'demo-project',
            'name': 'Demo Project',
            'current_phase': 'discovery',
            'context_json': '{}'
        }

        # Demo interaction
        print("\n🤖 System ready for optimized processing...")

        # Process sample input
        response = await orchestrator.process_user_input_optimized(
            "I want to build a web application for task management with user authentication"
        )
        print(f"\n📝 Response: {response}")

        # Show status
        status = await orchestrator.get_project_status_optimized()
        print(f"\n📊 Status: {status}")

    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())