#!/usr/bin/env python3
"""
Optimized Multi-Agent Socratic RAG System
A high-performance project development system using specialized agents for discovery, planning, and code generation

Key Optimizations:
- Improved async handling and concurrency
- Better error handling and resilience
- Enhanced agent communication and coordination
- Optimized context management
- Performance improvements and caching
- Better code organization and maintainability
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import anthropic
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
from functools import lru_cache, wraps
import weakref

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('socratic_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProjectPhase(Enum):
    """Enhanced project phases with better state management"""
    DISCOVERY = "discovery"
    PLANNING = "planning"
    GENERATION = "generation"
    VALIDATION = "validation"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ProjectContext:
    """Optimized project context with validation and caching"""
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

    # Optimization fields
    _hash: Optional[str] = field(default=None, init=False)
    _validation_cache: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._update_hash()
        self._validate_data()

    def _update_hash(self):
        """Update context hash for change detection"""
        content = json.dumps(asdict(self), sort_keys=True, default=str)
        self._hash = hashlib.md5(content.encode()).hexdigest()

    def _validate_data(self):
        """Validate context data integrity"""
        # Remove duplicates from lists
        for field_name in ['goals_requirements', 'functional_requirements',
                           'non_functional_requirements', 'technical_stack',
                           'database_requirements', 'api_specifications',
                           'ui_components', 'user_personas', 'user_flows',
                           'scalability_requirements', 'security_requirements']:
            field_value = getattr(self, field_name)
            if isinstance(field_value, list):
                setattr(self, field_name, list(dict.fromkeys(field_value)))  # Preserve order

        # Validate confidence scores
        self.confidence_scores = {k: max(0.0, min(1.0, v))
                                  for k, v in self.confidence_scores.items()}

    def has_changed(self, other_hash: str) -> bool:
        """Check if context has changed since last hash"""
        return self._hash != other_hash

    def get_completion_metrics(self) -> Dict[str, float]:
        """Get detailed completion metrics"""
        metrics = {
            'goals_completion': min(1.0, len(self.goals_requirements) / 3),
            'functional_completion': min(1.0, len(self.functional_requirements) / 5),
            'technical_completion': min(1.0, len(self.technical_stack) / 3),
            'architecture_completion': 1.0 if self.architecture_pattern else 0.0,
            'database_completion': min(1.0, len(self.database_requirements) / 2),
            'api_completion': min(1.0, len(self.api_specifications) / 3),
            'ui_completion': min(1.0, len(self.ui_components) / 3),
            'deployment_completion': 1.0 if self.deployment_target else 0.0
        }
        return metrics


@dataclass
class AgentResponse:
    """Enhanced agent response with better metadata"""
    agent_id: str
    content: str
    context_updates: Dict[str, Any]
    next_questions: List[str]
    confidence: float
    processing_time: float = 0.0
    requires_followup: bool = False
    priority: int = 1  # 1=high, 2=medium, 3=low
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectSpecification:
    """Enhanced project specification with validation"""
    project_name: str
    description: str
    technical_architecture: Dict[str, Any]
    database_schema: Dict[str, Any]
    api_endpoints: List[Dict[str, Any]]
    ui_components: List[Dict[str, Any]]
    deployment_config: Dict[str, Any]
    testing_strategy: Dict[str, Any]
    estimated_complexity: str = "medium"
    estimated_timeline: str = "4-6 weeks"

    def validate(self) -> List[str]:
        """Validate specification completeness"""
        issues = []
        if not self.project_name:
            issues.append("Project name is required")
        if not self.description:
            issues.append("Project description is required")
        if not self.technical_architecture:
            issues.append("Technical architecture is required")
        return issues


def timing_decorator(func):
    """Decorator to measure function execution time"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retry logic with exponential backoff"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(delay * (2 ** attempt))
                    logger.warning(f"Retry {attempt + 1} for {func.__name__}: {e}")
            return None

        return wrapper

    return decorator


class BaseAgent(ABC):
    """Enhanced base class for all agents with better error handling"""

    def __init__(self, agent_id: str, client: anthropic.Anthropic):
        self.agent_id = agent_id
        self.client = client
        self.expertise_areas = []
        self.conversation_history = []
        self._response_cache = {}
        self._last_context_hash = None
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'average_response_time': 0.0,
            'error_count': 0
        }

    @abstractmethod
    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        """Process user input and return structured response"""
        pass

    @retry_on_error(max_retries=3, delay=1.0)
    async def _generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Enhanced response generation with caching and error handling"""
        start_time = time.time()

        # Check cache first
        cache_key = hashlib.md5(f"{prompt}_{max_tokens}".encode()).hexdigest()
        if cache_key in self._response_cache:
            logger.debug(f"Cache hit for {self.agent_id}")
            return self._response_cache[cache_key]

        try:
            self.performance_metrics['total_requests'] += 1

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text

            # Cache the response
            self._response_cache[cache_key] = result

            # Update metrics
            self.performance_metrics['successful_requests'] += 1
            response_time = time.time() - start_time
            self.performance_metrics['average_response_time'] = (
                    (self.performance_metrics['average_response_time'] *
                     (self.performance_metrics['successful_requests'] - 1) + response_time) /
                    self.performance_metrics['successful_requests']
            )

            return result

        except Exception as e:
            self.performance_metrics['error_count'] += 1
            logger.error(f"Error generating response for {self.agent_id}: {e}")
            raise

    @lru_cache(maxsize=100)
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Cached JSON parsing with better error handling"""
        try:
            import re
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Return default structure if no JSON found
                return self._get_default_response_structure()
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for {self.agent_id}")
            return self._get_default_response_structure()

    def _get_default_response_structure(self) -> Dict[str, Any]:
        """Get default response structure for each agent type"""
        return {
            "analysis": "Unable to parse response",
            "follow_up_questions": [],
            "confidence": 0.5
        }

    def clear_cache(self):
        """Clear response cache"""
        self._response_cache.clear()
        self._parse_json_response.cache_clear()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return self.performance_metrics.copy()


class RequirementsAgent(BaseAgent):
    """Optimized requirements agent with better analysis"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("requirements_agent", client)
        self.expertise_areas = ["functional_requirements", "user_stories", "acceptance_criteria"]

    @timing_decorator
    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        start_time = time.time()

        # Check if we need to reprocess based on context changes
        if context.has_changed(self._last_context_hash):
            self.clear_cache()
            self._last_context_hash = context._hash

        prompt = f"""
        You are a Senior Business Analysis Agent specializing in requirements gathering and user story creation.

        User input: "{user_input}"

        Current requirements context:
        - Goals: {', '.join(context.goals_requirements)}
        - Functional: {', '.join(context.functional_requirements)}
        - Non-functional: {', '.join(context.non_functional_requirements)}

        Analysis tasks:
        1. Extract functional requirements (what the system should do)
        2. Identify non-functional requirements (performance, security, usability, scalability)
        3. Generate specific, actionable follow-up questions
        4. Assess confidence in requirement completeness (0-1)
        5. Identify any missing critical requirements

        Provide a comprehensive JSON response:
        {{
            "functional_requirements": ["specific requirement 1", "specific requirement 2"],
            "non_functional_requirements": ["performance requirement", "security requirement"],
            "follow_up_questions": ["What is the expected user load?", "Are there any compliance requirements?"],
            "confidence": 0.85,
            "analysis": "Detailed analysis of requirements extracted",
            "missing_areas": ["area1", "area2"],
            "priority_level": "high"
        }}
        """

        response_text = await self._generate_response(prompt)
        response_data = self._parse_json_response(response_text)

        # Enhanced context updates with validation
        context_updates = {
            "functional_requirements": [req for req in response_data.get("functional_requirements", [])
                                        if req and len(req.strip()) > 10],
            "non_functional_requirements": [req for req in response_data.get("non_functional_requirements", [])
                                            if req and len(req.strip()) > 10]
        }

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", "Requirements analysis completed"),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5),
            processing_time=processing_time,
            priority=1,  # High priority for requirements
            tags=["requirements", "business_analysis"],
            metadata={
                "missing_areas": response_data.get("missing_areas", []),
                "priority_level": response_data.get("priority_level", "medium")
            }
        )


class TechnicalAgent(BaseAgent):
    """Enhanced technical agent with architecture expertise"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("technical_agent", client)
        self.expertise_areas = ["technical_stack", "architecture", "database", "apis"]
        self.tech_stack_templates = {
            "web_app": ["React", "Node.js", "Express", "PostgreSQL"],
            "mobile_app": ["React Native", "Node.js", "MongoDB"],
            "data_science": ["Python", "FastAPI", "PostgreSQL", "Redis"],
            "enterprise": ["Java", "Spring Boot", "PostgreSQL", "Docker"]
        }

    @timing_decorator
    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        start_time = time.time()

        # Detect project type for better recommendations
        project_type = self._detect_project_type(user_input, context)

        prompt = f"""
        You are a Senior Technical Architecture Agent with expertise in system design and technology selection.

        User input: "{user_input}"
        Detected project type: {project_type}

        Current technical context:
        - Tech Stack: {', '.join(context.technical_stack)}
        - Architecture: {context.architecture_pattern}
        - Database: {', '.join(context.database_requirements)}
        - APIs: {', '.join(context.api_specifications)}

        Technical analysis tasks:
        1. Recommend appropriate technology stack based on requirements
        2. Suggest suitable architecture patterns (microservices, monolith, serverless, etc.)
        3. Design database schema and requirements
        4. Specify API architecture and endpoints
        5. Consider scalability and performance implications
        6. Assess technical feasibility and complexity

        Provide detailed JSON response:
        {{
            "technical_stack": ["technology1 with version", "technology2 with rationale"],
            "architecture_pattern": "pattern with detailed explanation",
            "database_requirements": ["schema requirement", "performance requirement"],
            "api_specifications": ["REST endpoints", "GraphQL schema"],
            "follow_up_questions": ["What is expected traffic volume?", "Any existing system integrations?"],
            "confidence": 0.9,
            "analysis": "Technical architecture analysis and recommendations",
            "complexity_assessment": "medium",
            "estimated_development_time": "4-6 weeks"
        }}
        """

        response_text = await self._generate_response(prompt, max_tokens=1500)
        response_data = self._parse_json_response(response_text)

        # Enhanced validation and processing
        context_updates = {
            "technical_stack": self._validate_tech_stack(response_data.get("technical_stack", [])),
            "architecture_pattern": response_data.get("architecture_pattern", ""),
            "database_requirements": response_data.get("database_requirements", []),
            "api_specifications": response_data.get("api_specifications", [])
        }

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", "Technical analysis completed"),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5),
            processing_time=processing_time,
            priority=1,
            tags=["technical", "architecture"],
            metadata={
                "complexity_assessment": response_data.get("complexity_assessment", "medium"),
                "estimated_development_time": response_data.get("estimated_development_time", "unknown"),
                "project_type": project_type
            }
        )

    def _detect_project_type(self, user_input: str, context: ProjectContext) -> str:
        """Detect project type based on input and context"""
        text = (user_input + " " + " ".join(context.goals_requirements)).lower()

        if any(keyword in text for keyword in ["mobile", "ios", "android", "app store"]):
            return "mobile_app"
        elif any(keyword in text for keyword in ["data", "analytics", "ml", "ai", "machine learning"]):
            return "data_science"
        elif any(keyword in text for keyword in ["enterprise", "corporate", "business", "crm", "erp"]):
            return "enterprise"
        else:
            return "web_app"

    def _validate_tech_stack(self, tech_stack: List[str]) -> List[str]:
        """Validate and enhance technology stack recommendations"""
        validated = []
        for tech in tech_stack:
            if tech and len(tech.strip()) > 2:
                validated.append(tech.strip())
        return validated


class UXAgent(BaseAgent):
    """Enhanced UX agent with modern design patterns"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("ux_agent", client)
        self.expertise_areas = ["ui_components", "user_flows", "user_personas", "design_systems"]

    @timing_decorator
    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        start_time = time.time()

        prompt = f"""
        You are a Senior UX/UI Designer Agent specializing in user experience design and interface architecture.

        User input: "{user_input}"

        Current UX context:
        - UI Components: {', '.join(context.ui_components)}
        - User Personas: {', '.join(context.user_personas)}
        - User Flows: {', '.join(context.user_flows)}
        - Functional Requirements: {', '.join(context.functional_requirements)}

        UX Design tasks:
        1. Identify key UI components and design patterns
        2. Create detailed user personas based on requirements
        3. Map critical user flows and interactions
        4. Consider accessibility and usability requirements
        5. Recommend design system and component library
        6. Assess user experience complexity

        Provide comprehensive JSON response:
        {{
            "ui_components": ["component with description", "component with props"],
            "user_personas": ["detailed persona 1", "detailed persona 2"],
            "user_flows": ["flow with steps", "flow with decision points"],
            "accessibility_requirements": ["WCAG compliance", "keyboard navigation"],
            "design_system": "Design system recommendations",
            "follow_up_questions": ["What devices will users primarily use?", "Are there branding guidelines?"],
            "confidence": 0.8,
            "analysis": "UX analysis and design recommendations",
            "complexity_level": "medium"
        }}
        """

        response_text = await self._generate_response(prompt, max_tokens=1500)
        response_data = self._parse_json_response(response_text)

        context_updates = {
            "ui_components": response_data.get("ui_components", []),
            "user_personas": response_data.get("user_personas", []),
            "user_flows": response_data.get("user_flows", [])
        }

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", "UX analysis completed"),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5),
            processing_time=processing_time,
            priority=2,
            tags=["ux", "design", "user_experience"],
            metadata={
                "accessibility_requirements": response_data.get("accessibility_requirements", []),
                "design_system": response_data.get("design_system", ""),
                "complexity_level": response_data.get("complexity_level", "medium")
            }
        )


class InfrastructureAgent(BaseAgent):
    """Enhanced infrastructure agent with DevOps expertise"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("infrastructure_agent", client)
        self.expertise_areas = ["deployment", "scalability", "security", "monitoring"]

    @timing_decorator
    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        start_time = time.time()

        prompt = f"""
        You are a Senior DevOps and Infrastructure Agent specializing in deployment, scalability, and security.

        User input: "{user_input}"

        Current infrastructure context:
        - Deployment: {context.deployment_target}
        - Scalability: {', '.join(context.scalability_requirements)}
        - Security: {', '.join(context.security_requirements)}
        - Technical Stack: {', '.join(context.technical_stack)}

        Infrastructure analysis tasks:
        1. Recommend deployment strategy and platforms
        2. Design scalability architecture
        3. Identify security requirements and best practices
        4. Suggest monitoring and logging solutions
        5. Estimate infrastructure costs
        6. Plan CI/CD pipeline

        Provide detailed JSON response:
        {{
            "deployment_target": "detailed deployment strategy",
            "scalability_requirements": ["auto-scaling", "load balancing"],
            "security_requirements": ["authentication", "data encryption"],
            "monitoring_strategy": ["metrics", "logging", "alerting"],
            "ci_cd_pipeline": "CI/CD recommendations",
            "cost_estimation": "Infrastructure cost estimate",
            "follow_up_questions": ["What is the expected user base?", "Any compliance requirements?"],
            "confidence": 0.85,
            "analysis": "Infrastructure analysis and recommendations"
        }}
        """

        response_text = await self._generate_response(prompt, max_tokens=1500)
        response_data = self._parse_json_response(response_text)

        context_updates = {
            "deployment_target": response_data.get("deployment_target", ""),
            "scalability_requirements": response_data.get("scalability_requirements", []),
            "security_requirements": response_data.get("security_requirements", [])
        }

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", "Infrastructure analysis completed"),
            context_updates=context_updates,
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5),
            processing_time=processing_time,
            priority=2,
            tags=["infrastructure", "devops", "security"],
            metadata={
                "monitoring_strategy": response_data.get("monitoring_strategy", []),
                "ci_cd_pipeline": response_data.get("ci_cd_pipeline", ""),
                "cost_estimation": response_data.get("cost_estimation", "")
            }
        )


class PlanningAgent(BaseAgent):
    """Enhanced planning agent with better specification generation"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("planning_agent", client)
        self.expertise_areas = ["project_planning", "specifications", "architecture_design"]

    @timing_decorator
    async def create_project_specification(self, context: ProjectContext) -> ProjectSpecification:
        """Create enhanced project specification from context"""
        completion_metrics = context.get_completion_metrics()

        prompt = f"""
        Create a comprehensive, production-ready project specification based on this context:

        PROJECT CONTEXT:
        Goals: {', '.join(context.goals_requirements)}
        Functional Requirements: {', '.join(context.functional_requirements)}
        Non-functional Requirements: {', '.join(context.non_functional_requirements)}

        Technical Stack: {', '.join(context.technical_stack)}
        Architecture: {context.architecture_pattern}
        Database: {', '.join(context.database_requirements)}
        APIs: {', '.join(context.api_specifications)}

        UI Components: {', '.join(context.ui_components)}
        User Personas: {', '.join(context.user_personas)}
        User Flows: {', '.join(context.user_flows)}

        Deployment: {context.deployment_target}
        Scalability: {', '.join(context.scalability_requirements)}
        Security: {', '.join(context.security_requirements)}

        Completion Metrics: {completion_metrics}

        Create a detailed specification including:
        1. Executive summary and project overview
        2. Technical architecture with detailed component design
        3. Database schema with relationships and constraints
        4. API specification with endpoints, methods, and data models
        5. UI/UX specification with component hierarchy
        6. Deployment and infrastructure requirements
        7. Security and compliance considerations
        8. Testing strategy and quality assurance
        9. Development timeline and milestones
        10. Risk assessment and mitigation strategies

        Return comprehensive JSON specification with all technical details.
        """

        response_text = await self._generate_response(prompt, max_tokens=4000)

        # Extract project name from context
        project_name = self._extract_project_name(context)
        description = self._generate_project_description(context)

        return ProjectSpecification(
            project_name=project_name,
            description=description,
            technical_architecture={
                "stack": context.technical_stack,
                "pattern": context.architecture_pattern,
                "detailed_specification": response_text,
                "completion_metrics": completion_metrics
            },
            database_schema={
                "requirements": context.database_requirements,
                "schema_design": self._extract_schema_from_response(response_text)
            },
            api_endpoints=self._extract_api_endpoints(response_text, context),
            ui_components=self._extract_ui_components(response_text, context),
            deployment_config={
                "target": context.deployment_target,
                "scalability": context.scalability_requirements,
                "security": context.security_requirements
            },
            testing_strategy={
                "approach": "comprehensive testing strategy",
                "levels": ["unit", "integration", "end-to-end", "performance"]
            },
            estimated_complexity=self._assess_complexity(context),
            estimated_timeline=self._estimate_timeline(context)
        )

    def _extract_project_name(self, context: ProjectContext) -> str:
        """Extract or generate project name from context"""
        # Simple name extraction logic
        if context.goals_requirements:
            first_goal = context.goals_requirements[0]
            words = first_goal.split()[:3]
            return " ".join(words).title() + " Project"
        return "Generated Project"

    def _generate_project_description(self, context: ProjectContext) -> str:
        """Generate comprehensive project description"""
        parts = []
        if context.goals_requirements:
            parts.append(f"Goals: {'; '.join(context.goals_requirements[:2])}")
        if context.technical_stack:
            parts.append(f"Built with: {', '.join(context.technical_stack[:3])}")
        if context.deployment_target:
            parts.append(f"Deployed on: {context.deployment_target}")

        return ". ".join(parts) if parts else "A comprehensive software project"

    def _extract_schema_from_response(self, response: str) -> Dict[str, Any]:
        """Extract database schema details from response"""
        # Simplified schema extraction
        return {"tables": [], "relationships": [], "indexes": []}

    def _extract_api_endpoints(self, response: str, context: ProjectContext) -> List[Dict[str, Any]]:
        """Extract API endpoints from response and context"""
        endpoints = []
        for spec in context.api_specifications:
            endpoints.append({
                "path": "/api/endpoint",
                "method": "GET",
                "description": spec,
                "parameters": [],
                "responses": {}
            })
        return endpoints

    def _extract_ui_components(self, response: str, context: ProjectContext) -> List[Dict[str, Any]]:
        """Extract UI components from response and context"""
        components = []
        for component in context.ui_components:
            components.append({
                "name": component,
                "type": "component",
                "props": [],
                "description": component
            })
        return components

    def _assess_complexity(self, context: ProjectContext) -> str:
        """Assess project complexity based on context"""
        complexity_score = 0

        complexity_score += len(context.functional_requirements) * 0.5
        complexity_score += len(context.technical_stack) * 0.3
        complexity_score += len(context.api_specifications) * 0.4
        complexity_score += len(context.ui_components) * 0.2

        if complexity_score < 5:
            return "low"
        elif complexity_score < 10:
            return "medium"
        else:
            return "high"

    def _estimate_timeline(self, context: ProjectContext) -> str:
        """Estimate project timeline based on complexity"""
        complexity = self._assess_complexity(context)

        base_weeks = {
            "low": 2,
            "medium": 4,
            "high": 8
        }

        # Adjust based on specific requirements
        weeks = base_weeks.get(complexity, 4)

        # Add time for additional complexity factors
        if len(context.scalability_requirements) > 2:
            weeks += 2
        elif len(context.security_requirements) > 3:
            weeks += 1
        elif context.deployment_target and "enterprise" in context.deployment_target.lower():
            weeks += 1

        return f"{weeks}-{weeks + 2} weeks"

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        """Process planning input and generate comprehensive response"""
        start_time = time.time()

        prompt = f"""
            You are a Senior Project Planning Agent specializing in software project specification and architecture design.

            User input: "{user_input}"

            Current project context completion:
            - Goals: {len(context.goals_requirements)} items
            - Functional Requirements: {len(context.functional_requirements)} items
            - Technical Stack: {len(context.technical_stack)} items
            - Architecture: {'✓' if context.architecture_pattern else '✗'}
            - Database: {len(context.database_requirements)} items
            - APIs: {len(context.api_specifications)} items
            - UI Components: {len(context.ui_components)} items
            - Deployment: {'✓' if context.deployment_target else '✗'}

            Planning tasks:
            1. Assess project readiness for implementation
            2. Identify any missing critical specifications
            3. Generate detailed project timeline
            4. Estimate resource requirements
            5. Identify potential risks and mitigation strategies
            6. Create implementation roadmap

            Provide detailed JSON response:
            {{
                "readiness_assessment": "detailed assessment of project readiness",
                "missing_specifications": ["missing item 1", "missing item 2"],
                "timeline_estimate": "detailed timeline with phases",
                "resource_requirements": ["developer", "designer", "devops"],
                "risk_assessment": ["risk 1", "risk 2"],
                "implementation_roadmap": ["phase 1", "phase 2", "phase 3"],
                "follow_up_questions": ["question 1", "question 2"],
                "confidence": 0.9,
                "analysis": "Comprehensive planning analysis"
            }}
            """

        response_text = await self._generate_response(prompt, max_tokens=2000)
        response_data = self._parse_json_response(response_text)

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", "Planning analysis completed"),
            context_updates={},  # Planning agent doesn't update context directly
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5),
            processing_time=processing_time,
            priority=1,
            tags=["planning", "specification", "roadmap"],
            metadata={
                "readiness_assessment": response_data.get("readiness_assessment", ""),
                "missing_specifications": response_data.get("missing_specifications", []),
                "timeline_estimate": response_data.get("timeline_estimate", ""),
                "resource_requirements": response_data.get("resource_requirements", []),
                "risk_assessment": response_data.get("risk_assessment", []),
                "implementation_roadmap": response_data.get("implementation_roadmap", [])
            }
        )


class CodeGenerationAgent(BaseAgent):
    """Enhanced code generation agent with better output quality"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("code_generation_agent", client)
        self.expertise_areas = ["code_generation", "implementation", "best_practices"]

    @timing_decorator
    async def generate_code(self, specification: ProjectSpecification) -> Dict[str, str]:
        """Generate code based on project specification"""

        # Generate different types of code based on specification
        code_modules = {}

        # Backend code generation
        if any(tech in specification.technical_architecture.get("stack", [])
               for tech in ["Node.js", "Python", "Java", "C#"]):
            code_modules["backend"] = await self._generate_backend_code(specification)

        # Frontend code generation
        if any(tech in specification.technical_architecture.get("stack", [])
               for tech in ["React", "Vue.js", "Angular", "HTML"]):
            code_modules["frontend"] = await self._generate_frontend_code(specification)

        # Database code generation
        if specification.database_schema.get("requirements"):
            code_modules["database"] = await self._generate_database_code(specification)

        # API code generation
        if specification.api_endpoints:
            code_modules["api"] = await self._generate_api_code(specification)

        # Configuration files
        code_modules["config"] = await self._generate_config_files(specification)

        # Tests
        code_modules["tests"] = await self._generate_test_code(specification)

        return code_modules

    async def _generate_backend_code(self, spec: ProjectSpecification) -> str:
        """Generate backend code based on specification"""
        tech_stack = spec.technical_architecture.get("stack", [])

        if "Node.js" in tech_stack:
            return await self._generate_nodejs_backend(spec)
        elif "Python" in tech_stack:
            return await self._generate_python_backend(spec)
        elif "Java" in tech_stack:
            return await self._generate_java_backend(spec)
        else:
            return "// Backend code generation not implemented for this stack"

    async def _generate_nodejs_backend(self, spec: ProjectSpecification) -> str:
        """Generate Node.js backend code"""
        prompt = f"""
            Generate a complete Node.js backend application based on this specification:

            Project: {spec.project_name}
            Description: {spec.description}
            Architecture: {spec.technical_architecture.get('pattern', 'MVC')}
            Database: {spec.database_schema.get('requirements', [])}
            API Endpoints: {spec.api_endpoints}

            Generate:
            1. Main server file with Express setup
            2. Route handlers for all endpoints
            3. Database models and connections
            4. Middleware for authentication and validation
            5. Error handling
            6. Configuration management

            Use modern Node.js practices with async/await, proper error handling, and security best practices.
            """

        return await self._generate_response(prompt, max_tokens=3000)

    async def _generate_python_backend(self, spec: ProjectSpecification) -> str:
        """Generate Python backend code"""
        prompt = f"""
            Generate a complete Python backend application using FastAPI based on this specification:

            Project: {spec.project_name}
            Description: {spec.description}
            Architecture: {spec.technical_architecture.get('pattern', 'MVC')}
            Database: {spec.database_schema.get('requirements', [])}
            API Endpoints: {spec.api_endpoints}

            Generate:
            1. Main FastAPI application with routes
            2. Pydantic models for data validation
            3. Database models using SQLAlchemy
            4. Authentication and authorization
            5. Error handling and logging
            6. Configuration management

            Use modern Python practices with type hints, async/await, and proper structure.
            """

        return await self._generate_response(prompt, max_tokens=3000)

    async def _generate_java_backend(self, spec: ProjectSpecification) -> str:
        """Generate Java backend code"""
        prompt = f"""
            Generate a complete Java Spring Boot backend application based on this specification:

            Project: {spec.project_name}
            Description: {spec.description}
            Architecture: {spec.technical_architecture.get('pattern', 'MVC')}
            Database: {spec.database_schema.get('requirements', [])}
            API Endpoints: {spec.api_endpoints}

            Generate:
            1. Spring Boot main application class
            2. REST controllers for all endpoints
            3. JPA entities and repositories
            4. Service layer with business logic
            5. Security configuration
            6. Configuration properties

            Use modern Spring Boot practices with annotations and proper layering.
            """

        return await self._generate_response(prompt, max_tokens=3000)

    async def _generate_frontend_code(self, spec: ProjectSpecification) -> str:
        """Generate frontend code based on specification"""
        tech_stack = spec.technical_architecture.get("stack", [])

        if "React" in tech_stack:
            return await self._generate_react_frontend(spec)
        elif "Vue.js" in tech_stack:
            return await self._generate_vue_frontend(spec)
        else:
            return "// Frontend code generation not implemented for this stack"

    async def _generate_react_frontend(self, spec: ProjectSpecification) -> str:
        """Generate React frontend code"""
        prompt = f"""
            Generate a complete React frontend application based on this specification:

            Project: {spec.project_name}
            Description: {spec.description}
            UI Components: {spec.ui_components}
            API Endpoints: {spec.api_endpoints}

            Generate:
            1. Main App component with routing
            2. Individual components for each UI element
            3. State management (Context API or Redux)
            4. API service layer
            5. Styling with modern CSS or styled-components
            6. Error boundaries and loading states

            Use modern React practices with hooks, functional components, and TypeScript if applicable.
            """

        return await self._generate_response(prompt, max_tokens=3000)

    async def _generate_vue_frontend(self, spec: ProjectSpecification) -> str:
        """Generate Vue.js frontend code"""
        prompt = f"""
            Generate a complete Vue.js frontend application based on this specification:

            Project: {spec.project_name}
            Description: {spec.description}
            UI Components: {spec.ui_components}
            API Endpoints: {spec.api_endpoints}

            Generate:
            1. Main Vue application with router
            2. Individual Vue components
            3. Vuex store for state management
            4. API service layer
            5. Styling with Vuetify or custom CSS
            6. Error handling and loading states

            Use modern Vue.js practices with Composition API and TypeScript if applicable.
            """

        return await self._generate_response(prompt, max_tokens=3000)

    async def _generate_database_code(self, spec: ProjectSpecification) -> str:
        """Generate database code and migrations"""
        prompt = f"""
            Generate database schema and migrations based on this specification:

            Project: {spec.project_name}
            Database Requirements: {spec.database_schema.get('requirements', [])}
            API Endpoints: {spec.api_endpoints}

            Generate:
            1. Database schema with tables and relationships
            2. Migration scripts
            3. Seed data for testing
            4. Indexes for performance
            5. Constraints and validations

            Use SQL DDL statements and include proper foreign key relationships.
            """

        return await self._generate_response(prompt, max_tokens=2000)

    async def _generate_api_code(self, spec: ProjectSpecification) -> str:
        """Generate API documentation and specifications"""
        prompt = f"""
            Generate comprehensive API documentation based on this specification:

            Project: {spec.project_name}
            API Endpoints: {spec.api_endpoints}
            Technical Stack: {spec.technical_architecture.get('stack', [])}

            Generate:
            1. OpenAPI/Swagger specification
            2. API endpoint documentation
            3. Request/response examples
            4. Authentication details
            5. Error response formats

            Provide complete API documentation in OpenAPI 3.0 format.
            """

        return await self._generate_response(prompt, max_tokens=2000)

    async def _generate_config_files(self, spec: ProjectSpecification) -> str:
        """Generate configuration files"""
        prompt = f"""
            Generate configuration files based on this specification:

            Project: {spec.project_name}
            Technical Stack: {spec.technical_architecture.get('stack', [])}
            Deployment: {spec.deployment_config}

            Generate:
            1. Package.json or requirements.txt
            2. Environment configuration files
            3. Docker configuration
            4. CI/CD pipeline configuration
            5. Database configuration

            Provide complete configuration files for the project.
            """

        return await self._generate_response(prompt, max_tokens=2000)

    async def _generate_test_code(self, spec: ProjectSpecification) -> str:
        """Generate test code"""
        prompt = f"""
            Generate comprehensive test suite based on this specification:

            Project: {spec.project_name}
            Technical Stack: {spec.technical_architecture.get('stack', [])}
            API Endpoints: {spec.api_endpoints}
            Testing Strategy: {spec.testing_strategy}

            Generate:
            1. Unit tests for core functionality
            2. Integration tests for API endpoints
            3. End-to-end tests for user flows
            4. Test configuration and setup
            5. Mock data and fixtures

            Use appropriate testing frameworks for the technology stack.
            """

        return await self._generate_response(prompt, max_tokens=2000)

    async def process_input(self, user_input: str, context: ProjectContext) -> AgentResponse:
        """Process code generation requests"""
        start_time = time.time()

        prompt = f"""
            You are a Senior Code Generation Agent specializing in producing high-quality, production-ready code.

            User input: "{user_input}"

            Current context readiness:
            - Technical Stack: {len(context.technical_stack)} items
            - Architecture: {'✓' if context.architecture_pattern else '✗'}
            - Database: {len(context.database_requirements)} items
            - APIs: {len(context.api_specifications)} items
            - UI Components: {len(context.ui_components)} items

            Code generation tasks:
            1. Assess code generation readiness
            2. Identify what code can be generated
            3. Determine code quality standards
            4. Plan code structure and organization
            5. Identify dependencies and requirements

            Provide detailed JSON response:
            {{
                "generation_readiness": "assessment of readiness for code generation",
                "generatable_components": ["component 1", "component 2"],
                "quality_standards": ["standard 1", "standard 2"],
                "code_structure": "recommended project structure",
                "dependencies": ["dependency 1", "dependency 2"],
                "follow_up_questions": ["question 1", "question 2"],
                "confidence": 0.8,
                "analysis": "Code generation analysis"
            }}
            """

        response_text = await self._generate_response(prompt, max_tokens=1500)
        response_data = self._parse_json_response(response_text)

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            content=response_data.get("analysis", "Code generation analysis completed"),
            context_updates={},  # Code generation doesn't update context
            next_questions=response_data.get("follow_up_questions", []),
            confidence=response_data.get("confidence", 0.5),
            processing_time=processing_time,
            priority=1,
            tags=["code_generation", "implementation"],
            metadata={
                "generation_readiness": response_data.get("generation_readiness", ""),
                "generatable_components": response_data.get("generatable_components", []),
                "quality_standards": response_data.get("quality_standards", []),
                "code_structure": response_data.get("code_structure", ""),
                "dependencies": response_data.get("dependencies", [])
            }
        )


class AgentCoordinator:
    """Enhanced agent coordinator with better orchestration"""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.agents = {
            "requirements": RequirementsAgent(client),
            "technical": TechnicalAgent(client),
            "ux": UXAgent(client),
            "infrastructure": InfrastructureAgent(client),
            "planning": PlanningAgent(client),
            "code_generation": CodeGenerationAgent(client)
        }
        self.context = ProjectContext()
        self.current_phase = ProjectPhase.DISCOVERY
        self.conversation_history = []
        self.session_metrics = {
            "total_interactions": 0,
            "phase_transitions": 0,
            "agent_activations": {},
            "average_confidence": 0.0,
            "session_start": datetime.now()
        }

    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Enhanced user input processing with better coordination"""
        start_time = time.time()
        self.session_metrics["total_interactions"] += 1

        # Determine which agents should respond
        relevant_agents = await self._determine_relevant_agents(user_input)

        # Process input with relevant agents concurrently
        agent_responses = await self._process_with_agents(user_input, relevant_agents)

        # Update context with agent responses
        await self._update_context_from_responses(agent_responses)

        # Determine next phase
        new_phase = await self._determine_next_phase()
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.session_metrics["phase_transitions"] += 1

        # Generate coordinated response
        coordinated_response = await self._generate_coordinated_response(
            user_input, agent_responses
        )

        # Update conversation history
        self.conversation_history.append({
            "user_input": user_input,
            "agent_responses": agent_responses,
            "coordinated_response": coordinated_response,
            "phase": self.current_phase.value,
            "timestamp": datetime.now().isoformat(),
            "processing_time": time.time() - start_time
        })

        return {
            "response": coordinated_response,
            "phase": self.current_phase.value,
            "context_summary": self._get_context_summary(),
            "next_questions": self._get_prioritized_questions(agent_responses),
            "progress_metrics": self._get_progress_metrics(),
            "session_metrics": self.session_metrics
        }

    async def _determine_relevant_agents(self, user_input: str) -> List[str]:
        """Determine which agents should respond to user input"""
        input_lower = user_input.lower()
        relevant_agents = []

        # Keywords for different agent types
        agent_keywords = {
            "requirements": ["requirement", "feature", "functionality", "user story", "goal"],
            "technical": ["technology", "framework", "architecture", "database", "api"],
            "ux": ["design", "user", "interface", "experience", "ui", "ux"],
            "infrastructure": ["deploy", "host", "scale", "security", "server", "cloud"],
            "planning": ["plan", "timeline", "roadmap", "specification", "ready"],
            "code_generation": ["code", "generate", "implement", "build", "create"]
        }

        # Check for relevant keywords
        for agent_type, keywords in agent_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                relevant_agents.append(agent_type)

        # Always include requirements agent in discovery phase
        if self.current_phase == ProjectPhase.DISCOVERY and "requirements" not in relevant_agents:
            relevant_agents.append("requirements")

        # Default to requirements agent if no specific match
        if not relevant_agents:
            relevant_agents = ["requirements"]

        return relevant_agents

    async def _process_with_agents(self, user_input: str, agent_names: List[str]) -> List[AgentResponse]:
        """Process input with multiple agents concurrently"""
        tasks = []

        for agent_name in agent_names:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                task = agent.process_input(user_input, self.context)
                tasks.append(task)

                # Update agent activation metrics
                self.session_metrics["agent_activations"][agent_name] = (
                        self.session_metrics["agent_activations"].get(agent_name, 0) + 1
                )

        # Execute tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        valid_responses = []
        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"Agent error: {response}")
            else:
                valid_responses.append(response)

        return valid_responses

    async def _update_context_from_responses(self, responses: List[AgentResponse]) -> None:
        """Update project context from agent responses"""
        for response in responses:
            for key, value in response.context_updates.items():
                current_value = getattr(self.context, key, None)

                if isinstance(value, list) and isinstance(current_value, list):
                    # Merge lists and remove duplicates
                    combined = current_value + value
                    setattr(self.context, key, list(dict.fromkeys(combined)))
                elif isinstance(value, str) and value.strip():
                    # Update string values
                    setattr(self.context, key, value)
                elif isinstance(value, dict) and isinstance(current_value, dict):
                    # Merge dictionaries
                    current_value.update(value)

        # Update confidence scores
        for response in responses:
            self.context.confidence_scores[response.agent_id] = response.confidence

        # Update completeness score
        self.context.completeness_score = self._calculate_completeness_score()
        self.context.last_updated = datetime.now().isoformat()

    def _calculate_completeness_score(self) -> float:
        """Calculate overall project completeness score"""
        metrics = self.context.get_completion_metrics()
        return sum(metrics.values()) / len(metrics) if metrics else 0.0

    async def _determine_next_phase(self) -> ProjectPhase:
        """Determine next project phase based on context completeness"""
        completion_score = self.context.completeness_score

        if completion_score < 0.3:
            return ProjectPhase.DISCOVERY
        elif completion_score < 0.7:
            return ProjectPhase.PLANNING
        elif completion_score < 0.9:
            return ProjectPhase.GENERATION
        else:
            return ProjectPhase.VALIDATION

    async def _generate_coordinated_response(
            self, user_input: str, agent_responses: List[AgentResponse]
    ) -> str:
        """Generate coordinated response from multiple agent responses"""
        if not agent_responses:
            return "I'm processing your request. Could you provide more details?"

        # Sort responses by priority and confidence
        sorted_responses = sorted(
            agent_responses,
            key=lambda r: (r.priority, -r.confidence)
        )

        # Create coordinated response
        response_parts = []

        # Add primary response
        primary_response = sorted_responses[0]
        response_parts.append(
            f"**{primary_response.agent_id.replace('_', ' ').title()}:** {primary_response.content}")

        # Add additional insights from other agents
        for response in sorted_responses[1:]:
            if response.confidence > 0.6:
                response_parts.append(f"\n**{response.agent_id.replace('_', ' ').title()}:** {response.content}")

        # Add progress update
        progress_info = self._get_progress_info()
        if progress_info:
            response_parts.append(f"\n**Progress Update:** {progress_info}")

        return "\n".join(response_parts)

    def _get_context_summary(self) -> Dict[str, Any]:
        """Get summarized context for user"""
        return {
            "phase": self.current_phase.value,
            "completeness": f"{self.context.completeness_score:.1%}",
            "goals": len(self.context.goals_requirements),
            "requirements": len(self.context.functional_requirements),
            "technical_stack": len(self.context.technical_stack),
            "ui_components": len(self.context.ui_components),
            "confidence_scores": self.context.confidence_scores
        }

    def _get_prioritized_questions(self, responses: List[AgentResponse]) -> List[str]:
        """Get prioritized follow-up questions from agent responses"""
        all_questions = []

        for response in responses:
            for question in response.next_questions:
                all_questions.append({
                    "question": question,
                    "priority": response.priority,
                    "confidence": response.confidence,
                    "agent": response.agent_id
                })

        # Sort by priority and confidence
        sorted_questions = sorted(
            all_questions,
            key=lambda q: (q["priority"], -q["confidence"])
        )

        # Return top 3 questions
        return [q["question"] for q in sorted_questions[:3]]

    def _get_progress_metrics(self) -> Dict[str, Any]:
        """Get detailed progress metrics"""
        metrics = self.context.get_completion_metrics()
        return {
            "overall_completion": f"{self.context.completeness_score:.1%}",
            "phase": self.current_phase.value,
            "detailed_metrics": {k: f"{v:.1%}" for k, v in metrics.items()},
            "next_phase": self._get_next_phase_info()
        }

    def _get_next_phase_info(self) -> str:
        """Get information about next phase"""
        if self.current_phase == ProjectPhase.DISCOVERY:
            return "Continue gathering requirements and technical details"
        elif self.current_phase == ProjectPhase.PLANNING:
            return "Preparing detailed project specification"
        elif self.current_phase == ProjectPhase.GENERATION:
            return "Ready for code generation"
        else:
            return "Project specification complete"

    def _get_progress_info(self) -> str:
        """Get human-readable progress information"""
        completion = self.context.completeness_score
        phase = self.current_phase.value

        if completion < 0.3:
            return f"We're in the {phase} phase ({completion:.1%} complete). Let's continue gathering requirements."
        elif completion < 0.7:
            return f"Good progress! We're {completion:.1%} complete and moving into {phase}."
        else:
            return f"Excellent! We're {completion:.1%} complete and ready for {phase}."

    async def generate_project_specification(self) -> ProjectSpecification:
        """Generate comprehensive project specification"""
        planning_agent = self.agents["planning"]
        return await planning_agent.create_project_specification(self.context)

    async def generate_project_code(self) -> Dict[str, str]:
        """Generate complete project code"""
        specification = await self.generate_project_specification()
        code_agent = self.agents["code_generation"]
        return await code_agent.generate_code(specification)

    def export_context(self) -> Dict[str, Any]:
        """Export current context for persistence"""
        return {
            "context": asdict(self.context),
            "current_phase": self.current_phase.value,
            "conversation_history": self.conversation_history,
            "session_metrics": self.session_metrics
        }

    def import_context(self, data: Dict[str, Any]) -> None:
        """Import context from persisted data"""
        if "context" in data:
            context_data = data["context"]
            self.context = ProjectContext(**context_data)

        if "current_phase" in data:
            self.current_phase = ProjectPhase(data["current_phase"])

        if "conversation_history" in data:
            self.conversation_history = data["conversation_history"]

        if "session_metrics" in data:
            self.session_metrics.update(data["session_metrics"])


class SocraticRAGSystem:
    """Enhanced main system with better user interface"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.coordinator = AgentCoordinator(self.client)
        self.system_metrics = {
            "total_sessions": 0,
            "total_projects": 0,
            "successful_completions": 0,
            "average_session_duration": 0.0
        }

    async def start_interactive_session(self):
        """Start enhanced interactive session"""
        print("Ουδέν οίδα, ούτε διδάσκω τι, αλλά διαπορώ μόνον.")
        print("Welcome to Multi-Agent Socratic RAG System!")
        print("=" * 60)
        print("This system will help you discover, plan, and generate your software project.")
        print("Type 'help' for commands, 'status' for progress, or 'exit' to quit.")
        print("=" * 60)

        self.system_metrics["total_sessions"] += 1
        session_start = time.time()

        while True:
            try:
                user_input = input("\n🤖 You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() == 'exit':
                    print("..τω Ασκληπιώ οφείλομεν αλετρυόνα, απόδοτε και μη αμελήσετε..")
                    print("Thank you for using the Socratic RAG System!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                elif user_input.lower() == 'export':
                    self._export_session()
                    continue
                elif user_input.lower() == 'generate':
                    await self._generate_project()
                    continue

                # Process user input
                print("\n🔄 Processing your input...")
                response_data = await self.coordinator.process_user_input(user_input)

                # Display response
                print(f"\n{response_data['response']}")

                # Show progress
                progress = response_data['progress_metrics']
                print(f"\n📊 Progress: {progress['overall_completion']} | Phase: {progress['phase']}")

                # Show next questions
                next_questions = response_data['next_questions']
                if next_questions:
                    print(f"\n❓ Consider these questions:")
                    for i, question in enumerate(next_questions, 1):
                        print(f"   {i}. {question}")

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Session error: {e}")
                print(f"\n❌ Error: {e}")
                print("Please try again or contact support.")

        # Update session metrics
        session_duration = time.time() - session_start
        self.system_metrics["average_session_duration"] = (
                (self.system_metrics["average_session_duration"] *
                 (self.system_metrics["total_sessions"] - 1) + session_duration) /
                self.system_metrics["total_sessions"]
        )

    def _show_help(self):
        """Display help information"""
        print("\n" + "=" * 50)
        print("📚 SOCRATIC RAG SYSTEM - HELP")
        print("=" * 50)
        print("Available Commands:")
        print("  help     - Show this help message")
        print("  status   - Show current project status and progress")
        print("  export   - Export current session data")
        print("  generate - Generate project specification and code")
        print("  exit     - Exit the system")
        print()
        print("How to Use:")
        print("  • Describe your project idea in natural language")
        print("  • Answer questions to refine requirements")
        print("  • The system will guide you through discovery → planning → generation")
        print("  • Use 'status' to check progress at any time")
        print()
        print("Tips:")
        print("  • Be specific about your requirements")
        print("  • Mention technologies you prefer")
        print("  • Describe your target users")
        print("  • Include any constraints or preferences")
        print("=" * 50)

    def _show_status(self):
        """Display current project status"""
        print("\n" + "=" * 50)
        print("📊 PROJECT STATUS")
        print("=" * 50)

        # Current phase
        phase = self.coordinator.current_phase.value
        print(f"Current Phase: {phase}")

        # Completion metrics
        context = self.coordinator.context
        completion = context.completeness_score
        print(f"Overall Completion: {completion:.1%}")

        # Detailed metrics
        metrics = context.get_completion_metrics()
        print("\nDetailed Progress:")
        for metric, value in metrics.items():
            status = "✅" if value > 0.8 else "🔄" if value > 0.3 else "❌"
            print(f"  {status} {metric.replace('_', ' ').title()}: {value:.1%}")

        # Context summary
        print(f"\nCurrent Context:")
        print(f"  • Goals: {len(context.goals_requirements)} defined")
        print(f"  • Requirements: {len(context.functional_requirements)} specified")
        print(f"  • Technical Stack: {len(context.technical_stack)} technologies")
        print(f"  • UI Components: {len(context.ui_components)} designed")
        print(f"  • API Endpoints: {len(context.api_specifications)} planned")

        # Next steps
        print(f"\nNext Steps:")
        if completion < 0.3:
            print("  • Continue defining project requirements")
            print("  • Specify technical preferences")
            print("  • Describe target users and use cases")
        elif completion < 0.7:
            print("  • Finalize technical architecture")
            print("  • Complete UI/UX specifications")
            print("  • Define deployment requirements")
        elif completion < 0.9:
            print("  • Ready for code generation")
            print("  • Use 'generate' command to create project")
        else:
            print("  • Project specification complete!")
            print("  • Ready for full code generation")

        # Session metrics
        session_metrics = self.coordinator.session_metrics
        print(f"\nSession Info:")
        print(f"  • Interactions: {session_metrics['total_interactions']}")
        print(f"  • Phase Transitions: {session_metrics['phase_transitions']}")
        print(f"  • Active Agents: {len(session_metrics['agent_activations'])}")

        print("=" * 50)

    def _export_session(self):
        """Export current session data"""
        try:
            export_data = self.coordinator.export_context()

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"socratic_rag_session_{timestamp}.json"

            # Save to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            print(f"\n💾 Session exported successfully to: {filename}")
            print(f"   • Total interactions: {export_data['session_metrics']['total_interactions']}")
            print(f"   • Current phase: {export_data['current_phase']}")
            print(f"   • Completion: {export_data['context']['completeness_score']:.1%}")

        except Exception as e:
            print(f"\n❌ Export failed: {e}")

        async def _generate_project(self):
            """Generate project specification and code"""
            try:
                print("\n🔄 Generating project specification...")

                # Check if ready for generation
                if self.coordinator.context.completeness_score < 0.7:
                    print("⚠️  Project not ready for generation.")
                    print("   Please continue defining requirements and technical details.")
                    print("   Use 'status' to check what's missing.")
                    return

                # Generate specification
                spec = await self.coordinator.generate_project_specification()
                print("✅ Project specification generated!")

                # Generate code
                print("\n🔄 Generating project code...")
                code_modules = await self.coordinator.generate_project_code()

                # Save generated files
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                project_dir = f"generated_project_{timestamp}"

                # Create project directory
                import os
                os.makedirs(project_dir, exist_ok=True)

                # Save specification
                spec_file = os.path.join(project_dir, "project_specification.json")
                with open(spec_file, 'w') as f:
                    json.dump(asdict(spec), f, indent=2, default=str)

                # Save code modules
                for module_name, code_content in code_modules.items():
                    module_file = os.path.join(project_dir, f"{module_name}.txt")
                    with open(module_file, 'w') as f:
                        f.write(code_content)

                print(f"✅ Project generated successfully!")
                print(f"   📁 Project directory: {project_dir}")
                print(f"   📄 Files generated: {len(code_modules) + 1}")
                print(f"   📊 Specification: {spec_file}")

                # Update metrics
                self.system_metrics["total_projects"] += 1
                self.system_metrics["successful_completions"] += 1

            except Exception as e:
                logger.error(f"Generation error: {e}")
                print(f"\n❌ Generation failed: {e}")
                print("   Please check your requirements and try again.")

    def load_session(self, filename: str):
        """Load a previous session"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.coordinator.import_context(data)
            print(f"✅ Session loaded from: {filename}")

        except Exception as e:
            print(f"❌ Failed to load session: {e}")

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        return {
                **self.system_metrics,
                "success_rate": (
                        self.system_metrics["successful_completions"] /
                        max(self.system_metrics["total_projects"], 1)
                ),
                "current_session": self.coordinator.session_metrics
            }

    # Example usage and main entry point


async def main():
    """Main entry point for the Socratic RAG System"""
    import os

    # Get API key from environment
    api_key = os.getenv("API_KEY_CLAUDE")
    if not api_key:
        print("❌ Error: API_KEY_CLAUDE environment variable not set")
        print("   Please set your Anthropic API key:")
        print("   export API_KEY_CLAUDE=your_api_key_here")
        return

    # Initialize system
    system = SocraticRAGSystem(api_key)

    # Start interactive session
    await system.start_interactive_session()


if __name__ == "__main__":
    asyncio.run(main())
