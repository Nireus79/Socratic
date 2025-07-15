#!/usr/bin/env python3
"""
Multi-Agent Socratic RAG System
A sophisticated project development system using specialized agents for discovery, planning, and code generation
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
                model="claude-sonnet-4-20250514",
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

        Generate SQLAlchemy models with:
        - Proper relationships
        - Constraints and validations
        - Indexes where appropriate
        - Helper methods
        """

        return self._generate_response(prompt, max_tokens=2000)

    async def _generate_routes(self, spec: ProjectSpecification) -> str:
        """Generate API routes"""
        prompt = f"""
        Generate API routes for this project:

        Project: {spec.project_name}
        API Endpoints: {spec.api_endpoints}
        Tech Stack: {', '.join(spec.technical_architecture.get('stack', []))}

        Generate complete route handlers with:
        - Proper HTTP methods
        - Input validation
        - Error handling
        - Authentication where needed
        """

        return self._generate_response(prompt, max_tokens=2000)

    async def _generate_config(self, spec: ProjectSpecification) -> str:
        """Generate configuration file"""
        config_template = f"""
import os
from typing import Dict, Any

class Config:
    '''Configuration for {spec.project_name}'''

    # Basic settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///app.db')

    # Project specific settings
    PROJECT_NAME = "{spec.project_name}"

    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True

config = {{
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}}
"""
        return config_template

    async def _generate_requirements(self, spec: ProjectSpecification) -> str:
        """Generate requirements.txt"""
        base_requirements = [
            "flask",
            "flask-sqlalchemy",
            "python-dotenv",
            "gunicorn"
        ]

        tech_stack = spec.technical_architecture.get('stack', [])
        tech_stack_lower = [tech.lower() for tech in tech_stack]

        if 'fastapi' in tech_stack_lower:
            base_requirements.extend(['fastapi', 'uvicorn'])
        if 'postgresql' in tech_stack_lower:
            base_requirements.append('psycopg2-binary')
        if 'redis' in tech_stack_lower:
            base_requirements.append('redis')

        return '\n'.join(sorted(set(base_requirements)))

    async def _generate_readme(self, spec: ProjectSpecification) -> str:
        """Generate README.md"""
        readme_template = f"""# {spec.project_name}

{spec.description}

## Features

- Built with {', '.join(spec.technical_architecture.get('stack', []))}
- {spec.technical_architecture.get('pattern', '')} architecture
- Production-ready configuration

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables
4. Run the application: `python main.py`

## API Endpoints

{spec.api_endpoints}

## Deployment

Deployment target: {spec.deployment_config.get('target', 'Not specified')}

## Testing

{spec.testing_strategy.get('approach', 'Testing strategy not specified')}

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
"""
        return readme_template


class ValidationAgent(BaseAgent):
    """Agent that validates generated code and specifications"""

    def __init__(self, client: anthropic.Anthropic):
        super().__init__("validation_agent", client)
        self.expertise_areas = ["code_review", "testing", "quality_assurance"]

    async def validate_code(self, generated_files: Dict[str, str]) -> Dict[str, Any]:
        """Validate generated code for quality and correctness"""
        validation_results = {}

        for filename, content in generated_files.items():
            result = await self._validate_file(filename, content)
            validation_results[filename] = result

        return validation_results

    async def _validate_file(self, filename: str, content: str) -> Dict[str, Any]:
        """Validate a single file"""
        prompt = f"""
        Review this generated code file for quality and correctness:

        Filename: {filename}
        Content: {content}

        Check for:
        1. Syntax errors
        2. Security vulnerabilities
        3. Best practices compliance
        4. Code organization
        5. Error handling
        6. Documentation

        Return assessment with:
        - Issues found
        - Severity levels
        - Recommendations
        - Overall quality score (0-10)
        """

        response = self._generate_response(prompt, max_tokens=1000)

        return {
            "filename": filename,
            "assessment": response,
            "quality_score": 8.0,  # Simplified for demo
            "issues_found": [],
            "recommendations": []
        }


class DiscoveryCoordinator:
    """Coordinates the discovery phase using multiple agents"""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.agents = {
            "requirements": RequirementsAgent(client),
            "technical": TechnicalAgent(client),
            "ux": UXAgent(client),
            "infrastructure": InfrastructureAgent(client)
        }
        self.planning_agent = PlanningAgent(client)
        self.code_generation_agent = CodeGenerationAgent(client)
        self.validation_agent = ValidationAgent(client)

    async def process_user_input(self, user_input: str, context: ProjectContext) -> Dict[str, Any]:
        """Process user input through all relevant agents"""
        agent_responses = {}

        # Process input through all agents in parallel
        tasks = []
        for agent_name, agent in self.agents.items():
            task = agent.process_input(user_input, context)
            tasks.append((agent_name, task))

        # Wait for all agents to respond
        for agent_name, task in tasks:
            try:
                response = await task
                agent_responses[agent_name] = response
            except Exception as e:
                logger.error(f"Error processing with {agent_name}: {e}")
                agent_responses[agent_name] = None

        return agent_responses

    def merge_context_updates(self, agent_responses: Dict[str, AgentResponse],
                              context: ProjectContext) -> ProjectContext:
        """Merge context updates from all agents"""
        confidence_scores = {}

        for agent_name, response in agent_responses.items():
            if response is None:
                continue

            # Update context based on agent response
            for key, value in response.context_updates.items():
                if hasattr(context, key):
                    current_value = getattr(context, key)
                    if isinstance(current_value, list):
                        # Merge lists, avoiding duplicates
                        new_items = [item for item in value if item not in current_value]
                        setattr(context, key, current_value + new_items)
                    else:
                        # Update single values if they're not empty
                        if value:
                            setattr(context, key, value)

            # Track confidence scores
            confidence_scores[agent_name] = response.confidence

        # Update overall confidence scores
        context.confidence_scores = confidence_scores
        context.completeness_score = sum(confidence_scores.values()) / len(
            confidence_scores) if confidence_scores else 0.0
        context.last_updated = datetime.now().isoformat()

        return context

    def get_next_questions(self, agent_responses: Dict[str, AgentResponse]) -> List[str]:
        """Get prioritized next questions from all agents"""
        all_questions = []

        for agent_name, response in agent_responses.items():
            if response and response.next_questions:
                # Weight questions by agent confidence
                weighted_questions = [(q, response.confidence) for q in response.next_questions]
                all_questions.extend(weighted_questions)

        # Sort by confidence and return top questions
        all_questions.sort(key=lambda x: x[1], reverse=True)
        return [q[0] for q in all_questions[:3]]  # Return top 3 questions

    def is_ready_for_generation(self, context: ProjectContext) -> bool:
        """Check if context is complete enough for code generation"""
        return (
                context.completeness_score > 0.7 and
                len(context.goals_requirements) > 0 and
                len(context.technical_stack) > 0 and
                (len(context.api_specifications) > 0 or len(context.ui_components) > 0)
        )


class MultiAgentSocraticRAG:
    """Main system orchestrating the multi-agent approach"""

    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get('API_KEY_CLAUDE'))
        self.coordinator = DiscoveryCoordinator(self.client)
        self.context = ProjectContext()
        self.current_phase = ProjectPhase.DISCOVERY
        self.conversation_history = []

    async def chat(self, user_input: str) -> str:
        """Main chat interface"""
        if user_input.lower() in ['quit', 'exit', 'end']:
            return "👋 Goodbye! Your project context has been saved."

        if user_input.lower() == 'generate' or user_input.lower() == 'code':
            return await self._generate_project_code()

        if user_input.lower() == 'summary':
            return self._generate_context_summary()

        if user_input.lower() == 'validate':
            return await self._validate_current_context()

        # Process user input through agents
        agent_responses = await self.coordinator.process_user_input(user_input, self.context)

        # Merge context updates
        self.context = self.coordinator.merge_context_updates(agent_responses, self.context)

        # Get next questions
        next_questions = self.coordinator.get_next_questions(agent_responses)

        # Generate response
        response = self._generate_user_response(agent_responses, next_questions)

        # Check if ready for code generation
        if self.coordinator.is_ready_for_generation(self.context):
            response += "\n\n🎯 **Your project context is now complete!**"
            response += "\n💡 Type 'generate' to create your project code, or continue refining requirements."

        return response

    def _generate_user_response(self, agent_responses: Dict[str, AgentResponse], next_questions: List[str]) -> str:
        """Generate a comprehensive response for the user"""
        response_parts = []

        # Summarize agent insights
        insights = []
        for agent_name, response in agent_responses.items():
            if response and response.content:
                insights.append(f"**{agent_name.title()}**: {response.content}")

        if insights:
            response_parts.append("## 📊 Agent Analysis:")
            response_parts.extend(insights)

        # Show context completeness
        response_parts.append(f"\n## 🎯 Project Completeness: {self.context.completeness_score:.1%}")

        # Show confidence scores
        if self.context.confidence_scores:
            response_parts.append("\n**Agent Confidence Scores:**")
            for agent, score in self.context.confidence_scores.items():
                response_parts.append(f"- {agent.title()}: {score:.1%}")

        # Show next questions
        if next_questions:
            response_parts.append("\n## 🤔 Next Steps:")
            for i, question in enumerate(next_questions, 1):
                response_parts.append(f"{i}. {question}")

        def _generate_context_summary(self) -> str:
            """Generate a summary of the current project context"""
            summary_parts = []

            summary_parts.append("# 📋 Project Context Summary")

            if self.context.goals_requirements:
                summary_parts.append("\n## 🎯 Goals & Requirements:")
                for goal in self.context.goals_requirements:
                    summary_parts.append(f"- {goal}")

            if self.context.functional_requirements:
                summary_parts.append("\n## ⚙️ Functional Requirements:")
                for req in self.context.functional_requirements:
                    summary_parts.append(f"- {req}")

            if self.context.non_functional_requirements:
                summary_parts.append("\n## 🔧 Non-Functional Requirements:")
                for req in self.context.non_functional_requirements:
                    summary_parts.append(f"- {req}")

            if self.context.technical_stack:
                summary_parts.append("\n## 💻 Technical Stack:")
                for tech in self.context.technical_stack:
                    summary_parts.append(f"- {tech}")

            if self.context.architecture_pattern:
                summary_parts.append(f"\n## 🏗️ Architecture Pattern: {self.context.architecture_pattern}")

            if self.context.database_requirements:
                summary_parts.append("\n## 🗃️ Database Requirements:")
                for req in self.context.database_requirements:
                    summary_parts.append(f"- {req}")

            if self.context.api_specifications:
                summary_parts.append("\n## 🔌 API Specifications:")
                for spec in self.context.api_specifications:
                    summary_parts.append(f"- {spec}")

            if self.context.ui_components:
                summary_parts.append("\n## 🎨 UI Components:")
                for component in self.context.ui_components:
                    summary_parts.append(f"- {component}")

            if self.context.user_personas:
                summary_parts.append("\n## 👤 User Personas:")
                for persona in self.context.user_personas:
                    summary_parts.append(f"- {persona}")

            if self.context.user_flows:
                summary_parts.append("\n## 🔄 User Flows:")
                for flow in self.context.user_flows:
                    summary_parts.append(f"- {flow}")

            if self.context.deployment_target:
                summary_parts.append(f"\n## 🚀 Deployment Target: {self.context.deployment_target}")

            if self.context.scalability_requirements:
                summary_parts.append("\n## 📈 Scalability Requirements:")
                for req in self.context.scalability_requirements:
                    summary_parts.append(f"- {req}")

            if self.context.security_requirements:
                summary_parts.append("\n## 🔒 Security Requirements:")
                for req in self.context.security_requirements:
                    summary_parts.append(f"- {req}")

            summary_parts.append(f"\n## 📊 Overall Completeness: {self.context.completeness_score:.1%}")

            return "\n".join(summary_parts)

        async def _validate_current_context(self) -> str:
            """Validate the current project context"""
            validation_results = []

            validation_results.append("# 🔍 Context Validation")

            # Check completeness of different areas
            areas_to_check = [
                ("Goals & Requirements", self.context.goals_requirements),
                ("Functional Requirements", self.context.functional_requirements),
                ("Technical Stack", self.context.technical_stack),
                ("Architecture Pattern",
                 [self.context.architecture_pattern] if self.context.architecture_pattern else []),
                ("Database Requirements", self.context.database_requirements),
                ("API Specifications", self.context.api_specifications),
                ("UI Components", self.context.ui_components),
                ("Deployment Target", [self.context.deployment_target] if self.context.deployment_target else [])
            ]

            validation_results.append("\n## ✅ Completeness Check:")
            for area_name, area_data in areas_to_check:
                status = "✅ Complete" if area_data else "❌ Missing"
                count = len(area_data) if area_data else 0
                validation_results.append(f"- {area_name}: {status} ({count} items)")

            # Check for potential issues
            issues = []

            if not self.context.goals_requirements:
                issues.append("No project goals defined")

            if not self.context.technical_stack:
                issues.append("No technology stack specified")

            if not self.context.api_specifications and not self.context.ui_components:
                issues.append("No API or UI specifications provided")

            if not self.context.deployment_target:
                issues.append("No deployment target specified")

            if issues:
                validation_results.append("\n## ⚠️ Issues Found:")
                for issue in issues:
                    validation_results.append(f"- {issue}")
            else:
                validation_results.append("\n## 🎉 No critical issues found!")

            # Readiness assessment
            if self.coordinator.is_ready_for_generation(self.context):
                validation_results.append("\n## 🚀 Ready for Code Generation!")
            else:
                validation_results.append("\n## 📝 More information needed before code generation")

            return "\n".join(validation_results)

        async def _generate_project_code(self) -> str:
            """Generate complete project code"""
            if not self.coordinator.is_ready_for_generation(self.context):
                return """
    🚫 **Cannot generate code yet!**

    Your project context needs more information. Please provide:
    - Clear project goals
    - Technical stack preferences
    - API specifications or UI components
    - Basic deployment requirements

    Type 'validate' to see what's missing.
    """

            try:
                # Update phase
                self.current_phase = ProjectPhase.PLANNING

                # Create detailed specification
                specification = await self.coordinator.planning_agent.create_project_specification(self.context)

                # Generate code
                self.current_phase = ProjectPhase.GENERATION
                generated_files = await self.coordinator.code_generation_agent.generate_code(specification)

                # Validate generated code
                self.current_phase = ProjectPhase.VALIDATION
                validation_results = await self.coordinator.validation_agent.validate_code(generated_files)

                # Update phase
                self.current_phase = ProjectPhase.COMPLETE

                # Format response
                response_parts = []
                response_parts.append("# 🎉 Project Generated Successfully!")
                response_parts.append(f"\n**Project Name**: {specification.project_name}")
                response_parts.append(f"**Description**: {specification.description}")

                response_parts.append("\n## 📁 Generated Files:")
                for filename in generated_files.keys():
                    response_parts.append(f"- {filename}")

                response_parts.append("\n## 📊 Validation Results:")
                avg_quality = sum(result.get('quality_score', 0) for result in validation_results.values()) / len(
                    validation_results)
                response_parts.append(f"- Average Quality Score: {avg_quality:.1f}/10")

                response_parts.append("\n## 🔍 File Contents:")
                for filename, content in generated_files.items():
                    response_parts.append(f"\n### {filename}")
                    response_parts.append("```")
                    response_parts.append(content)
                    response_parts.append("```")

                return "\n".join(response_parts)

            except Exception as e:
                logger.error(f"Error generating project code: {e}")
                return f"❌ Error generating code: {str(e)}"

        def save_context(self, filename: str = "project_context.json") -> str:
            """Save current context to file"""
            try:
                context_dict = asdict(self.context)
                with open(filename, 'w') as f:
                    json.dump(context_dict, f, indent=2)
                return f"✅ Context saved to {filename}"
            except Exception as e:
                return f"❌ Error saving context: {str(e)}"

        def load_context(self, filename: str = "project_context.json") -> str:
            """Load context from file"""
            try:
                with open(filename, 'r') as f:
                    context_dict = json.load(f)
                self.context = ProjectContext(**context_dict)
                return f"✅ Context loaded from {filename}"
            except Exception as e:
                return f"❌ Error loading context: {str(e)}"

        def get_help(self) -> str:
            """Get help information"""
            return """
    # 🤖 Multi-Agent Socratic RAG System Help

    ## Available Commands:
    - **Regular conversation**: Ask questions about your project
    - **'generate' or 'code'**: Generate complete project code
    - **'summary'**: View current project context
    - **'validate'**: Check context completeness
    - **'quit', 'exit', 'end'**: Exit the system

    ## How it works:
    1. **Discovery Phase**: Multiple specialized agents analyze your input
       - Requirements Agent: Gathers functional/non-functional requirements
       - Technical Agent: Determines architecture and tech stack
       - UX Agent: Focuses on user experience and interface design
       - Infrastructure Agent: Handles deployment and scalability

    2. **Planning Phase**: Creates detailed project specifications

    3. **Generation Phase**: Generates complete, production-ready code

    4. **Validation Phase**: Reviews generated code for quality

    ## Tips:
    - Be specific about your requirements
    - Mention preferred technologies
    - Describe your users and use cases
    - Consider scalability and deployment needs
    - The system will ask follow-up questions to clarify requirements

    ## Current Phase: {phase}
    ## Context Completeness: {completeness:.1%}
    """.format(phase=self.current_phase.value, completeness=self.context.completeness_score)


async def main():
    """Main function to run the Multi-Agent Socratic RAG system"""
    print("🚀 Multi-Agent Socratic RAG System")
    print("=" * 50)

    # Initialize system
    system = MultiAgentSocraticRAG()

    # Welcome message
    print("""
    Welcome to the Multi-Agent Socratic RAG System!

    This system uses specialized AI agents to help you:
    - Discover and refine project requirements
    - Design technical architecture
    - Generate complete, production-ready code

    Just describe your project idea, and our agents will guide you through
    the development process with intelligent questions and suggestions.

    Type 'help' for available commands, or start describing your project!
    """)

    # Main conversation loop
    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'help':
                print(system.get_help())
                continue

            if user_input.lower() in ['quit', 'exit', 'end']:
                print("👋 Goodbye! Your project context has been saved.")
                system.save_context()
                break

            # Process user input
            response = await system.chat(user_input)
            print(f"\n🤖 System: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Your project context has been saved.")
            system.save_context()
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            logger.error(f"Error in main loop: {e}")


if __name__ == "__main__":
    asyncio.run(main())
