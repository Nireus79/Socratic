#!/usr/bin/env python3

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
import numpy as np
from colorama import init, Fore, Back, Style

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

# Configuration
CONFIG = {
    'MAX_CONTEXT_LENGTH': 8000,
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    'CLAUDE_MODEL': 'claude-3-5-sonnet-20241022',
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


@dataclass
class ConflictInfo:
    conflict_id: str
    conflict_type: str  # 'tech_stack', 'requirements', 'goals', 'constraints'
    old_value: str
    new_value: str
    old_author: str
    new_author: str
    old_timestamp: str
    new_timestamp: str
    severity: str  # 'low', 'medium', 'high'
    suggestions: List[str]


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
        elif action == 'list_collaborators':
            return self._list_collaborators(request)
        elif action == 'remove_collaborator':
            return self._remove_collaborator(request)

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

    def _list_collaborators(self, request: Dict) -> Dict:
        """List all collaborators for a project"""
        project = request.get('project')

        collaborators_info = []
        # Add owner info
        collaborators_info.append({
            'username': project.owner,
            'role': 'owner'
        })

        # Add collaborators info
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
        """Remove a collaborator from project"""
        project = request.get('project')
        username = request.get('username')
        requester = request.get('requester')

        # Only owner can remove collaborators
        if requester != project.owner:
            return {'status': 'error', 'message': 'Only project owner can remove collaborators'}

        # Cannot remove owner
        if username == project.owner:
            return {'status': 'error', 'message': 'Cannot remove project owner'}

        if username in project.collaborators:
            project.collaborators.remove(username)
            self.orchestrator.database.save_project(project)
            self.log(f"Removed collaborator '{username}' from project '{project.name}'")
            return {'status': 'success'}
        else:
            return {'status': 'error', 'message': 'User is not a collaborator'}


class SocraticCounselorAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("SocraticCounselor", orchestrator)
        self.use_dynamic_questions = True  # Toggle for dynamic vs static questions
        self.max_questions_per_phase = 5

        # Fallback static questions if Claude is unavailable
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

        return {'status': 'error', 'message': 'Unknown action'}

    def _generate_question(self, request: Dict) -> Dict:
        project = request.get('project')
        context = self.orchestrator.context_analyzer.get_context_summary(project)

        # Count questions already asked in this phase
        phase_questions = [msg for msg in project.conversation_history
                           if msg.get('type') == 'assistant' and msg.get('phase') == project.phase]

        if self.use_dynamic_questions:
            question = self._generate_dynamic_question(project, context, len(phase_questions))
        else:
            question = self._generate_static_question(project, len(phase_questions))

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
        """Generate contextual questions using Claude"""

        # Get conversation history for context
        recent_conversation = ""
        if project.conversation_history:
            recent_messages = project.conversation_history[-4:]  # Last 4 messages
            for msg in recent_messages:
                role = "Assistant" if msg['type'] == 'assistant' else "User"
                recent_conversation += f"{role}: {msg['content']}\n"

        # Get relevant knowledge from vector database
        relevant_knowledge = ""
        if context:
            knowledge_results = self.orchestrator.vector_db.search_similar(context, top_k=3)
            if knowledge_results:
                relevant_knowledge = "\n".join([result['content'][:200] + "..." for result in knowledge_results])

        prompt = self._build_question_prompt(project, context, recent_conversation, relevant_knowledge, question_count)

        try:
            question = self.orchestrator.claude_client.generate_socratic_question(prompt)
            self.log(f"Generated dynamic question for {project.phase} phase")
            return question
        except Exception as e:
            self.log(f"Failed to generate dynamic question: {e}, falling back to static", "WARN")
            return self._generate_static_question(project, question_count)

    def _build_question_prompt(self, project: ProjectContext, context: str,
                               recent_conversation: str, relevant_knowledge: str, question_count: int) -> str:
        """Build prompt for dynamic question generation"""

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

        return f"""You are a Socratic tutor helping a developer think through their software project. 

Project Details:
- Name: {project.name}
- Current Phase: {project.phase} ({phase_descriptions.get(project.phase, '')})
- Goals: {project.goals}
- Tech Stack: {', '.join(project.tech_stack) if project.tech_stack else 'Not specified'}
- Requirements: {', '.join(project.requirements) if project.requirements else 'Not specified'}

Project Context:
{context}

Recent Conversation:
{recent_conversation}

Relevant Knowledge:
{relevant_knowledge}

This is question #{question_count + 1} in the {project.phase} phase. Focus on: {phase_focus.get(project.phase, '')}.

Generate ONE insightful Socratic question that:
1. Builds on what we've discussed so far
2. Helps the user think deeper about their project
3. Is specific to the {project.phase} phase
4. Encourages critical thinking rather than just information gathering
5. Is relevant to their stated goals and tech stack

The question should be thought-provoking but not overwhelming. Make it conversational and engaging.

Return only the question, no additional text or explanation."""

    def _generate_static_question(self, project: ProjectContext, question_count: int) -> str:
        """Generate questions from static predefined lists"""
        questions = self.static_questions.get(project.phase, [])

        if question_count < len(questions):
            return questions[question_count]
        else:
            # Fallback questions when we've exhausted the static list
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

        # Add to conversation history with phase information
        project.conversation_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'user',
            'content': user_response,
            'phase': project.phase,
            'author': current_user  # Track who said what
        })

        # Extract insights using Claude
        insights = self.orchestrator.claude_client.extract_insights(user_response, project)

        # REAL-TIME CONFLICT DETECTION
        if insights:
            conflict_result = self.orchestrator.process_request('conflict_detector', {
                'action': 'detect_conflicts',
                'project': project,
                'new_insights': insights,
                'current_user': current_user
            })

            if conflict_result['status'] == 'success' and conflict_result['conflicts']:
                # Handle conflicts before updating context
                conflicts_resolved = self._handle_conflicts_realtime(conflict_result['conflicts'], project)
                if not conflicts_resolved:
                    # User chose not to resolve conflicts, don't update context
                    return {'status': 'success', 'insights': insights, 'conflicts_pending': True}

        # Update context only if no conflicts or conflicts were resolved
        self._update_project_context(project, insights)

        return {'status': 'success', 'insights': insights}

    def _handle_conflicts_realtime(self, conflicts: List[ConflictInfo], project: ProjectContext) -> bool:
        """Handle conflicts in real-time during conversation"""
        for conflict in conflicts:
            print(f"\n{Fore.RED}⚠️  CONFLICT DETECTED!")
            print(f"{Fore.YELLOW}Type: {conflict.conflict_type}")
            print(f"{Fore.WHITE}Existing: '{conflict.old_value}' (by {conflict.old_author})")
            print(f"{Fore.WHITE}New: '{conflict.new_value}' (by {conflict.new_author})")
            print(f"{Fore.RED}Severity: {conflict.severity}")

            # Get AI-generated suggestions
            suggestions = self.orchestrator.claude_client.generate_conflict_resolution_suggestions(conflict, project)
            print(f"\n{Fore.MAGENTA}{suggestions}")

            print(f"\n{Fore.CYAN}Resolution Options:")
            print("1. Keep existing specification")
            print("2. Replace with new specification")
            print("3. Skip this specification (continue without adding)")
            print("4. Manual resolution (edit both)")

            while True:
                choice = input(f"{Fore.WHITE}Choose resolution (1-4): ").strip()

                if choice == '1':
                    print(f"{Fore.GREEN}✓ Keeping existing: '{conflict.old_value}'")
                    # Remove new value from insights so it won't be added
                    self._remove_from_insights(conflict.new_value, conflict.conflict_type)
                    break
                elif choice == '2':
                    print(f"{Fore.GREEN}✓ Replacing with: '{conflict.new_value}'")
                    # Remove old value from project context
                    self._remove_from_project_context(project, conflict.old_value, conflict.conflict_type)
                    break
                elif choice == '3':
                    print(f"{Fore.YELLOW}⏭️  Skipping specification")
                    self._remove_from_insights(conflict.new_value, conflict.conflict_type)
                    break
                elif choice == '4':
                    resolved_value = self._manual_resolution(conflict)
                    if resolved_value:
                        # Replace both old and new with manually resolved value
                        self._remove_from_project_context(project, conflict.old_value, conflict.conflict_type)
                        self._update_insights_value(conflict.new_value, resolved_value, conflict.conflict_type)
                        print(f"{Fore.GREEN}✓ Updated to: '{resolved_value}'")
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.")

        return True  # Conflicts handled

    def _remove_from_project_context(self, project: ProjectContext, value: str, context_type: str):
        """Remove a value from project context"""
        if context_type == 'tech_stack' and value in project.tech_stack:
            project.tech_stack.remove(value)
        elif context_type == 'requirements' and value in project.requirements:
            project.requirements.remove(value)
        elif context_type == 'constraints' and value in project.constraints:
            project.constraints.remove(value)
        elif context_type == 'goals':
            project.goals = ""

    def _manual_resolution(self, conflict: ConflictInfo) -> str:
        """Allow user to manually resolve conflict"""
        print(f"\n{Fore.CYAN}Manual Resolution:")
        print(f"Current options: '{conflict.old_value}' vs '{conflict.new_value}'")

        new_value = input(f"{Fore.WHITE}Enter resolved specification: ").strip()
        if new_value:
            return new_value
        return ""

    def _advance_phase(self, request: Dict) -> Dict:
        project = request.get('project')
        phases = ['discovery', 'analysis', 'design', 'implementation']

        current_index = phases.index(project.phase)
        if current_index < len(phases) - 1:
            project.phase = phases[current_index + 1]
            self.log(f"Advanced project to {project.phase} phase")

        return {'status': 'success', 'new_phase': project.phase}

    def _update_project_context(self, project: ProjectContext, insights: Dict):
        """Update project context based on extracted insights - with better error handling"""
        if not insights or not isinstance(insights, dict):
            return  # Skip if insights is None or not a dict

        try:
            if 'goals' in insights and insights['goals']:
                if isinstance(insights['goals'], str):
                    project.goals = insights['goals']

            if 'requirements' in insights and insights['requirements']:
                if isinstance(insights['requirements'], list):
                    # Filter out empty strings and duplicates
                    new_requirements = [req for req in insights['requirements']
                                        if req and isinstance(req, str) and req not in project.requirements]
                    project.requirements.extend(new_requirements)
                elif isinstance(insights['requirements'], str):
                    if insights['requirements'] not in project.requirements:
                        project.requirements.append(insights['requirements'])

            if 'tech_stack' in insights and insights['tech_stack']:
                if isinstance(insights['tech_stack'], list):
                    new_tech = [tech for tech in insights['tech_stack']
                                if tech and isinstance(tech, str) and tech not in project.tech_stack]
                    project.tech_stack.extend(new_tech)
                elif isinstance(insights['tech_stack'], str):
                    if insights['tech_stack'] not in project.tech_stack:
                        project.tech_stack.append(insights['tech_stack'])

            if 'constraints' in insights and insights['constraints']:
                if isinstance(insights['constraints'], list):
                    new_constraints = [constraint for constraint in insights['constraints']
                                       if constraint and isinstance(constraint,
                                                                    str) and constraint not in project.constraints]
                    project.constraints.extend(new_constraints)
                elif isinstance(insights['constraints'], str):
                    if insights['constraints'] not in project.constraints:
                        project.constraints.append(insights['constraints'])

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error updating project context: {e}")
            # Continue execution even if context update fails

    def _remove_from_insights(self, value: str, insight_type: str):
        """Remove a value from insights before context update"""
        # This would be used to prevent conflicting values from being added
        # Implementation depends on how insights are structured
        pass

    def _update_insights_value(self, old_value: str, new_value: str, insight_type: str):
        """Update a value in insights before context update"""
        # This would replace a value in insights with resolved value
        # Implementation depends on how insights are structured
        pass


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

    def get_context_summary(self, project: ProjectContext) -> str:
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

    def _get_summary(self, request: Dict) -> Dict:
        project = request.get('project')
        summary = self.get_context_summary(project)
        return {'status': 'success', 'summary': summary}

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


class ConflictDetectorAgent(Agent):
    def __init__(self, orchestrator):
        super().__init__("ConflictDetector", orchestrator)

        # Define known conflicting technologies/concepts
        self.conflict_rules = {
            'databases': ['mysql', 'postgresql', 'sqlite', 'mongodb', 'redis'],
            'frontend_frameworks': ['react', 'vue', 'angular', 'svelte'],
            'backend_frameworks': ['django', 'flask', 'fastapi', 'express'],
            'languages': ['python', 'javascript', 'java', 'go', 'rust'],
            'deployment': ['aws', 'azure', 'gcp', 'heroku', 'vercel'],
            'mobile': ['react native', 'flutter', 'native ios', 'native android']
        }

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get('action')

        if action == 'detect_conflicts':
            return self._detect_conflicts(request)
        elif action == 'resolve_conflict':
            return self._resolve_conflict(request)
        elif action == 'get_suggestions':
            return self._get_conflict_suggestions(request)

        return {'status': 'error', 'message': 'Unknown action'}

    def _detect_conflicts(self, request: Dict) -> Dict:
        """Detect conflicts in new insights against existing project context"""
        project = request.get('project')
        new_insights = request.get('new_insights')
        current_user = request.get('current_user')

        conflicts = []

        if not new_insights or not isinstance(new_insights, dict):
            return {'status': 'success', 'conflicts': []}

        # Check each type of specification for conflicts
        conflicts.extend(self._check_tech_stack_conflicts(project, new_insights, current_user))
        conflicts.extend(self._check_requirements_conflicts(project, new_insights, current_user))
        conflicts.extend(self._check_goals_conflicts(project, new_insights, current_user))
        conflicts.extend(self._check_constraints_conflicts(project, new_insights, current_user))

        return {'status': 'success', 'conflicts': conflicts}

    def _check_tech_stack_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        """Check for technology stack conflicts"""
        conflicts = []
        new_tech = new_insights.get('tech_stack', [])
        if not isinstance(new_tech, list):
            new_tech = [new_tech] if new_tech else []

        for new_item in new_tech:
            if not new_item:
                continue

            new_item_lower = new_item.lower()

            # Check against existing tech stack
            for existing_item in project.tech_stack:
                existing_lower = existing_item.lower()

                # Check if they belong to same conflicting category
                conflict_category = self._find_conflict_category(new_item_lower, existing_lower)
                if conflict_category:
                    # Find who added the original (simplified - assume owner for now)
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

    def _check_requirements_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        """Check for requirement conflicts"""
        conflicts = []
        new_requirements = new_insights.get('requirements', [])
        if not isinstance(new_requirements, list):
            new_requirements = [new_requirements] if new_requirements else []

        # Use Claude to detect semantic conflicts in requirements
        for new_req in new_requirements:
            if not new_req:
                continue

            semantic_conflicts = self._check_semantic_conflicts(new_req, project.requirements, 'requirements')
            for conflict_data in semantic_conflicts:
                conflict = ConflictInfo(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type='requirements',
                    old_value=conflict_data['existing'],
                    new_value=new_req,
                    old_author=self._find_spec_author(project, 'requirements', conflict_data['existing']),
                    new_author=current_user,
                    old_timestamp=project.created_at.isoformat(),
                    new_timestamp=datetime.datetime.now().isoformat(),
                    severity=conflict_data['severity'],
                    suggestions=conflict_data['suggestions']
                )
                conflicts.append(conflict)

        return conflicts

    def _check_goals_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        """Check for goal conflicts"""
        conflicts = []
        new_goals = new_insights.get('goals', '')

        if new_goals and project.goals and new_goals.lower() != project.goals.lower():
            # Use Claude to determine if goals actually conflict
            conflict_data = self._check_semantic_conflicts(new_goals, [project.goals], 'goals')
            if conflict_data:
                conflict = ConflictInfo(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type='goals',
                    old_value=project.goals,
                    new_value=new_goals,
                    old_author=self._find_spec_author(project, 'goals', project.goals),
                    new_author=current_user,
                    old_timestamp=project.created_at.isoformat(),
                    new_timestamp=datetime.datetime.now().isoformat(),
                    severity='high',
                    suggestions=conflict_data[0]['suggestions'] if conflict_data else []
                )
                conflicts.append(conflict)

        return conflicts

    def _check_constraints_conflicts(self, project: ProjectContext, new_insights: Dict, current_user: str) -> List[
        ConflictInfo]:
        """Check for constraint conflicts"""
        conflicts = []
        new_constraints = new_insights.get('constraints', [])
        if not isinstance(new_constraints, list):
            new_constraints = [new_constraints] if new_constraints else []

        for new_constraint in new_constraints:
            if not new_constraint:
                continue

            semantic_conflicts = self._check_semantic_conflicts(new_constraint, project.constraints, 'constraints')
            for conflict_data in semantic_conflicts:
                conflict = ConflictInfo(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type='constraints',
                    old_value=conflict_data['existing'],
                    new_value=new_constraint,
                    old_author=self._find_spec_author(project, 'constraints', conflict_data['existing']),
                    new_author=current_user,
                    old_timestamp=project.created_at.isoformat(),
                    new_timestamp=datetime.datetime.now().isoformat(),
                    severity=conflict_data['severity'],
                    suggestions=conflict_data['suggestions']
                )
                conflicts.append(conflict)

        return conflicts

    def _find_conflict_category(self, item1: str, item2: str) -> Optional[str]:
        """Find if two items belong to same conflicting category"""
        for category, items in self.conflict_rules.items():
            if any(item1 in tech.lower() for tech in items) and any(item2 in tech.lower() for tech in items):
                return category
        return None

    def _check_semantic_conflicts(self, new_item: str, existing_items: List[str], context_type: str) -> List[Dict]:
        """Use Claude to detect semantic conflicts"""
        if not existing_items:
            return []

        prompt = f"""Analyze if this new {context_type} conflicts with existing ones:

New {context_type}: "{new_item}"

Existing {context_type}:
{chr(10).join([f"- {item}" for item in existing_items])}

Determine if there are any conflicts. A conflict exists when:
1. They are mutually exclusive (can't both be true)
2. They contradict each other technically
3. They represent different approaches to the same problem

For each conflict found, return JSON format:
{{
  "conflicts": [
    {{
      "existing": "conflicting existing item",
      "severity": "high|medium|low",
      "reason": "brief explanation",
      "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
    }}
  ]
}}

If no conflicts, return: {{"conflicts": []}}
"""

        try:
            response = self.orchestrator.claude_client.client.messages.create(
                model=CONFIG['CLAUDE_MODEL'],
                max_tokens=800,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Parse JSON response
            import json
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()

            start = response_text.find('{')
            end = response_text.rfind('}') + 1

            if 0 <= start < end:
                json_text = response_text[start:end]
                result = json.loads(json_text)
                return result.get('conflicts', [])

        except Exception as e:
            self.log(f"Error in semantic conflict detection: {e}", "WARN")

        return []

    def _find_spec_author(self, project: ProjectContext, spec_type: str, spec_value: str) -> str:
        """Find who originally added a specification (simplified implementation)"""
        # This is a simplified version - in reality you'd need to track authorship in conversation history
        return project.owner  # Fallback to owner

    def _generate_tech_suggestions(self, category: str, old_tech: str, new_tech: str) -> List[str]:
        """Generate suggestions for technology conflicts"""
        suggestions = [
            f"Choose {old_tech} if you prefer stability and existing team expertise",
            f"Choose {new_tech} if you want newer features and better performance",
            f"Consider if both technologies can coexist in different parts of the system",
            "Research the specific advantages of each option for your use case"
        ]
        return suggestions


# Database and Storage Classes
class VectorDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("socratic_knowledge")
        self.embedding_model = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])
        self.knowledge_loaded = False  # FIX: Track if knowledge is already loaded

    def add_knowledge(self, entry: KnowledgeEntry):
        """Add knowledge entry to vector database"""
        # FIX: Check if entry already exists before adding
        try:
            existing = self.collection.get(ids=[entry.id])
            if existing['ids']:
                print(f"Knowledge entry '{entry.id}' already exists, skipping...")
                return
        except Exception:
            pass  # Entry doesn't exist, proceed with adding

        if not entry.embedding:
            entry.embedding = self.embedding_model.encode(entry.content).tolist()

        try:
            self.collection.add(
                documents=[entry.content],
                metadatas=[entry.metadata],
                ids=[entry.id],
                embeddings=[entry.embedding]
            )
            print(f"Added knowledge entry: {entry.id}")
        except Exception as e:
            print(f"Warning: Could not add knowledge entry {entry.id}: {e}")

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar knowledge entries"""
        if not query.strip():
            return []

        try:
            query_embedding = self.embedding_model.encode(query).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count())
            )

            if not results['documents'] or not results['documents'][0]:
                return []

            return [{
                'content': doc,
                'metadata': meta,
                'score': dist
            } for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )]
        except Exception as e:
            print(f"Warning: Search failed: {e}")
            return []

    def delete_entry(self, entry_id: str):
        """Delete knowledge entry"""
        try:
            self.collection.delete(ids=[entry_id])
        except Exception as e:
            print(f"Warning: Could not delete entry {entry_id}: {e}")


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

        # FIX: Use a simpler query that works with pickle data
        cursor.execute('SELECT project_id, data FROM projects', ())
        results = cursor.fetchall()
        conn.close()

        projects = []
        for project_id, data in results:
            try:
                project_data = pickle.loads(data)
                # Check if user is owner or collaborator
                if (project_data['owner'] == username or
                        username in project_data.get('collaborators', [])):
                    projects.append({
                        'project_id': project_id,
                        'name': project_data['name'],
                        'phase': project_data['phase'],
                        'updated_at': project_data['updated_at']
                    })
            except Exception as e:
                print(f"Warning: Could not load project {project_id}: {e}")

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

    def user_exists(self, username: str) -> bool:
        """Check if a user exists in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()

        conn.close()
        return result is not None


# Claude API Client
class ClaudeClient:
    def __init__(self, api_key: str, orchestrator: 'AgentOrchestrator'):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.orchestrator = orchestrator

    def extract_insights(self, user_response: str, project: ProjectContext) -> Dict:
        """Extract insights from user response using Claude"""

        # Handle empty or non-informative responses
        if not user_response or len(user_response.strip()) < 3:
            return {}

        # Handle common non-informative responses
        non_informative = ["i don't know", "idk", "not sure", "no idea", "dunno", "unsure"]
        if user_response.lower().strip() in non_informative:
            return {'note': 'User expressed uncertainty - may need more guidance'}

        # Build prompt using string concatenation to avoid brace issues
        prompt = f"""
        Analyze this user response in the context of their project and extract structured insights:

        Project Context:
        - Goals: {project.goals or 'Not specified'}
        - Phase: {project.phase}
        - Tech Stack: {', '.join(project.tech_stack) if project.tech_stack else 'Not specified'}

        User Response: "{user_response}"

        Please extract and return any mentions of:
        1. Goals or objectives
        2. Technical requirements 
        3. Technology preferences
        4. Constraints or limitations
        5. Team structure preferences

        """ + """Return as valid JSON with keys: goals, requirements, tech_stack, constraints, team_structure
        If no insights found, return empty JSON object {}.
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
                response_text = response.content[0].text.strip()

                # Clean up the response - sometimes Claude adds extra text
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    response_text = response_text.replace('```', '').strip()

                # Find JSON object in the response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1

                if 0 <= start < end:
                    json_text = response_text[start:end]
                    parsed_insights = json.loads(json_text)

                    # Ensure it's a dictionary
                    if isinstance(parsed_insights, dict):
                        return parsed_insights
                    else:
                        return {}
                else:
                    # No JSON found, return empty dict
                    return {}

            except (json.JSONDecodeError, ValueError, IndexError) as json_error:
                print(f"{Fore.YELLOW}Warning: Could not parse JSON response: {json_error}")
                # Fallback to simple text analysis
                return {'extracted_text': response.content[0].text}

        except Exception as e:
            print(f"{Fore.RED}Error extracting insights: {e}")
            return {}  # Return empty dict instead of None

    def generate_conflict_resolution_suggestions(self, conflict: ConflictInfo, project: ProjectContext) -> str:
        """Generate suggestions for resolving a specific conflict"""

        context_summary = self.orchestrator.context_analyzer.get_context_summary(project)

        prompt = f"""Help resolve this project specification conflict:

    Project: {project.name} ({project.phase} phase)
    Project Context: {context_summary}

    Conflict Details:
    - Type: {conflict.conflict_type}
    - Original: "{conflict.old_value}" (by {conflict.old_author})
    - New: "{conflict.new_value}" (by {conflict.new_author})
    - Severity: {conflict.severity}

    Provide 3-4 specific, actionable suggestions for resolving this conflict. Consider:
    1. Technical implications of each choice
    2. Project goals and constraints
    3. Team collaboration aspects
    4. Potential compromise solutions

    Format as:
    🔧 Conflict Resolution Suggestions:

    1. [First option with clear pros/cons]
    2. [Second option with rationale] 
    3. [Third option or compromise solution]
    4. [Additional perspective if helpful]

    Be specific and practical, not just theoretical."""

        try:
            response = self.client.messages.create(
                model=CONFIG['CLAUDE_MODEL'],
                max_tokens=600,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text.strip()

        except Exception as e:
            return f"Error generating suggestions: {e}"

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

    def test_connection(self):
        """Test connection to Claude API"""
        try:
            response = self.client.messages.create(
                model=CONFIG['CLAUDE_MODEL'],
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": "Test"}]
            )
            return True
        except Exception as e:
            raise e

    def _calculate_cost(self, usage) -> float:
        """Calculate estimated cost based on token usage"""
        # Claude 3 Sonnet pricing (approximate)
        input_cost_per_1k = 0.003  # $0.003 per 1K input tokens
        output_cost_per_1k = 0.015  # $0.015 per 1K output tokens

        input_cost = (usage.input_tokens / 1000) * input_cost_per_1k
        output_cost = (usage.output_tokens / 1000) * output_cost_per_1k

        return input_cost + output_cost

    def generate_socratic_question(self, prompt: str) -> str:
        """Generate a Socratic question using Claude"""
        try:
            response = self.client.messages.create(
                model=CONFIG['CLAUDE_MODEL'],
                max_tokens=200,  # Questions should be concise
                temperature=0.7,  # Some creativity for varied questions
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

            return response.content[0].text.strip()

        except Exception as e:
            print(f"{Fore.RED}Error generating Socratic question: {e}")
            raise e

    def generate_suggestions(self, current_question: str, project: ProjectContext) -> str:
        """Generate helpful suggestions when user can't answer a question"""

        # Get recent conversation for context
        recent_conversation = ""
        if project.conversation_history:
            recent_messages = project.conversation_history[-6:]  # Last 6 messages
            for msg in recent_messages:
                role = "Assistant" if msg['type'] == 'assistant' else "User"
                recent_conversation += f"{role}: {msg['content']}\n"

        # Get relevant knowledge from vector database
        relevant_knowledge = ""
        knowledge_results = self.orchestrator.vector_db.search_similar(current_question, top_k=3)
        if knowledge_results:
            relevant_knowledge = "\n".join([result['content'][:300] for result in knowledge_results])

        # Build context summary
        context_summary = self.orchestrator.context_analyzer.get_context_summary(project)

        prompt = f"""You are helping a developer who is stuck on a Socratic question about their software project.

    Project Details:
    - Name: {project.name}
    - Phase: {project.phase}
    - Context: {context_summary}

    Current Question They Can't Answer:
    "{current_question}"

    Recent Conversation:
    {recent_conversation}

    Relevant Knowledge:
    {relevant_knowledge}

    The user is having difficulty answering this question. Provide 3-4 helpful suggestions that:

    1. Give concrete examples or options they could consider
    2. Break down the question into smaller, easier parts
    3. Provide relevant industry examples or common approaches
    4. Suggest specific things they could research or think about

    Format your response as:
    Here are some suggestions to help you think through this:

    • [First suggestion with specific example]
    • [Second suggestion with actionable advice]
    • [Third suggestion with alternative perspective]
    • [Optional fourth suggestion]

    Keep suggestions practical, specific, and encouraging. Don't just ask more questions.
    """

        try:
            response = self.client.messages.create(
                model=CONFIG['CLAUDE_MODEL'],
                max_tokens=800,
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

            return response.content[0].text.strip()

        except Exception as e:
            # Fallback suggestions if Claude API fails
            fallback_suggestions = {
                'discovery': """Here are some suggestions to help you think through this:

    • Consider researching similar applications or tools in your problem domain
    • Think about specific pain points you've experienced that this could solve
    • Ask potential users what features would be most valuable to them
    • Look at existing solutions and identify what's missing or could be improved""",

                'analysis': """Here are some suggestions to help you think through this:

    • Break down the technical challenge into smaller, specific problems
    • Research what libraries or frameworks are commonly used for this type of project
    • Consider scalability, security, and performance requirements early
    • Look up case studies of similar technical implementations""",

                'design': """Here are some suggestions to help you think through this:

    • Start with a simple architecture and plan how to extend it later
    • Consider using established design patterns like MVC, Repository, or Factory
    • Think about how different components will communicate with each other
    • Sketch out the data flow and user interaction patterns""",

                'implementation': """Here are some suggestions to help you think through this:

    • Break the project into small, manageable milestones
    • Consider starting with a minimal viable version first
    • Think about your development environment and tooling needs
    • Plan your testing strategy alongside your implementation approach"""
            }

            return fallback_suggestions.get(project.phase,
                                            "Consider breaking the question into smaller parts and researching each "
                                            "aspect individually.")


# Agent Orchestrator
class AgentOrchestrator:
    def __init__(self, api_key: str):
        self.api_key = api_key

        # Initialize database components
        data_dir = CONFIG['DATA_DIR']
        os.makedirs(data_dir, exist_ok=True)

        self.database = ProjectDatabase(os.path.join(data_dir, 'projects.db'))
        self.vector_db = VectorDatabase(os.path.join(data_dir, 'vector_db'))

        # Initialize Claude client
        self.claude_client = ClaudeClient(api_key, self)
        self._initialize_agents()

        # Load default knowledge base
        self._load_knowledge_base()

        print(f"{Fore.GREEN}✓ Socratic RAG System v7.0 initialized successfully!")

    def _initialize_agents(self):
        """Initialize agents after orchestrator is fully set up"""
        self.project_manager = ProjectManagerAgent(self)
        self.socratic_counselor = SocraticCounselorAgent(self)
        self.context_analyzer = ContextAnalyzerAgent(self)
        self.code_generator = CodeGeneratorAgent(self)
        self.system_monitor = SystemMonitorAgent(self)
        self.conflict_detector = ConflictDetectorAgent(self)

    def _load_knowledge_base(self):
        """Load default knowledge base if not already loaded"""
        if self.vector_db.knowledge_loaded:
            return

        print(f"{Fore.YELLOW}Loading knowledge base...")

        default_knowledge = [
            {
                'id': 'software_architecture_patterns',
                'content': 'Common software architecture patterns include MVC (Model-View-Controller), '
                           'MVP (Model-View-Presenter), MVVM (Model-View-ViewModel), microservices architecture, '
                           'layered architecture, and event-driven architecture. Each pattern has specific use cases '
                           'and trade-offs.',
                'category': 'architecture',
                'metadata': {'topic': 'patterns', 'difficulty': 'intermediate'}
            },
            {
                'id': 'python_best_practices',
                'content': 'Python best practices include following PEP 8 style guide, using virtual environments, '
                           'writing docstrings, implementing proper error handling, using type hints, following the '
                           'principle of least privilege, and writing unit tests.',
                'category': 'python',
                'metadata': {'topic': 'best_practices', 'language': 'python'}
            },
            {
                'id': 'api_design_principles',
                'content': 'REST API design principles include using appropriate HTTP methods, meaningful resource '
                           'URLs, consistent naming conventions, proper status codes, versioning, authentication and '
                           'authorization, rate limiting, and comprehensive documentation.',
                'category': 'api_design',
                'metadata': {'topic': 'rest_api', 'difficulty': 'intermediate'}
            },
            {
                'id': 'database_design_basics',
                'content': 'Database design fundamentals include normalization, defining primary and foreign keys, '
                           'indexing strategy, choosing appropriate data types, avoiding SQL injection, implementing '
                           'proper backup strategies, and optimizing queries for performance.',
                'category': 'database',
                'metadata': {'topic': 'design', 'difficulty': 'beginner'}
            },
            {
                'id': 'security_considerations',
                'content': 'Security considerations in software development include input validation, authentication '
                           'and authorization, secure communication (HTTPS), data encryption, regular security '
                           'updates, logging and monitoring, and following the principle of least privilege.',
                'category': 'security',
                'metadata': {'topic': 'general_security', 'difficulty': 'intermediate'}
            }
        ]

        for knowledge_data in default_knowledge:
            entry = KnowledgeEntry(**knowledge_data)
            self.vector_db.add_knowledge(entry)

        self.vector_db.knowledge_loaded = True
        print(f"{Fore.GREEN}✓ Knowledge base loaded ({len(default_knowledge)} entries)")

    def process_request(self, agent_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate agent"""
        agents = {
            'project_manager': self.project_manager,
            'socratic_counselor': self.socratic_counselor,
            'context_analyzer': self.context_analyzer,
            'code_generator': self.code_generator,
            'system_monitor': self.system_monitor,
            'conflict_detector': self.conflict_detector  # ADD THIS LINE
        }

        agent = agents.get(agent_name)
        if agent:
            return agent.process(request)
        else:
            return {'status': 'error', 'message': f'Unknown agent: {agent_name}'}


# Main Application Class
class SocraticRAGSystem:
    def __init__(self):
        self.orchestrator = None
        self.current_user = None
        self.current_project = None
        self.session_start = datetime.datetime.now()

    def start(self):
        """Start the Socratic RAG System"""
        print(f"{Fore.CYAN}{Style.BRIGHT}")
        print("╔═══════════════════════════════════════════════╗")
        print("║        Enhanced Socratic RAG System           ║")
        print("║                Version 7.0                    ║")
        print("║Ουδέν οίδα, ούτε διδάσκω τι, αλλά διαπορώ μόνον║")
        print("╚═══════════════════════════════════════════════╝")
        print(f"{Style.RESET_ALL}")

        # Get API key
        api_key = self._get_api_key()
        if not api_key:
            print(f"{Fore.RED}No API key provided. Exiting...")
            return

        try:
            # Initialize orchestrator
            self.orchestrator = AgentOrchestrator(api_key)

            # Login or create user
            if not self._handle_user_authentication():
                return

            # Main loop
            self._main_loop()

        except Exception as e:
            print(f"{Fore.RED}System error: {e}")

    def _get_api_key(self) -> Optional[str]:
        """Get Claude API key from environment or user input"""
        api_key = os.getenv('API_KEY_CLAUDE')
        if not api_key:
            print(f"{Fore.YELLOW}Claude API key not found in environment.")
            api_key = getpass.getpass("Please enter your Claude API key: ")
        return api_key

    def _handle_user_authentication(self) -> bool:
        """Handle user login or registration"""
        while True:
            print(f"\n{Fore.CYAN}Authentication Options:")
            print("1. Login with existing account")
            print("2. Create new account")
            print("3. Exit")

            choice = input(f"{Fore.WHITE}Choose option (1-3): ").strip()

            if choice == '1':
                if self._login():
                    return True
            elif choice == '2':
                if self._create_account():
                    return True
            elif choice == '3':
                return False
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.")

    def _login(self) -> bool:
        """Handle user login"""
        username = input(f"{Fore.WHITE}Username: ").strip()
        if not username:
            print(f"{Fore.RED}Username cannot be empty.")
            return False

        passcode = input(f"{Fore.WHITE}Passcode: ").strip()
        if not passcode:
            print(f"{Fore.RED}Passcode cannot be empty.")
            return False

        # Load user from database
        user = self.orchestrator.database.load_user(username)
        if not user:
            print(f"{Fore.RED}User not found.")
            return False

        # Verify passcode
        passcode_hash = hashlib.sha256(passcode.encode()).hexdigest()
        if user.passcode_hash != passcode_hash:
            print(f"{Fore.RED}Invalid passcode.")
            return False

        self.current_user = user
        print(f"{Fore.GREEN}✓ Welcome back, {username}!")
        return True

    def _create_account(self) -> bool:
        """Handle account creation"""
        print(f"\n{Fore.CYAN}Create New Account")

        username = input(f"{Fore.WHITE}Username: ").strip()
        if not username:
            print(f"{Fore.RED}Username cannot be empty.")
            return False

        # Check if user already exists
        existing_user = self.orchestrator.database.load_user(username)
        if existing_user:
            print(f"{Fore.RED}Username already exists.")
            return False

        passcode = input(f"{Fore.WHITE}Passcode: ").strip()
        if not passcode:
            print(f"{Fore.RED}Passcode cannot be empty.")
            return False

        confirm_passcode = input(f"{Fore.WHITE}Confirm passcode: ").strip()
        if passcode != confirm_passcode:
            print(f"{Fore.RED}Passcodes do not match.")
            return False

        # Create user
        passcode_hash = hashlib.sha256(passcode.encode()).hexdigest()
        user = User(
            username=username,
            passcode_hash=passcode_hash,
            created_at=datetime.datetime.now(),
            projects=[]
        )

        self.orchestrator.database.save_user(user)
        self.current_user = user

        print(f"{Fore.GREEN}✓ Account created successfully! Welcome, {username}!")
        return True

    def _main_loop(self):
        """Main application loop"""
        while True:
            try:
                print(f"\n{Fore.CYAN}═" * 5)
                print(f"{Fore.CYAN}{Style.BRIGHT}Main Menu")
                print(f"{Fore.WHITE}Current User: {self.current_user.username}")
                if self.current_project:
                    print(f"Current Project: {self.current_project.name} ({self.current_project.phase})")

                print(f"\n{Fore.YELLOW}Options:")
                print("1. Create new project")
                print("2. Load existing project")
                print("3. Continue current project")
                print("4. Generate code")
                print("5. Manage collaborators")
                print("6. View system status")
                print("7. Switch user")
                print("8. Exit")

                choice = input(f"{Fore.WHITE}Choose option (1-8): ").strip()

                if choice == '1':
                    self._create_project()
                elif choice == '2':
                    self._load_project()
                elif choice == '3':
                    if self.current_project:
                        self._continue_project()
                    else:
                        print(f"{Fore.RED}No current project loaded.")
                elif choice == '4':
                    if self.current_project:
                        self._generate_code()
                    else:
                        print(f"{Fore.RED}No current project loaded.")
                elif choice == '5':
                    if self.current_project:
                        self._manage_collaborators()
                    else:
                        print(f"{Fore.RED}No current project loaded.")
                elif choice == '6':
                    self._show_system_status()
                elif choice == '7':
                    if self._handle_user_authentication():
                        self.current_project = None
                elif choice == '8':
                    print(f"{Fore.GREEN}           Thank you for using Socratic RAG System")
                    print(f"{Fore.GREEN}..τω Ασκληπιώ οφείλομεν αλετρυόνα, απόδοτε και μη αμελήσετε..")
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use option 7 to exit properly.")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}")

    def _create_project(self):
        """Create a new project"""
        print(f"\n{Fore.CYAN}Create New Project")

        project_name = input(f"{Fore.WHITE}Project name: ").strip()
        if not project_name:
            print(f"{Fore.RED}Project name cannot be empty.")
            return

        # Create project using orchestrator
        result = self.orchestrator.process_request('project_manager', {
            'action': 'create_project',
            'project_name': project_name,
            'owner': self.current_user.username
        })

        if result['status'] == 'success':
            self.current_project = result['project']
            print(f"{Fore.GREEN}✓ Project '{project_name}' created successfully!")
            self._continue_project()
        else:
            print(f"{Fore.RED}Error creating project: {result['message']}")

    def _load_project(self):
        """Load an existing project"""
        # Get user's projects
        result = self.orchestrator.process_request('project_manager', {
            'action': 'list_projects',
            'username': self.current_user.username
        })

        if result['status'] != 'success' or not result['projects']:
            print(f"{Fore.YELLOW}No projects found.")
            return

        print(f"\n{Fore.CYAN}Your Projects:")
        projects = result['projects']

        for i, project in enumerate(projects, 1):
            print(f"{i}. {project['name']} ({project['phase']}) - {project['updated_at']}")

        try:
            choice = int(input(f"{Fore.WHITE}Select project (1-{len(projects)}): ")) - 1
            if 0 <= choice < len(projects):
                project_id = projects[choice]['project_id']

                # Load project
                result = self.orchestrator.process_request('project_manager', {
                    'action': 'load_project',
                    'project_id': project_id
                })

                if result['status'] == 'success':
                    self.current_project = result['project']
                    print(f"{Fore.GREEN}✓ Project loaded successfully!")
                else:
                    print(f"{Fore.RED}Error loading project: {result['message']}")
            else:
                print(f"{Fore.RED}Invalid selection.")
        except ValueError:
            print(f"{Fore.RED}Invalid input.")

    def _continue_project(self):
        """Continue working on current project"""
        if not self.current_project:
            return

        print(f"\n{Fore.CYAN}Socratic Guidance Session")
        print(f"{Fore.WHITE}Project: {self.current_project.name}")
        print(f"Phase: {self.current_project.phase}")

        while True:
            result = self.orchestrator.process_request('socratic_counselor', {
                'action': 'process_response',
                'project': self.current_project,
                'response': response,  # TODO
                'current_user': self.current_user.username  # ADD THIS LINE
            })

            if result['status'] == 'success':
                if result.get('conflicts_pending'):
                    print(f"{Fore.YELLOW}⚠️  Some specifications were not added due to conflicts")
                elif result['insights']:
                    print(f"{Fore.GREEN}✓ Insights captured and integrated!")

            question = result['question']
            print(f"\n{Fore.BLUE}🤔 {question}")

            # Get user response
            print(
                f"{Fore.YELLOW}Your response (type 'done' to finish, 'advance' to move to next phase, 'help' for "
                f"suggestions):")
            response = input(f"{Fore.WHITE}> ").strip()

            if response.lower() == 'done':
                break
            elif response.lower() == 'advance':
                result = self.orchestrator.process_request('socratic_counselor', {
                    'action': 'advance_phase',
                    'project': self.current_project
                })
                if result['status'] == 'success':
                    print(f"{Fore.GREEN}✓ Advanced to {result['new_phase']} phase!")
                continue
            elif response.lower() in ['help', 'suggestions', 'options', 'hint']:
                # Generate suggestions automatically
                suggestions = self.orchestrator.claude_client.generate_suggestions(question, self.current_project)
                print(f"\n{Fore.MAGENTA}💡 {suggestions}")
                print(f"{Fore.YELLOW}Now, would you like to try answering the question?")
                continue
            elif not response:
                continue

    def _generate_code(self):
        """Generate code for current project"""
        if not self.current_project:
            return

        print(f"\n{Fore.CYAN}Generating Code...")

        result = self.orchestrator.process_request('code_generator', {
            'action': 'generate_script',
            'project': self.current_project
        })

        if result['status'] == 'success':
            script = result['script']
            print(f"\n{Fore.GREEN}✓ Code Generated Successfully!")
            print(f"{Fore.YELLOW}{'=' * 5}")
            print(f"{Fore.WHITE}{script}")
            print(f"{Fore.YELLOW}{'=' * 5}")

            # Ask if user wants documentation
            doc_choice = input(f"\n{Fore.CYAN}Generate documentation? (y/n): ").lower()
            if doc_choice == 'y':
                doc_result = self.orchestrator.process_request('code_generator', {
                    'action': 'generate_documentation',
                    'project': self.current_project,
                    'script': script
                })

                if doc_result['status'] == 'success':
                    print(f"\n{Fore.GREEN}✓ Documentation Generated!")
                    print(f"{Fore.YELLOW}{'=' * 5}")
                    print(f"{Fore.WHITE}{doc_result['documentation']}")
                    print(f"{Fore.YELLOW}{'=' * 5}")
        else:
            print(f"{Fore.RED}Error generating code: {result['message']}")

    def _show_system_status(self):
        """Show system status and statistics"""
        print(f"\n{Fore.CYAN}System Status")

        # Get system stats
        result = self.orchestrator.process_request('system_monitor', {
            'action': 'get_stats'
        })

        if result['status'] == 'success':
            stats = result
            print(f"{Fore.WHITE}Total Tokens Used: {stats['total_tokens']}")
            print(f"Estimated Cost: ${stats['total_cost']:.4f}")
            print(f"API Calls Made: {stats['api_calls']}")
            print(f"Connection Status: {'✓' if stats['connection_status'] else '✗'}")

        # Check for warnings
        result = self.orchestrator.process_request('system_monitor', {
            'action': 'check_limits'
        })

        if result['status'] == 'success' and result['warnings']:
            print(f"\n{Fore.YELLOW}Warnings:")
            for warning in result['warnings']:
                print(f"⚠ {warning}")

    def _manage_collaborators(self):
        """Manage project collaborators"""
        if not self.current_project:
            print(f"{Fore.RED}No current project loaded.")
            return

        while True:
            print(f"\n{Fore.CYAN}Collaborator Management")
            print(f"{Fore.WHITE}Project: {self.current_project.name}")

            # Show current collaborators
            result = self.orchestrator.process_request('project_manager', {
                'action': 'list_collaborators',
                'project': self.current_project
            })

            if result['status'] == 'success':
                print(f"\n{Fore.YELLOW}Current Team:")
                for member in result['collaborators']:
                    role_color = Fore.GREEN if member['role'] == 'owner' else Fore.WHITE
                    print(f"{role_color}  • {member['username']} ({member['role']})")

            print(f"\n{Fore.YELLOW}Options:")
            print("1. Add collaborator")
            print("2. Remove collaborator")
            print("3. Back to main menu")

            choice = input(f"{Fore.WHITE}Choose option (1-3): ").strip()

            if choice == '1':
                self._add_collaborator_ui()
            elif choice == '2':
                self._remove_collaborator_ui()
            elif choice == '3':
                break
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.")

    def _add_collaborator_ui(self):
        """UI for adding collaborators"""
        # Only owner can add collaborators
        if self.current_user.username != self.current_project.owner:
            print(f"{Fore.RED}Only the project owner can add collaborators.")
            return

        username = input(f"{Fore.WHITE}Username to add: ").strip()
        if not username:
            print(f"{Fore.RED}Username cannot be empty.")
            return

        # Check if user exists
        if not self.orchestrator.database.user_exists(username):
            print(f"{Fore.RED}User '{username}' does not exist in the system.")
            return

        # Check if already owner
        if username == self.current_project.owner:
            print(f"{Fore.RED}User is already the project owner.")
            return

        # Add collaborator
        result = self.orchestrator.process_request('project_manager', {
            'action': 'add_collaborator',
            'project': self.current_project,
            'username': username
        })

        if result['status'] == 'success':
            print(f"{Fore.GREEN}✓ Added '{username}' as collaborator!")
        else:
            print(f"{Fore.RED}Error: {result['message']}")

    def _remove_collaborator_ui(self):
        """UI for removing collaborators"""
        # Only owner can remove collaborators
        if self.current_user.username != self.current_project.owner:
            print(f"{Fore.RED}Only the project owner can remove collaborators.")
            return

        if not self.current_project.collaborators:
            print(f"{Fore.YELLOW}No collaborators to remove.")
            return

        print(f"\n{Fore.YELLOW}Current Collaborators:")
        for i, collaborator in enumerate(self.current_project.collaborators, 1):
            print(f"{i}. {collaborator}")

        try:
            choice = int(
                input(f"{Fore.WHITE}Select collaborator to remove (1-{len(self.current_project.collaborators)}): ")) - 1
            if 0 <= choice < len(self.current_project.collaborators):
                username = self.current_project.collaborators[choice]

                confirm = input(f"{Fore.YELLOW}Remove '{username}'? (y/n): ").lower()
                if confirm == 'y':
                    result = self.orchestrator.process_request('project_manager', {
                        'action': 'remove_collaborator',
                        'project': self.current_project,
                        'username': username,
                        'requester': self.current_user.username
                    })

                    if result['status'] == 'success':
                        print(f"{Fore.GREEN}✓ Removed '{username}' from project!")
                    else:
                        print(f"{Fore.RED}Error: {result['message']}")
            else:
                print(f"{Fore.RED}Invalid selection.")
        except ValueError:
            print(f"{Fore.RED}Invalid input.")


# Entry Point
def main():
    try:
        system = SocraticRAGSystem()
        system.start()
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
