#!/usr/bin/env python3
"""
User Simulation Agent for Socratic RAG System
Simulates diverse user behaviors for comprehensive testing and validation
"""

import random
import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import datetime


class UserPersonality(Enum):
    """Different user personality types for simulation"""
    NOVICE_EXPLORER = "novice_explorer"  # New to development, asks basic questions
    EXPERIENCED_ARCHITECT = "experienced_architect"  # Knows what they want, technical
    COLLABORATIVE_TEAM_LEAD = "collaborative_team_lead"  # Focuses on team dynamics
    PERFECTIONIST_DEVELOPER = "perfectionist_developer"  # Very detailed, changes mind often
    IMPATIENT_STARTUP = "impatient_startup"  # Wants quick results, minimal process
    ACADEMIC_RESEARCHER = "academic_researcher"  # Thorough, theoretical approach
    ENTERPRISE_MANAGER = "enterprise_manager"  # Process-focused, compliance-aware


class InteractionPattern(Enum):
    """Different interaction patterns users might exhibit"""
    LINEAR_PROGRESSION = "linear"  # Follows suggested flow
    EXPLORATORY = "exploratory"  # Jumps around topics
    COLLABORATIVE = "collaborative"  # Works with others frequently
    ITERATIVE_REFINEMENT = "iterative"  # Makes many small changes
    GOAL_FOCUSED = "goal_focused"  # Direct path to objectives


@dataclass
class UserSimulationProfile:
    """Profile defining how a simulated user behaves"""
    user_id: str
    personality: UserPersonality
    interaction_pattern: InteractionPattern
    technical_expertise: int = field(default=5)  # 1-10 scale
    collaboration_tendency: float = field(default=0.5)  # 0-1 probability
    change_frequency: float = field(default=0.3)  # How often they change requirements
    response_time_range: tuple = field(default=(1, 5))  # Seconds between interactions
    preferred_project_types: List[str] = field(default_factory=list)
    typical_goals: List[str] = field(default_factory=list)
    common_tech_stack: List[str] = field(default_factory=list)


class UserSimulationAgent:
    """Agent that simulates realistic user behavior patterns"""

    def __init__(self, orchestrator, simulation_profiles: Optional[List[UserSimulationProfile]] = None):
        self.orchestrator = orchestrator
        self.simulation_profiles = simulation_profiles or self._create_default_profiles()
        self.active_simulations = {}
        self.simulation_history = []

    def _create_default_profiles(self) -> List[UserSimulationProfile]:
        """Create default user simulation profiles"""
        return [
            UserSimulationProfile(
                user_id="novice_alice",
                personality=UserPersonality.NOVICE_EXPLORER,
                interaction_pattern=InteractionPattern.EXPLORATORY,
                technical_expertise=2,
                collaboration_tendency=0.8,
                change_frequency=0.6,
                preferred_project_types=["web_app", "mobile_app"],
                typical_goals=["learn programming", "build first project", "understand basics"],
                common_tech_stack=["python", "html", "css"]
            ),
            UserSimulationProfile(
                user_id="architect_bob",
                personality=UserPersonality.EXPERIENCED_ARCHITECT,
                interaction_pattern=InteractionPattern.GOAL_FOCUSED,
                technical_expertise=9,
                collaboration_tendency=0.3,
                change_frequency=0.1,
                preferred_project_types=["enterprise_system", "microservices", "cloud_native"],
                typical_goals=["scalable architecture", "performance optimization", "security"],
                common_tech_stack=["kubernetes", "microservices", "postgresql", "redis"]
            ),
            UserSimulationProfile(
                user_id="teamlead_carol",
                personality=UserPersonality.COLLABORATIVE_TEAM_LEAD,
                interaction_pattern=InteractionPattern.COLLABORATIVE,
                technical_expertise=7,
                collaboration_tendency=0.9,
                change_frequency=0.4,
                preferred_project_types=["team_project", "agile_development"],
                typical_goals=["team coordination", "project management", "code quality"],
                common_tech_stack=["react", "node.js", "docker", "jenkins"]
            ),
            UserSimulationProfile(
                user_id="perfectionist_dave",
                personality=UserPersonality.PERFECTIONIST_DEVELOPER,
                interaction_pattern=InteractionPattern.ITERATIVE_REFINEMENT,
                technical_expertise=8,
                collaboration_tendency=0.2,
                change_frequency=0.8,
                preferred_project_types=["high_quality_system", "detailed_architecture"],
                typical_goals=["perfect code quality", "comprehensive testing", "documentation"],
                common_tech_stack=["java", "spring", "junit", "sonarqube"]
            ),
            UserSimulationProfile(
                user_id="startup_eve",
                personality=UserPersonality.IMPATIENT_STARTUP,
                interaction_pattern=InteractionPattern.LINEAR_PROGRESSION,
                technical_expertise=4,
                collaboration_tendency=0.6,
                change_frequency=0.5,
                response_time_range=(0.5, 2),
                preferred_project_types=["mvp", "prototype", "quick_launch"],
                typical_goals=["fast development", "minimal viable product", "market validation"],
                common_tech_stack=["react", "node.js", "mongodb", "heroku"]
            )
        ]

    def start_simulation(self, profile: UserSimulationProfile, scenario: Dict[str, Any]) -> str:
        """Start a user behavior simulation"""
        simulation_id = str(uuid.uuid4())

        simulation_context = {
            'simulation_id': simulation_id,
            'profile': profile,
            'scenario': scenario,
            'current_step': 0,
            'project': None,
            'conversation_history': [],
            'start_time': datetime.datetime.now(),
            'status': 'active'
        }

        self.active_simulations[simulation_id] = simulation_context

        # Initialize user in the system
        self._create_simulated_user(profile)

        print(f"🎭 Started simulation for {profile.personality.value} user: {profile.user_id}")
        return simulation_id

    def _create_simulated_user(self, profile: UserSimulationProfile):
        """Create a user account for simulation"""
        from Socratic7 import User
        import hashlib

        user = User(
            username=profile.user_id,
            passcode_hash=hashlib.sha256(f"sim_{profile.user_id}".encode()).hexdigest(),
            created_at=datetime.datetime.now(),
            projects=[]
        )

        self.orchestrator.database.save_user(user)

    def simulate_user_session(self, simulation_id: str, max_interactions: int = 20) -> Dict[str, Any]:
        """Simulate a complete user session"""
        if simulation_id not in self.active_simulations:
            return {'error': 'Simulation not found'}

        context = self.active_simulations[simulation_id]
        profile = context['profile']

        session_log = {
            'simulation_id': simulation_id,
            'profile': profile.user_id,
            'personality': profile.personality.value,
            'interactions': [],
            'outcomes': {},
            'conflicts_encountered': 0,
            'collaboration_events': 0
        }

        try:
            # Phase 1: Project Creation
            project_result = self._simulate_project_creation(context)
            session_log['interactions'].append(project_result)

            if project_result['success']:
                context['project'] = project_result['project']

                # Phase 2: Requirements Gathering
                for i in range(max_interactions):
                    if context['status'] != 'active':
                        break

                    interaction = self._simulate_interaction(context, i)
                    session_log['interactions'].append(interaction)

                    # Update metrics
                    if 'conflict' in interaction.get('type', ''):
                        session_log['conflicts_encountered'] += 1
                    if 'collaboration' in interaction.get('type', ''):
                        session_log['collaboration_events'] += 1

                    # Simulate thinking time
                    time.sleep(random.uniform(*profile.response_time_range))

                    # Check if user would terminate session
                    if self._should_end_session(context, i):
                        context['status'] = 'completed'
                        break

                # Phase 3: Final Assessment
                session_log['outcomes'] = self._assess_session_outcomes(context)

        except Exception as e:
            session_log['error'] = str(e)
            context['status'] = 'error'

        # Store session history
        self.simulation_history.append(session_log)
        return session_log

    def _simulate_project_creation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate user creating a project"""
        profile = context['profile']

        # Generate project based on profile preferences
        project_type = random.choice(profile.preferred_project_types)
        project_name = f"{profile.user_id}_{project_type}_{random.randint(1000, 9999)}"

        result = self.orchestrator.process_request('project_manager', {
            'action': 'create_project',
            'project_name': project_name,
            'owner': profile.user_id
        })

        return {
            'type': 'project_creation',
            'timestamp': datetime.datetime.now(),
            'success': result.get('status') == 'success',
            'project': result.get('project'),
            'details': f"Created {project_type} project: {project_name}"
        }

    def _simulate_interaction(self, context: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Simulate a single user interaction"""
        profile = context['profile']
        project = context['project']

        # Determine interaction type based on personality and pattern
        interaction_type = self._choose_interaction_type(profile, step)

        if interaction_type == 'requirements_update':
            return self._simulate_requirements_update(context)
        elif interaction_type == 'tech_stack_change':
            return self._simulate_tech_stack_change(context)
        elif interaction_type == 'collaboration_request':
            return self._simulate_collaboration_request(context)
        elif interaction_type == 'question_response':
            return self._simulate_question_response(context)
        elif interaction_type == 'conflict_resolution':
            return self._simulate_conflict_resolution(context)
        else:
            return self._simulate_general_update(context)

    def _choose_interaction_type(self, profile: UserSimulationProfile, step: int) -> str:
        """Choose what type of interaction this user would make"""
        weights = {
            'requirements_update': 0.3,
            'tech_stack_change': 0.2,
            'collaboration_request': profile.collaboration_tendency * 0.3,
            'question_response': 0.4,
            'conflict_resolution': 0.1,
            'general_update': 0.2
        }

        # Adjust weights based on personality
        if profile.personality == UserPersonality.PERFECTIONIST_DEVELOPER:
            weights['requirements_update'] *= 2
            weights['tech_stack_change'] *= 1.5
        elif profile.personality == UserPersonality.COLLABORATIVE_TEAM_LEAD:
            weights['collaboration_request'] *= 2
        elif profile.personality == UserPersonality.IMPATIENT_STARTUP:
            weights['question_response'] *= 2
            weights['requirements_update'] *= 0.5

        # Choose based on weights
        choices = list(weights.keys())
        probabilities = list(weights.values())
        return random.choices(choices, weights=probabilities)[0]

    def _simulate_requirements_update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate user updating project requirements"""
        profile = context['profile']
        project = context['project']

        # Generate realistic requirements based on profile
        new_requirements = self._generate_requirements(profile)

        # Add to project
        if hasattr(project, 'requirements'):
            project.requirements.extend(new_requirements)

        return {
            'type': 'requirements_update',
            'timestamp': datetime.datetime.now(),
            'success': True,
            'details': f"Added requirements: {', '.join(new_requirements)}",
            'new_requirements': new_requirements
        }

    def _simulate_tech_stack_change(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate user changing technology stack"""
        profile = context['profile']
        project = context['project']

        # Might introduce conflicts
        new_tech = random.choice(profile.common_tech_stack)

        if hasattr(project, 'tech_stack'):
            old_stack = project.tech_stack.copy()
            project.tech_stack.append(new_tech)

            # Check for conflicts
            conflict_result = self.orchestrator.process_request('conflict_detector', {
                'action': 'detect_conflicts',
                'project': project,
                'new_insights': {'tech_stack': [new_tech]},
                'current_user': profile.user_id
            })

            conflicts_found = len(conflict_result.get('conflicts', []))

            return {
                'type': 'tech_stack_change',
                'timestamp': datetime.datetime.now(),
                'success': True,
                'details': f"Added {new_tech} to tech stack",
                'conflicts_detected': conflicts_found,
                'old_stack': old_stack,
                'new_tech': new_tech
            }

        return {'type': 'tech_stack_change', 'success': False, 'details': 'No tech stack to modify'}

    def _simulate_collaboration_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate user requesting collaboration"""
        profile = context['profile']
        project = context['project']

        # Pick another user from available profiles
        other_profiles = [p for p in self.simulation_profiles if p.user_id != profile.user_id]
        if not other_profiles:
            return {'type': 'collaboration_request', 'success': False, 'details': 'No other users available'}

        collaborator = random.choice(other_profiles)

        result = self.orchestrator.process_request('project_manager', {
            'action': 'add_collaborator',
            'project': project,
            'username': collaborator.user_id
        })

        return {
            'type': 'collaboration_request',
            'timestamp': datetime.datetime.now(),
            'success': result.get('status') == 'success',
            'details': f"Added collaborator: {collaborator.user_id}",
            'collaborator': collaborator.user_id
        }

    def _simulate_question_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate user responding to Socratic questions"""
        profile = context['profile']
        project = context['project']

        # Get a question from the Socratic counselor
        question_result = self.orchestrator.process_request('socratic_counselor', {
            'action': 'generate_question',
            'project': project
        })

        if question_result.get('status') == 'success':
            question = question_result.get('question', 'What are your goals for this project?')

            # Generate response based on personality
            response = self._generate_response(profile, question)

            # Process the response
            process_result = self.orchestrator.process_request('socratic_counselor', {
                'action': 'process_response',
                'project': project,
                'response': response,
                'current_user': profile.user_id
            })

            return {
                'type': 'question_response',
                'timestamp': datetime.datetime.now(),
                'success': process_result.get('status') == 'success',
                'question': question,
                'response': response,
                'insights_gained': process_result.get('insights', {})
            }

        return {'type': 'question_response', 'success': False, 'details': 'Could not generate question'}

    def _generate_requirements(self, profile: UserSimulationProfile) -> List[str]:
        """Generate realistic requirements based on user profile"""
        requirement_pools = {
            UserPersonality.NOVICE_EXPLORER: [
                "user-friendly interface", "simple navigation", "help documentation", "tutorials"
            ],
            UserPersonality.EXPERIENCED_ARCHITECT: [
                "high scalability", "microservices architecture", "caching layer", "load balancing"
            ],
            UserPersonality.COLLABORATIVE_TEAM_LEAD: [
                "team collaboration features", "project tracking", "code review process", "communication tools"
            ],
            UserPersonality.PERFECTIONIST_DEVELOPER: [
                "comprehensive testing", "code quality metrics", "detailed documentation", "error handling"
            ],
            UserPersonality.IMPATIENT_STARTUP: [
                "fast development", "minimal viable product", "quick deployment", "cost-effective"
            ]
        }

        pool = requirement_pools.get(profile.personality, ["basic functionality", "good performance"])
        return random.sample(pool, min(2, len(pool)))

    def _generate_response(self, profile: UserSimulationProfile, question: str) -> str:
        """Generate realistic response based on user personality"""
        response_templates = {
            UserPersonality.NOVICE_EXPLORER: [
                "I'm not sure about the technical details, but I want something that works well.",
                "Can you explain more about the options? I'm still learning.",
                "I think I need something simple to start with."
            ],
            UserPersonality.EXPERIENCED_ARCHITECT: [
                "I need a solution that can handle enterprise-level traffic and scale horizontally.",
                "The architecture should follow microservices patterns with proper service mesh.",
                "Performance and reliability are critical requirements for this system."
            ],
            UserPersonality.COLLABORATIVE_TEAM_LEAD: [
                "We need to consider how the whole team will work with this.",
                "Let me check with my team members about their preferences.",
                "How will this support our collaborative development process?"
            ],
            UserPersonality.PERFECTIONIST_DEVELOPER: [
                "I want to make sure we consider all edge cases and potential issues.",
                "The solution needs to be thoroughly tested and well-documented.",
                "Are there any quality assurance concerns we should address?"
            ],
            UserPersonality.IMPATIENT_STARTUP: [
                "We need to move fast on this. What's the quickest approach?",
                "Can we start with an MVP and iterate from there?",
                "Time to market is crucial for our startup."
            ]
        }

        templates = response_templates.get(profile.personality, ["That sounds good to me."])
        return random.choice(templates)

    def _simulate_conflict_resolution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate how user handles conflicts"""
        profile = context['profile']

        # Simulate different conflict resolution approaches
        resolution_approaches = {
            UserPersonality.NOVICE_EXPLORER: "ask_for_help",
            UserPersonality.EXPERIENCED_ARCHITECT: "technical_analysis",
            UserPersonality.COLLABORATIVE_TEAM_LEAD: "team_discussion",
            UserPersonality.PERFECTIONIST_DEVELOPER: "detailed_research",
            UserPersonality.IMPATIENT_STARTUP: "quick_decision"
        }

        approach = resolution_approaches.get(profile.personality, "default_handling")

        return {
            'type': 'conflict_resolution',
            'timestamp': datetime.datetime.now(),
            'success': True,
            'approach': approach,
            'details': f"User handled conflict using {approach} approach"
        }

    def _simulate_general_update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate general project updates"""
        profile = context['profile']
        project = context['project']

        updates = [
            "Updated project goals",
            "Refined project scope",
            "Added constraints",
            "Modified timeline expectations"
        ]

        update = random.choice(updates)

        return {
            'type': 'general_update',
            'timestamp': datetime.datetime.now(),
            'success': True,
            'details': update
        }

    def _should_end_session(self, context: Dict[str, Any], current_step: int) -> bool:
        """Determine if user would end the session"""
        profile = context['profile']

        # Different personalities have different session lengths
        max_steps = {
            UserPersonality.NOVICE_EXPLORER: 15,
            UserPersonality.EXPERIENCED_ARCHITECT: 25,
            UserPersonality.COLLABORATIVE_TEAM_LEAD: 20,
            UserPersonality.PERFECTIONIST_DEVELOPER: 30,
            UserPersonality.IMPATIENT_STARTUP: 10
        }

        max_for_personality = max_steps.get(profile.personality, 20)

        # Random chance to end early
        if current_step > 5:
            end_probability = (current_step / max_for_personality) * 0.1
            if random.random() < end_probability:
                return True

        return current_step >= max_for_personality

    def _assess_session_outcomes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the outcomes of a simulation session"""
        project = context['project']
        interactions = context.get('conversation_history', [])

        return {
            'project_completion_score': self._calculate_completion_score(project),
            'user_satisfaction_estimate': self._estimate_user_satisfaction(context),
            'collaboration_success': len(project.collaborators) if hasattr(project, 'collaborators') else 0,
            'requirements_gathered': len(project.requirements) if hasattr(project, 'requirements') else 0,
            'tech_stack_finalized': len(project.tech_stack) if hasattr(project, 'tech_stack') else 0,
            'session_duration': len(interactions)
        }

    def _calculate_completion_score(self, project) -> float:
        """Calculate how complete the project definition is"""
        score = 0.0
        total_aspects = 7

        if hasattr(project, 'goals') and project.goals:
            score += 1
        if hasattr(project, 'requirements') and project.requirements:
            score += 1
        if hasattr(project, 'tech_stack') and project.tech_stack:
            score += 1
        if hasattr(project, 'constraints') and project.constraints:
            score += 1
        if hasattr(project, 'team_structure') and project.team_structure:
            score += 1
        if hasattr(project, 'deployment_target') and project.deployment_target:
            score += 1
        if hasattr(project, 'phase') and project.phase != 'discovery':
            score += 1

        return score / total_aspects

    def _estimate_user_satisfaction(self, context: Dict[str, Any]) -> float:
        """Estimate user satisfaction based on session dynamics"""
        profile = context['profile']

        # Base satisfaction varies by personality
        base_satisfaction = {
            UserPersonality.NOVICE_EXPLORER: 0.7,
            UserPersonality.EXPERIENCED_ARCHITECT: 0.8,
            UserPersonality.COLLABORATIVE_TEAM_LEAD: 0.75,
            UserPersonality.PERFECTIONIST_DEVELOPER: 0.6,  # Harder to satisfy
            UserPersonality.IMPATIENT_STARTUP: 0.65
        }

        satisfaction = base_satisfaction.get(profile.personality, 0.7)

        # Adjust based on session outcomes
        # This is a simplified model - could be much more sophisticated
        return min(1.0, max(0.0, satisfaction + random.uniform(-0.2, 0.2)))

    def get_simulation_analytics(self) -> Dict[str, Any]:
        """Get analytics from all completed simulations"""
        if not self.simulation_history:
            return {'error': 'No simulation history available'}

        analytics = {
            'total_simulations': len(self.simulation_history),
            'personality_distribution': {},
            'average_session_length': 0,
            'common_interaction_types': {},
            'conflict_frequency': 0,
            'collaboration_success_rate': 0,
            'completion_scores': []
        }

        # Analyze patterns
        total_interactions = 0
        total_conflicts = 0
        successful_collaborations = 0

        for session in self.simulation_history:
            # Count personalities
            personality = session.get('personality', 'unknown')
            analytics['personality_distribution'][personality] = \
                analytics['personality_distribution'].get(personality, 0) + 1

            # Track interaction patterns
            interactions = session.get('interactions', [])
            total_interactions += len(interactions)

            for interaction in interactions:
                itype = interaction.get('type', 'unknown')
                analytics['common_interaction_types'][itype] = \
                    analytics['common_interaction_types'].get(itype, 0) + 1

            # Count conflicts and collaborations
            total_conflicts += session.get('conflicts_encountered', 0)
            if session.get('collaboration_events', 0) > 0:
                successful_collaborations += 1

        analytics['average_session_length'] = total_interactions / len(self.simulation_history)
        analytics['conflict_frequency'] = total_conflicts / len(self.simulation_history)
        analytics['collaboration_success_rate'] = successful_collaborations / len(self.simulation_history)

        return analytics

    def run_multi_user_scenario(self, scenario_name: str, participants: List[str]) -> Dict[str, Any]:
        """Run a scenario with multiple simulated users interacting"""
        print(f"🎯 Starting multi-user scenario: {scenario_name}")

        scenario_results = {
            'scenario_name': scenario_name,
            'participants': participants,
            'start_time': datetime.datetime.now(),
            'interactions': [],
            'conflicts': [],
            'collaborations': [],
            'final_state': {}
        }

        # Get profiles for participants
        participant_profiles = [p for p in self.simulation_profiles if p.user_id in participants]

        if len(participant_profiles) != len(participants):
            return {'error': 'Some participants not found in simulation profiles'}

        # Create shared project
        lead_profile = participant_profiles[0]
        project_result = self.orchestrator.process_request('project_manager', {
            'action': 'create_project',
            'project_name': f'MultiUser_{scenario_name}_{random.randint(1000, 9999)}',
            'owner': lead_profile.user_id
        })

        if project_result.get('status') != 'success':
            return {'error': 'Failed to create shared project'}

        shared_project = project_result['project']

        # Add all participants as collaborators
        for profile in participant_profiles[1:]:
            self.orchestrator.process_request('project_manager', {
                'action': 'add_collaborator',
                'project': shared_project,
                'username': profile.user_id
            })

        # Simulate interactions between users
        for round_num in range(10):  # 10 rounds of interactions
            for profile in participant_profiles:
                # Each user takes a turn
                interaction = self._simulate_multi_user_interaction(
                    profile, shared_project, scenario_results, round_num
                )
                scenario_results['interactions'].append(interaction)

                # Check for conflicts after each interaction
                conflicts = self._check_for_conflicts(shared_project, profile.user_id)
                if conflicts:
                    scenario_results['conflicts'].extend(conflicts)

                time.sleep(0.1)  # Brief pause between user actions

        scenario_results['end_time'] = datetime.datetime.now()
        scenario_results['final_state'] = self._capture_final_project_state(shared_project)

        return scenario_results

    def _simulate_multi_user_interaction(self, profile: UserSimulationProfile,
                                         shared_project, scenario_results: Dict,
                                         round_num: int) -> Dict[str, Any]:
        """Simulate one user's interaction in a multi-user scenario"""
        interaction_types = ['add_requirement', 'modify_tech_stack', 'update_goals', 'ask_question']
        interaction_type = random.choice(interaction_types)

        if interaction_type == 'add_requirement':
            new_req = f"{profile.personality.value} requirement from round {round_num}"
            if hasattr(shared_project, 'requirements'):
                shared_project.requirements.append(new_req)
            return {
                'user': profile.user_id,
                'type': 'add_requirement',
                'content': new_req,
                'timestamp': datetime.datetime.now()
            }

        elif interaction_type == 'modify_tech_stack':
            new_tech = random.choice(profile.common_tech_stack)
            if hasattr(shared_project, 'tech_stack'):
                shared_project.tech_stack.append(new_tech)
            return {
                'user': profile.user_id,
                'type': 'modify_tech_stack',
                'content': new_tech,
                'timestamp': datetime.datetime.now()
            }

        elif interaction_type == 'update_goals':
            goal_update = f"Goal update by {profile.user_id}: {random.choice(profile.typical_goals)}"
            if hasattr(shared_project, 'goals'):
                shared_project.goals = goal_update
            return {
                'user': profile.user_id,
                'type': 'update_goals',
                'content': goal_update,
                'timestamp': datetime.datetime.now()
            }

        else:  # ask_question
            return {
                'user': profile.user_id,
                'type': 'ask_question',
                'content': f"Question from {profile.personality.value} perspective",
                'timestamp': datetime.datetime.now()
            }

    def _check_for_conflicts(self, project, current_user: str) -> List[Dict[str, Any]]:
        """Check for conflicts in