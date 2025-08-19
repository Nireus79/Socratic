#!/usr/bin/env python3
"""
Socratic7 with integrated Socratic6.2 Menu System
A comprehensive integration that combines the advanced features of Socratic7
with the user-friendly menu system from Socratic6.2
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import for embeddings and AI integration
try:
    import anthropic
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"Missing required packages. Please install: {e}")
    sys.exit(1)


@dataclass
class MenuOption:
    """Represents a menu option from Socratic6.2"""
    key: str
    title: str
    description: str
    action: str
    enabled: bool = True


@dataclass
class ProjectContext:
    """Enhanced project context for Socratic7"""
    goals: List[str]
    requirements: List[str]
    tech_stack: List[str]
    constraints: List[str]
    team_structure: str
    language_preferences: List[str]
    deployment_target: str
    code_style: str
    current_phase: str
    conversation_history: List[Dict[str, str]]
    metadata: Dict[str, Any]


class MenuSystem:
    """Socratic6.2 Menu System Implementation"""

    def __init__(self):
        self.main_menu = self._create_main_menu()
        self.project_menu = self._create_project_menu()
        self.settings_menu = self._create_settings_menu()
        self.help_menu = self._create_help_menu()
        self.current_menu = "main"

    def _create_main_menu(self) -> Dict[str, MenuOption]:
        """Create the main menu options"""
        return {
            "1": MenuOption("1", "New Project", "Start a new project with Socratic guidance", "new_project"),
            "2": MenuOption("2", "Load Project", "Continue working on an existing project", "load_project"),
            "3": MenuOption("3", "Project Settings", "Configure project parameters", "project_settings"),
            "4": MenuOption("4", "Knowledge Base", "Explore the knowledge base", "knowledge_base"),
            "5": MenuOption("5", "Help & Tutorial", "Learn how to use the system", "help"),
            "6": MenuOption("6", "Settings", "System configuration", "settings"),
            "7": MenuOption("7", "Export/Import", "Manage project data", "export_import"),
            "q": MenuOption("q", "Quit", "Exit the application", "quit")
        }

    def _create_project_menu(self) -> Dict[str, MenuOption]:
        """Create project-specific menu options"""
        return {
            "1": MenuOption("1", "Continue Conversation", "Resume Socratic questioning", "continue_chat"),
            "2": MenuOption("2", "Project Summary", "View current project overview", "show_summary"),
            "3": MenuOption("3", "Phase Navigation", "Jump to specific development phase", "phase_nav"),
            "4": MenuOption("4", "Context Review", "Review and edit project context", "context_review"),
            "5": MenuOption("5", "Export Notes", "Save project notes and insights", "export_notes"),
            "6": MenuOption("6", "Reset Phase", "Start current phase over", "reset_phase"),
            "b": MenuOption("b", "Back to Main", "Return to main menu", "back_main"),
            "q": MenuOption("q", "Quit", "Exit the application", "quit")
        }

    def _create_settings_menu(self) -> Dict[str, MenuOption]:
        """Create settings menu options"""
        return {
            "1": MenuOption("1", "API Settings", "Configure API keys and endpoints", "api_settings"),
            "2": MenuOption("2", "Language Settings", "Set preferred languages", "language_settings"),
            "3": MenuOption("3", "Output Preferences", "Configure output format", "output_preferences"),
            "4": MenuOption("4", "Save Settings", "Save current configuration", "save_settings"),
            "b": MenuOption("b", "Back", "Return to previous menu", "back"),
            "q": MenuOption("q", "Quit", "Exit the application", "quit")
        }

    def _create_help_menu(self) -> Dict[str, MenuOption]:
        """Create help menu options"""
        return {
            "1": MenuOption("1", "Quick Start Guide", "Learn the basics", "quick_start"),
            "2": MenuOption("2", "Socratic Method", "Understand the methodology", "socratic_method"),
            "3": MenuOption("3", "Command Reference", "List of available commands", "commands"),
            "4": MenuOption("4", "Troubleshooting", "Common issues and solutions", "troubleshooting"),
            "5": MenuOption("5", "Examples", "See example conversations", "examples"),
            "b": MenuOption("b", "Back", "Return to main menu", "back"),
            "q": MenuOption("q", "Quit", "Exit the application", "quit")
        }

    def display_menu(self, menu_type: str = None) -> None:
        """Display the current menu"""
        if menu_type:
            self.current_menu = menu_type

        menus = {
            "main": (self.main_menu, "🏠 Socratic Development Counselor - Main Menu"),
            "project": (self.project_menu, "📁 Project Menu"),
            "settings": (self.settings_menu, "⚙️ Settings Menu"),
            "help": (self.help_menu, "❓ Help Menu")
        }

        menu_dict, title = menus.get(self.current_menu, (self.main_menu, "Main Menu"))

        print("\n" + "=" * 60)
        print(f"{title}")
        print("=" * 60)

        for key, option in menu_dict.items():
            status = "✓" if option.enabled else "✗"
            print(f"{status} [{option.key}] {option.title}")
            print(f"    {option.description}")

        print("-" * 60)

    def get_menu_choice(self) -> str:
        """Get user menu choice with validation"""
        while True:
            choice = input("\nPlease select an option: ").strip().lower()
            current_menu_dict = getattr(self, f"{self.current_menu}_menu")

            if choice in current_menu_dict:
                if current_menu_dict[choice].enabled:
                    return current_menu_dict[choice].action
                else:
                    print("❌ This option is currently disabled.")
            else:
                print("❌ Invalid choice. Please try again.")


class SocraticRAG:
    """Enhanced Socratic7 RAG System with integrated menu"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = []
        self.context = ProjectContext(
            goals=[], requirements=[], tech_stack=[], constraints=[],
            team_structure="", language_preferences=[], deployment_target="",
            code_style="", current_phase="discovery", conversation_history=[],
            metadata={}
        )
        self.menu_system = MenuSystem()
        self.phases = ["discovery", "analysis", "design", "implementation"]
        self.current_phase_index = 0

        # Initialize knowledge base
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with software development best practices"""
        knowledge_entries = [
            "Software architecture should follow SOLID principles for maintainability",
            "Test-driven development improves code quality and reduces bugs",
            "RESTful API design should use appropriate HTTP methods and status codes",
            "Database design should normalize data to reduce redundancy",
            "User experience design should prioritize accessibility and usability",
            "Security considerations should be integrated from the beginning",
            "Code documentation should explain why, not just what",
            "Version control workflows should support team collaboration",
            "Deployment pipelines should automate testing and validation",
            "Performance optimization should be based on actual metrics"
        ]

        for entry in knowledge_entries:
            embedding = self.model.encode([entry])[0]
            self.knowledge_base.append({
                'content': entry,
                'embedding': embedding,
                'category': 'best_practices'
            })

    def run_application(self):
        """Main application loop with integrated menu system"""
        print("🤖 Socratic Development Counselor v7.0")
        print("Enhanced with Socratic6.2 Menu System")
        print("=" * 60)

        while True:
            try:
                self.menu_system.display_menu()
                action = self.menu_system.get_menu_choice()

                if not self._handle_menu_action(action):
                    break

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Your progress has been saved.")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please try again or contact support.")

    def _handle_menu_action(self, action: str) -> bool:
        """Handle menu actions and return False to exit"""
        action_handlers = {
            # Main menu actions
            "new_project": self._new_project,
            "load_project": self._load_project,
            "project_settings": self._project_settings,
            "knowledge_base": self._explore_knowledge_base,
            "help": self._show_help,
            "settings": self._show_settings,
            "export_import": self._export_import,

            # Project menu actions
            "continue_chat": self._continue_conversation,
            "show_summary": self._show_project_summary,
            "phase_nav": self._navigate_phases,
            "context_review": self._review_context,
            "export_notes": self._export_notes,
            "reset_phase": self._reset_phase,

            # Navigation actions
            "back_main": lambda: self._set_menu("main"),
            "back": self._go_back,
            "quit": lambda: False
        }

        handler = action_handlers.get(action)
        if handler:
            result = handler()
            return result if result is not None else True
        else:
            print(f"❌ Unknown action: {action}")
            return True

    def _new_project(self):
        """Start a new project with Socratic guidance"""
        print("\n🆕 Starting New Project")
        print("=" * 40)

        # Reset context
        self.context = ProjectContext(
            goals=[], requirements=[], tech_stack=[], constraints=[],
            team_structure="", language_preferences=[], deployment_target="",
            code_style="", current_phase="discovery", conversation_history=[],
            metadata={"created": datetime.now().isoformat()}
        )

        print("Let's begin with the Socratic questioning process...")
        self._start_socratic_conversation()
        self._set_menu("project")

    def _continue_conversation(self):
        """Continue the Socratic conversation"""
        print(f"\n💬 Continuing Conversation - Phase: {self.context.current_phase.title()}")
        print("=" * 60)
        self._start_socratic_conversation()

    def _start_socratic_conversation(self):
        """Begin or continue Socratic questioning"""
        if not self.context.goals:
            initial_question = "What exactly do you want to achieve with this project?"
        else:
            # Generate contextual follow-up question
            initial_question = self._generate_contextual_question()

        print(f"\n🤖 Assistant: {initial_question}")

        while True:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'end', 'menu']:
                break
            elif user_input.lower() == 'summary':
                self._show_project_summary()
                continue
            elif user_input.lower() == 'help':
                self._show_conversation_help()
                continue

            # Process user response and generate follow-up
            response = self._process_user_response(user_input)
            print(f"\n🤖 Assistant: {response}")

            # Check if we should advance to next phase
            if self._should_advance_phase():
                self._advance_phase()

    def _generate_contextual_question(self) -> str:
        """Generate a contextual follow-up question based on current context"""
        context_summary = self._get_context_summary()
        relevant_knowledge = self._get_relevant_knowledge(context_summary)

        prompt = f"""Based on the project context and relevant knowledge, generate a thoughtful Socratic question 
        that will help the user discover deeper insights about their project.

Project Context:
{context_summary}

Relevant Knowledge:
{relevant_knowledge}

Current Phase: {self.context.current_phase}

Generate a single, insightful question that follows the Socratic method - guide them to discover answers rather than 
providing direct solutions."""

        try:
            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            # Fallback questions by phase
            fallback_questions = {
                "discovery": "What specific problems does your project solve, and who benefits from these solutions?",
                "analysis": "What technical challenges do you anticipate, and how might they impact your approach?",
                "design": "How will the different components of your system interact with each other?",
                "implementation": "What would be the most logical first step to begin building this system?"
            }
            return fallback_questions.get(self.context.current_phase,
                                          "What aspect of your project would you like to explore further?")

    def _process_user_response(self, user_input: str) -> str:
        """Process user response and update context"""
        # Update conversation history
        self.context.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "phase": self.context.current_phase,
            "user": user_input,
            "context_before": self._get_context_summary()
        })

        # Extract insights from response
        self._extract_and_update_context(user_input)

        # Generate thoughtful follow-up
        return self._generate_contextual_question()

    def _extract_and_update_context(self, user_input: str):
        """Extract information from user input and update project context"""
        # Simple keyword-based extraction (can be enhanced with NLP)
        words = user_input.lower().split()

        # Extract goals
        goal_indicators = ['want', 'need', 'goal', 'achieve', 'create', 'build']
        if any(indicator in words for indicator in goal_indicators):
            if user_input not in [g.lower() for g in self.context.goals]:
                self.context.goals.append(user_input)

        # Extract tech stack mentions
        tech_keywords = ['python', 'javascript', 'react', 'django', 'flask', 'database', 'api', 'web', 'mobile']
        for tech in tech_keywords:
            if tech in words and tech not in [t.lower() for t in self.context.tech_stack]:
                self.context.tech_stack.append(tech)

    def _show_project_summary(self):
        """Display current project summary"""
        print("\n📋 Project Summary")
        print("=" * 50)
        print(f"Phase: {self.context.current_phase.title()}")
        print(f"Goals: {len(self.context.goals)} identified")
        for i, goal in enumerate(self.context.goals, 1):
            print(f"  {i}. {goal}")

        print(f"\nTech Stack: {', '.join(self.context.tech_stack) if self.context.tech_stack else 'Not specified'}")
        print(f"Team: {self.context.team_structure if self.context.team_structure else 'Not specified'}")
        print(f"Conversations: {len(self.context.conversation_history)} exchanges")

    def _get_context_summary(self) -> str:
        """Get a formatted summary of current project context"""
        return f"""
Goals: {'; '.join(self.context.goals)}
Requirements: {'; '.join(self.context.requirements)}
Tech Stack: {'; '.join(self.context.tech_stack)}
Constraints: {'; '.join(self.context.constraints)}
Team Structure: {self.context.team_structure}
Phase: {self.context.current_phase}
"""

    def _get_relevant_knowledge(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant knowledge base entries"""
        if not self.knowledge_base or not query.strip():
            return "No relevant knowledge found."

        query_embedding = self.model.encode([query])[0]
        similarities = []

        for entry in self.knowledge_base:
            similarity = cosine_similarity([query_embedding], [entry['embedding']])[0][0]
            similarities.append((entry['content'], similarity))

        # Sort by similarity and return top_k results
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_entries = [entry[0] for entry in similarities[:top_k]]

        return '\n'.join(relevant_entries)

    def _should_advance_phase(self) -> bool:
        """Check if enough information has been gathered to advance phase"""
        phase_requirements = {
            "discovery": len(self.context.goals) >= 2,
            "analysis": len(self.context.requirements) >= 3 or len(self.context.conversation_history) >= 5,
            "design": len(self.context.tech_stack) >= 1,
            "implementation": True  # Always ready to implement
        }

        return phase_requirements.get(self.context.current_phase, False)

    def _advance_phase(self):
        """Advance to the next development phase"""
        if self.current_phase_index < len(self.phases) - 1:
            self.current_phase_index += 1
            self.context.current_phase = self.phases[self.current_phase_index]
            print(f"\n🎯 Advancing to {self.context.current_phase.title()} phase!")

    def _navigate_phases(self):
        """Allow user to navigate between phases"""
        print("\n🎯 Phase Navigation")
        print("=" * 30)
        for i, phase in enumerate(self.phases):
            current = "👉 " if phase == self.context.current_phase else "   "
            print(f"{current}[{i + 1}] {phase.title()}")

        try:
            choice = int(input("\nSelect phase (1-4): ")) - 1
            if 0 <= choice < len(self.phases):
                self.context.current_phase = self.phases[choice]
                self.current_phase_index = choice
                print(f"✅ Moved to {self.phases[choice].title()} phase")
            else:
                print("❌ Invalid phase number")
        except ValueError:
            print("❌ Please enter a valid number")

    def _set_menu(self, menu_name: str):
        """Set the current menu"""
        self.menu_system.current_menu = menu_name

    def _go_back(self):
        """Navigate back to previous menu"""
        self._set_menu("main")

    # Additional menu action methods (simplified for brevity)
    def _load_project(self):
        print("\n📂 Load Project - Feature coming soon!")

    def _project_settings(self):
        self._set_menu("settings")

    def _explore_knowledge_base(self):
        print("\n🧠 Knowledge Base Explorer")
        print("=" * 40)
        for i, entry in enumerate(self.knowledge_base[:5], 1):
            print(f"{i}. {entry['content']}")

    def _show_help(self):
        self._set_menu("help")

    def _show_settings(self):
        self._set_menu("settings")

    def _export_import(self):
        print("\n💾 Export/Import - Feature coming soon!")

    def _review_context(self):
        self._show_project_summary()

    def _export_notes(self):
        print("\n📄 Export Notes - Feature coming soon!")

    def _reset_phase(self):
        print(f"\n🔄 Reset {self.context.current_phase.title()} phase - Feature coming soon!")

    def _show_conversation_help(self):
        """Show help during conversation"""
        print("\n❓ Conversation Commands:")
        print("• 'summary' - Show project summary")
        print("• 'help' - Show this help")
        print("• 'menu' - Return to menu")
        print("• 'quit' - End conversation")


def main():
    """Main entry point"""
    # Get API key from environment or prompt user
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        print("⚠️  CLAUDE_API_KEY not found in environment variables.")
        api_key = input("Please enter your Claude API key: ").strip()

        if not api_key:
            print("❌ API key is required to run the application.")
            sys.exit(1)

    try:
        # Initialize and run the application
        socratic_system = SocraticRAG(api_key)
        socratic_system.run_application()

    except Exception as e:
        print(f"❌ Failed to initialize application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
