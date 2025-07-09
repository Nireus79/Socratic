import os
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ProjectContext:
    """Stores and manages project-specific context and information"""

    def __init__(self):
        self.goals = []
        self.requirements = []
        self.tech_stack = []
        self.constraints = []
        self.team_structure = "individual"
        self.language_preferences = {}
        self.deployment_target = ""
        self.code_style_preferences = {}
        self.current_phase = "discovery"
        self.conversation_history = []

    def add_goal(self, goal: str):
        if goal not in self.goals:
            self.goals.append(goal)

    def add_requirement(self, requirement: str):
        if requirement not in self.requirements:
            self.requirements.append(requirement)

    def add_tech_stack_item(self, tech: str):
        if tech not in self.tech_stack:
            self.tech_stack.append(tech)

    def to_dict(self) -> dict:
        return {
            'goals': self.goals,
            'requirements': self.requirements,
            'tech_stack': self.tech_stack,
            'constraints': self.constraints,
            'team_structure': self.team_structure,
            'language_preferences': self.language_preferences,
            'deployment_target': self.deployment_target,
            'code_style_preferences': self.code_style_preferences,
            'current_phase': self.current_phase,
            'conversation_history': self.conversation_history
        }

    def from_dict(self, data: dict):
        self.goals = data.get('goals', [])
        self.requirements = data.get('requirements', [])
        self.tech_stack = data.get('tech_stack', [])
        self.constraints = data.get('constraints', [])
        self.team_structure = data.get('team_structure', 'individual')
        self.language_preferences = data.get('language_preferences', {})
        self.deployment_target = data.get('deployment_target', '')
        self.code_style_preferences = data.get('code_style_preferences', {})
        self.current_phase = data.get('current_phase', 'discovery')
        self.conversation_history = data.get('conversation_history', [])


class KnowledgeEntry:
    """Represents a single entry in the knowledge base with embeddings"""

    def __init__(self, content: str, category: str, embedding: np.ndarray = None):
        self.content = content
        self.category = category
        self.embedding = embedding
        self.created_at = datetime.now()


class DatabaseManager:
    """Handles all database operations for users, projects, and knowledge base"""

    def __init__(self, db_path: str = "socratic_rag.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                project_name TEXT NOT NULL,
                context_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, project_name)
            )
        ''')

        # Knowledge base table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Conversation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_user(self, username: str) -> int:
        """Create a new user and return user ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('INSERT INTO users (username) VALUES (?)', (username,))
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def get_user(self, username: str) -> Optional[dict]:
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'id': result[0],
                'username': result[1],
                'created_at': result[2],
                'last_active': result[3]
            }
        return None

    def create_project(self, user_id: int, project_name: str, context: ProjectContext) -> int:
        """Create a new project for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            context_json = json.dumps(context.to_dict())
            cursor.execute('''
                INSERT INTO projects (user_id, project_name, context_data) 
                VALUES (?, ?, ?)
            ''', (user_id, project_name, context_json))
            project_id = cursor.lastrowid
            conn.commit()
            return project_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def get_project(self, user_id: int, project_name: str) -> Optional[dict]:
        """Get project by user ID and project name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM projects 
            WHERE user_id = ? AND project_name = ?
        ''', (user_id, project_name))
        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'id': result[0],
                'user_id': result[1],
                'project_name': result[2],
                'context_data': result[3],
                'created_at': result[4],
                'updated_at': result[5]
            }
        return None

    def update_project_context(self, project_id: int, context: ProjectContext):
        """Update project context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        context_json = json.dumps(context.to_dict())
        cursor.execute('''
            UPDATE projects 
            SET context_data = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (context_json, project_id))
        conn.commit()
        conn.close()

    def get_user_projects(self, user_id: int) -> List[dict]:
        """Get all projects for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM projects WHERE user_id = ?', (user_id,))
        results = cursor.fetchall()
        conn.close()

        projects = []
        for result in results:
            projects.append({
                'id': result[0],
                'user_id': result[1],
                'project_name': result[2],
                'context_data': result[3],
                'created_at': result[4],
                'updated_at': result[5]
            })
        return projects

    def save_conversation_message(self, project_id: int, role: str, content: str):
        """Save a conversation message"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO conversation_history (project_id, role, content) 
            VALUES (?, ?, ?)
        ''', (project_id, role, content))
        conn.commit()
        conn.close()

    def get_conversation_history(self, project_id: int) -> List[dict]:
        """Get conversation history for a project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT role, content, timestamp 
            FROM conversation_history 
            WHERE project_id = ? 
            ORDER BY timestamp
        ''', (project_id,))
        results = cursor.fetchall()
        conn.close()

        history = []
        for result in results:
            history.append({
                'role': result[0],
                'content': result[1],
                'timestamp': result[2]
            })
        return history


class SocraticRAG:
    """Main RAG system with Socratic questioning methodology"""

    def __init__(self, api_key: str, db_path: str = "socratic_rag.db"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.db_manager = DatabaseManager(db_path)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = []
        self.current_user = None
        self.current_project = None
        self.current_project_id = None
        self.context = ProjectContext()

        # Initialize knowledge base
        self.load_knowledge_base()
        if not self.knowledge_base:
            self.initialize_knowledge_base()

    def load_knowledge_base(self):
        """Load existing knowledge base from database"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT content, category, embedding FROM knowledge_base')
        results = cursor.fetchall()
        conn.close()

        for result in results:
            content, category, embedding_blob = result
            if embedding_blob:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                entry = KnowledgeEntry(content, category, embedding)
                self.knowledge_base.append(entry)

    def initialize_knowledge_base(self):
        """Initialize the knowledge base with software development best practices"""
        knowledge_items = [
            ("Software development follows phases: requirements gathering, design, implementation, testing, "
             "deployment, and maintenance.",
             "methodology"),
            ("Agile methodology emphasizes iterative development, collaboration, and responding to change.",
             "methodology"),
            ("Version control systems like Git help track changes and enable collaboration.", "tools"),
            ("Code should be readable, maintainable, and well-documented.", "best_practices"),
            ("Testing is crucial for software quality and includes unit, integration, and system testing.", "testing"),
            ("Security should be considered throughout the development lifecycle.", "security"),
            ("Performance optimization should be based on profiling and measurement.", "performance"),
            ("Database design should consider normalization, indexing, and scalability.", "database"),
            ("API design should be RESTful, well-documented, and versioned.", "api_design"),
            ("User experience (UX) should be intuitive and accessible.", "user_experience"),
        ]

        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()

        for content, category in knowledge_items:
            embedding = self.encoder.encode(content)
            entry = KnowledgeEntry(content, category, embedding)
            self.knowledge_base.append(entry)

            # Save to database
            embedding_blob = embedding.tobytes()
            cursor.execute('''
                INSERT INTO knowledge_base (content, category, embedding) 
                VALUES (?, ?, ?)
            ''', (content, category, embedding_blob))

        conn.commit()
        conn.close()

    def login_or_create_user(self) -> bool:
        """Handle user login or creation"""
        print("🔐 User Authentication")
        print("=" * 50)

        username = input("Enter username: ").strip()
        if not username:
            print("Username cannot be empty!")
            return False

        user = self.db_manager.get_user(username)
        if user:
            print(f"Welcome back, {username}!")
            self.current_user = user
        else:
            create = input(f"User '{username}' not found. Create new user? (y/n): ").lower()
            if create == 'y':
                user_id = self.db_manager.create_user(username)
                if user_id:
                    self.current_user = {
                        'id': user_id,
                        'username': username
                    }
                    print(f"Created new user: {username}")
                else:
                    print("Error creating user!")
                    return False
            else:
                return False

        return True

    def select_or_create_project(self) -> bool:
        """Handle project selection or creation"""
        print(f"\n📁 Project Management for {self.current_user['username']}")
        print("=" * 50)

        # Get existing projects
        projects = self.db_manager.get_user_projects(self.current_user['id'])

        if projects:
            print("Existing projects:")
            for i, project in enumerate(projects, 1):
                print(f"{i}. {project['project_name']} (Updated: {project['updated_at']})")

            print(f"{len(projects) + 1}. Create new project")

            try:
                choice = int(input("\nSelect option: "))
                if 1 <= choice <= len(projects):
                    # Load existing project
                    selected_project = projects[choice - 1]
                    self.current_project = selected_project
                    self.current_project_id = selected_project['id']

                    # Load context
                    context_data = json.loads(selected_project['context_data'])
                    self.context.from_dict(context_data)

                    print(f"Loaded project: {selected_project['project_name']}")
                    return True
                elif choice == len(projects) + 1:
                    return self.create_new_project()
                else:
                    print("Invalid choice!")
                    return False
            except ValueError:
                print("Please enter a valid number!")
                return False
        else:
            print("No existing projects found.")
            return self.create_new_project()

    def create_new_project(self) -> bool:
        """Create a new project"""
        project_name = input("Enter project name: ").strip()
        if not project_name:
            print("Project name cannot be empty!")
            return False

        # Check if project already exists
        existing = self.db_manager.get_project(self.current_user['id'], project_name)
        if existing:
            print("Project with this name already exists!")
            return False

        # Create new project
        project_id = self.db_manager.create_project(
            self.current_user['id'],
            project_name,
            self.context
        )

        if project_id:
            self.current_project_id = project_id
            self.current_project = {
                'id': project_id,
                'project_name': project_name,
                'user_id': self.current_user['id']
            }
            print(f"Created new project: {project_name}")
            return True
        else:
            print("Error creating project!")
            return False

    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[KnowledgeEntry]:
        """Retrieve relevant knowledge entries based on query"""
        if not self.knowledge_base:
            return []

        query_embedding = self.encoder.encode(query)

        similarities = []
        for entry in self.knowledge_base:
            if entry.embedding is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    entry.embedding.reshape(1, -1)
                )[0][0]
                similarities.append((entry, similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in similarities[:top_k]]

    def generate_socratic_response(self, user_input: str) -> str:
        """Generate a Socratic response using Claude"""
        # Retrieve relevant knowledge
        relevant_knowledge = self.retrieve_relevant_knowledge(user_input)
        knowledge_context = "\n".join([entry.content for entry in relevant_knowledge])

        # Build context summary
        context_summary = f"""
        Project Goals: {', '.join(self.context.goals) if self.context.goals else 'Not defined'}
        Requirements: {', '.join(self.context.requirements) if self.context.requirements else 'Not defined'}
        Tech Stack: {', '.join(self.context.tech_stack) if self.context.tech_stack else 'Not defined'}
        Current Phase: {self.context.current_phase}
        """

        # Create the prompt
        prompt = f"""You are a Socratic counselor for software development. Use the Socratic method to guide the developer through thoughtful questioning rather than giving direct answers.

Context about the current project:
{context_summary}

Relevant knowledge base:
{knowledge_context}

User's latest input: {user_input}

Previous conversation context: {self.context.conversation_history[-5:] if self.context.conversation_history else 'None'}

Respond as a Socratic counselor by:
1. Asking thoughtful questions that help the user discover solutions
2. Building on their responses to deepen understanding
3. Guiding them through the current phase: {self.context.current_phase}
4. Helping them think through problems systematically
5. Encouraging critical thinking about their choices

Keep responses focused, insightful, and question-driven. Avoid giving direct answers unless absolutely necessary."""

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return (f"I apologize, but I encountered an error: {str(e)}. Let me ask you this instead: What specific "
                    f"aspect of your project would you like to explore further?")

    def update_context_from_conversation(self, user_input: str, assistant_response: str):
        """Update project context based on conversation"""
        # Simple keyword-based context extraction
        user_lower = user_input.lower()

        # Extract goals
        if any(word in user_lower for word in ['want to', 'goal', 'achieve', 'build', 'create']):
            self.context.add_goal(user_input)

        # Extract requirements
        if any(word in user_lower for word in ['need', 'require', 'must', 'should']):
            self.context.add_requirement(user_input)

        # Extract tech stack
        tech_keywords = ['python', 'javascript', 'react', 'django', 'flask', 'nodejs', 'database', 'sql', 'mongodb',
                         'docker']
        for tech in tech_keywords:
            if tech in user_lower:
                self.context.add_tech_stack_item(tech)

        # Update conversation history
        self.context.conversation_history.append({
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': datetime.now().isoformat()
        })

        # Keep only last 20 exchanges
        if len(self.context.conversation_history) > 20:
            self.context.conversation_history = self.context.conversation_history[-20:]

    def save_session_data(self):
        """Save current session data to database"""
        if self.current_project_id:
            self.db_manager.update_project_context(self.current_project_id, self.context)

    def display_project_summary(self):
        """Display current project summary"""
        print("\n" + "=" * 50)
        print("📋 PROJECT SUMMARY")
        print("=" * 50)
        print(f"Project: {self.current_project['project_name'] if self.current_project else 'None'}")
        print(f"Current Phase: {self.context.current_phase}")
        print(f"Goals: {', '.join(self.context.goals) if self.context.goals else 'None defined'}")
        print(f"Requirements: {', '.join(self.context.requirements) if self.context.requirements else 'None defined'}")
        print(f"Tech Stack: {', '.join(self.context.tech_stack) if self.context.tech_stack else 'None defined'}")
        print("=" * 50)

    def start_chat(self):
        """Start the main chat interaction"""
        print(f"\n🤖 Socratic Counselor - Project: {self.current_project['project_name']}")
        print("=" * 70)
        print("I'm here to guide you through your software development journey using Socratic questioning.")
        print("Type 'summary' to see your project overview, or 'quit' to exit.")
        print("=" * 70)

        # Load conversation history
        history = self.db_manager.get_conversation_history(self.current_project_id)
        if history:
            print("\n📚 Previous conversation:")
            for msg in history[-3:]:  # Show last 3 messages
                role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                print(f"{role_emoji} {msg['content'][:100]}...")
            print()

        # Initial question if new project
        if not self.context.goals:
            initial_question = "What exactly do you want to achieve with this project?"
            print(f"Assistant: {initial_question}")
            self.db_manager.save_conversation_message(self.current_project_id, 'assistant', initial_question)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'end']:
                    print("Thank you for using Socratic Counselor. Your progress has been saved!")
                    break
                elif user_input.lower() == 'summary':
                    self.display_project_summary()
                    continue

                # Save user message
                self.db_manager.save_conversation_message(self.current_project_id, 'user', user_input)

                # Generate response
                response = self.generate_socratic_response(user_input)
                print(f"Assistant: {response}")

                # Save assistant message
                self.db_manager.save_conversation_message(self.current_project_id, 'assistant', response)

                # Update context
                self.update_context_from_conversation(user_input, response)

                # Save session data periodically
                self.save_session_data()

            except KeyboardInterrupt:
                print("\n\nSession interrupted. Saving progress...")
                self.save_session_data()
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                continue


def main():
    """Main function to run the Socratic RAG system"""
    print("🚀 Socratic RAG System - Enhanced Version")
    print("Ουδέν οίδα, ούτε διδάσκω τι, αλλά διαπορώ μόνον.")
    print("=" * 60)

    # Get API key
    api_key = os.getenv('API_KEY_CLAUDE')
    if not api_key:
        api_key = input("Enter your Claude API key: ").strip()
        if not api_key:
            print("API key is required!")
            return

    try:
        # Initialize the system
        rag_system = SocraticRAG(api_key)

        # User authentication
        if not rag_system.login_or_create_user():
            print("Authentication failed!")
            return

        # Project selection/creation
        if not rag_system.select_or_create_project():
            print("Project setup failed!")
            return

        # Start the chat
        rag_system.start_chat()

    except Exception as e:
        print(f"System error: {str(e)}")
        print("Please check your API key and try again.")


if __name__ == "__main__":
    main()
