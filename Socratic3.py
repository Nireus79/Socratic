import sqlite3
import uuid
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

"""
Socratic Collaborative System - Main Application
"""


class ProjectRole(Enum):
    OWNER = "owner"
    COLLABORATOR = "collaborator"
    VIEWER = "viewer"


@dataclass
class ProjectMember:
    user_id: str
    project_id: str
    role: ProjectRole
    added_at: str
    added_by: str


class CollaborativeDatabaseManager:
    """Database manager with collaborative project support"""

    def __init__(self, db_path: str = "socratic.db"):
        self.db_path = db_path
        self.init_all_tables()

    def init_all_tables(self):
        """Initialize all database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    project_name TEXT,
                    project_description TEXT,
                    current_phase TEXT DEFAULT 'planning',
                    context_data TEXT,
                    owner_id TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (owner_id) REFERENCES users (user_id)
                )
            ''')

            # Project members table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    project_id TEXT,
                    role TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    added_by TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (project_id) REFERENCES projects (project_id),
                    FOREIGN KEY (added_by) REFERENCES users (user_id),
                    UNIQUE(user_id, project_id)
                )
            ''')

            # Project activity log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS project_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    user_id TEXT,
                    activity_type TEXT,
                    description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')

            conn.commit()

    def create_user(self, user_id: str, display_name: str, email: str = None) -> bool:
        """Create a new user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO users (user_id, display_name, email)
                    VALUES (?, ?, ?)
                ''', (user_id, display_name, email))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def user_exists(self, user_id: str) -> bool:
        """Check if user exists"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM users WHERE user_id = ?', (user_id,))
            return cursor.fetchone() is not None

    def create_project(self, owner_id: str, project_name: str, description: str = "") -> str:
        """Create a new project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            project_id = str(uuid.uuid4())

            # Create project
            cursor.execute('''
                INSERT INTO projects (project_id, owner_id, project_name, project_description, context_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (project_id, owner_id, project_name, description, json.dumps({})))

            # Add creator as owner
            cursor.execute('''
                INSERT INTO project_members (user_id, project_id, role, added_by)
                VALUES (?, ?, ?, ?)
            ''', (owner_id, project_id, ProjectRole.OWNER.value, owner_id))

            # Log project creation
            cursor.execute('''
                INSERT INTO project_activity (project_id, user_id, activity_type, description)
                VALUES (?, ?, ?, ?)
            ''', (project_id, owner_id, 'project_created', f"Created project '{project_name}'"))

            conn.commit()
            return project_id

    def get_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all projects a user has access to"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.project_id, p.project_name, p.project_description, 
                       p.current_phase, p.created_at, p.updated_at, pm.role
                FROM projects p
                JOIN project_members pm ON p.project_id = pm.project_id
                WHERE pm.user_id = ? AND pm.is_active = 1 AND p.is_active = 1
                ORDER BY p.updated_at DESC
            ''', (user_id,))

            return [
                {
                    "project_id": row[0],
                    "project_name": row[1],
                    "project_description": row[2],
                    "current_phase": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "role": row[6]
                }
                for row in cursor.fetchall()
            ]

    def get_project_details(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project details"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT project_id, project_name, project_description, 
                       current_phase, owner_id, created_at, updated_at
                FROM projects
                WHERE project_id = ? AND is_active = 1
            ''', (project_id,))

            result = cursor.fetchone()
            if result:
                return {
                    "project_id": result[0],
                    "project_name": result[1],
                    "project_description": result[2],
                    "current_phase": result[3],
                    "owner_id": result[4],
                    "created_at": result[5],
                    "updated_at": result[6]
                }
            return None

    def add_project_member(self, project_id: str, user_id: str, role: ProjectRole, added_by: str) -> bool:
        """Add a user to a project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check permissions
            cursor.execute('''
                SELECT role FROM project_members 
                WHERE project_id = ? AND user_id = ? AND is_active = 1
            ''', (project_id, added_by))

            adder_role = cursor.fetchone()
            if not adder_role or adder_role[0] not in ['owner', 'collaborator']:
                return False

            try:
                cursor.execute('''
                    INSERT INTO project_members (user_id, project_id, role, added_by)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, project_id, role.value, added_by))

                cursor.execute('''
                    INSERT INTO project_activity (project_id, user_id, activity_type, description)
                    VALUES (?, ?, ?, ?)
                ''', (project_id, added_by, 'member_added', f"Added {user_id} as {role.value}"))

                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def can_user_access_project(self, user_id: str, project_id: str) -> Optional[str]:
        """Check if user can access project and return their role"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT role FROM project_members 
                WHERE user_id = ? AND project_id = ? AND is_active = 1
            ''', (user_id, project_id))

            result = cursor.fetchone()
            return result[0] if result else None


class SocraticSystem:
    """Main Socratic System"""

    def __init__(self):
        self.db = CollaborativeDatabaseManager()
        self.current_user = None

    def start(self):
        """Start the application"""
        print("🎓 Socratic Collaborative System")
        print("Ουδέν οίδα, ούτε διδάσκω τι, αλλά διαπορώ μόνον.")
        print("=" * 40)

        # Get or create current user
        self.setup_user()

        # Main menu loop
        self.main_menu()

    def setup_user(self):
        """Setup current user"""
        print("\n👤 User Setup")
        print("-" * 20)

        while True:
            user_id = input("Enter your user ID (or 'new' to create): ").strip()

            if user_id.lower() == 'new':
                self.create_new_user()
                break
            elif user_id and self.db.user_exists(user_id):
                self.current_user = user_id
                print(f"✅ Logged in as {user_id}")
                break
            else:
                print("❌ User not found. Try again or type 'new' to create a user.")

    def create_new_user(self):
        """Create a new user"""
        while True:
            user_id = input("Enter new user ID: ").strip()
            if not user_id:
                print("❌ User ID cannot be empty")
                continue

            if self.db.user_exists(user_id):
                print("❌ User ID already exists")
                continue

            display_name = input("Enter display name: ").strip()
            if not display_name:
                display_name = user_id

            email = input("Enter email (optional): ").strip() or None

            if self.db.create_user(user_id, display_name, email):
                self.current_user = user_id
                print(f"✅ User {user_id} created successfully")
                break
            else:
                print("❌ Failed to create user")

    def main_menu(self):
        """Main application menu"""
        while True:
            print(f"\n🎯 Main Menu - {self.current_user}")
            print("1. Create Project")
            print("2. View My Projects")
            print("3. Open Project")
            print("4. Project Management")
            print("5. Switch User")
            print("6. Exit")

            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                self.create_project()
            elif choice == "2":
                self.view_my_projects()
            elif choice == "3":
                self.open_project()
            elif choice == "4":
                self.project_management_menu()
            elif choice == "5":
                self.setup_user()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")

    def create_project(self):
        """Create a new project"""
        print("\n📝 Create New Project")
        print("-" * 30)

        project_name = input("Enter project name: ").strip()
        if not project_name:
            print("❌ Project name cannot be empty")
            return

        description = input("Enter project description (optional): ").strip()

        try:
            project_id = self.db.create_project(self.current_user, project_name, description)
            print(f"✅ Project '{project_name}' created successfully!")
            print(f"Project ID: {project_id}")
        except Exception as e:
            print(f"❌ Failed to create project: {e}")

    def view_my_projects(self):
        """View user's projects"""
        print("\n📂 My Projects")
        print("-" * 20)

        projects = self.db.get_user_projects(self.current_user)

        if not projects:
            print("No projects found. Create a project first!")
            return

        for i, project in enumerate(projects, 1):
            print(f"\n{i}. {project['project_name']}")
            print(f"   Description: {project['project_description'] or 'No description'}")
            print(f"   Role: {project['role'].title()}")
            print(f"   Phase: {project['current_phase']}")
            print(f"   Created: {project['created_at']}")

    def open_project(self):
        """Open a specific project"""
        print("\n🔍 Open Project")
        print("-" * 20)

        projects = self.db.get_user_projects(self.current_user)

        if not projects:
            print("No projects found.")
            return

        # List projects
        for i, project in enumerate(projects, 1):
            print(f"{i}. {project['project_name']} ({project['role']})")

        try:
            choice = int(input("\nEnter project number: ")) - 1
            if 0 <= choice < len(projects):
                project = projects[choice]
                self.project_workspace(project)
            else:
                print("❌ Invalid project number")
        except ValueError:
            print("❌ Please enter a valid number")

    def project_workspace(self, project):
        """Project workspace"""
        print(f"\n🏢 Project: {project['project_name']}")
        print("=" * 50)

        details = self.db.get_project_details(project['project_id'])
        if details:
            print(f"Description: {details['project_description'] or 'No description'}")
            print(f"Current Phase: {details['current_phase']}")
            print(f"Your Role: {project['role']}")

        print("\nProject workspace features would go here...")
        print("(This is where you'd implement Socratic questioning, collaboration tools, etc.)")

        input("\nPress Enter to return to main menu...")

    def project_management_menu(self):
        """Project management submenu"""
        while True:
            print(f"\n⚙️ Project Management - {self.current_user}")
            print("1. Add Member to Project")
            print("2. View Project Members")
            print("3. Back to Main Menu")

            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                self.add_project_member()
            elif choice == "2":
                self.view_project_members()
            elif choice == "3":
                break
            else:
                print("❌ Invalid choice. Please try again.")

    def add_project_member(self):
        """Add a member to a project"""
        print("\n👥 Add Project Member")
        print("-" * 25)

        # Show user's projects where they can add members
        projects = self.db.get_user_projects(self.current_user)
        owner_projects = [p for p in projects if p['role'] in ['owner', 'collaborator']]

        if not owner_projects:
            print("You don't have permission to add members to any projects.")
            return

        print("Projects you can manage:")
        for i, project in enumerate(owner_projects, 1):
            print(f"{i}. {project['project_name']} ({project['role']})")

        try:
            choice = int(input("\nSelect project: ")) - 1
            if 0 <= choice < len(owner_projects):
                project = owner_projects[choice]

                user_id = input("Enter user ID to add: ").strip()
                if not self.db.user_exists(user_id):
                    print("❌ User not found")
                    return

                role = input("Enter role (collaborator/viewer): ").strip().lower()
                if role not in ['collaborator', 'viewer']:
                    role = 'collaborator'

                role_enum = ProjectRole(role)
                success = self.db.add_project_member(
                    project['project_id'], user_id, role_enum, self.current_user
                )

                if success:
                    print(f"✅ User {user_id} added as {role}")
                else:
                    print("❌ Failed to add user (already member or permission denied)")
            else:
                print("❌ Invalid project number")
        except ValueError:
            print("❌ Please enter a valid number")

    def view_project_members(self):
        """View members of a project"""
        print("\n👥 View Project Members")
        print("-" * 25)

        projects = self.db.get_user_projects(self.current_user)

        if not projects:
            print("No projects found.")
            return

        for i, project in enumerate(projects, 1):
            print(f"{i}. {project['project_name']}")

        try:
            choice = int(input("\nSelect project: ")) - 1
            if 0 <= choice < len(projects):
                project_id = projects[choice]['project_id']

                # Get members (this would need to be implemented in the database class)
                print(f"\nMembers of '{projects[choice]['project_name']}':")
                print("(Member listing functionality would be implemented here)")
            else:
                print("❌ Invalid project number")
        except ValueError:
            print("❌ Please enter a valid number")


def main():
    """Main entry point"""
    system = SocraticSystem()
    system.start()


if __name__ == "__main__":
    main()