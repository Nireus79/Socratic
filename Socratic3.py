import uuid
import sqlite3
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


# Extension to support multiple users working on the same project
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
    """Extended database manager with collaborative project support"""

    def __init__(self, db_path: str = "collaborative_socratic.db"):
        self.db_path = db_path
        self.init_collaborative_tables()

    def init_collaborative_tables(self):
        """Initialize tables for collaborative features"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

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

            # Modified projects table to remove single user_id constraint
            cursor.execute('''
                ALTER TABLE projects ADD COLUMN owner_id TEXT
            ''')

            # Migrate existing projects to have owner_id
            cursor.execute('''
                UPDATE projects SET owner_id = user_id WHERE owner_id IS NULL
            ''')

            conn.commit()

    def add_project_member(self, project_id: str, user_id: str, role: ProjectRole,
                           added_by: str) -> bool:
        """Add a user to a project with specified role"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if the person adding has permission (owner or collaborator)
            cursor.execute('''
                SELECT role FROM project_members 
                WHERE project_id = ? AND user_id = ? AND is_active = 1
            ''', (project_id, added_by))

            adder_role = cursor.fetchone()
            if not adder_role or adder_role[0] not in ['owner', 'collaborator']:
                return False

            # Add the member
            try:
                cursor.execute('''
                    INSERT INTO project_members (user_id, project_id, role, added_by)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, project_id, role.value, added_by))

                # Log the activity
                cursor.execute('''
                    INSERT INTO project_activity (project_id, user_id, activity_type, description)
                    VALUES (?, ?, ?, ?)
                ''', (project_id, added_by, 'member_added',
                      f"Added {user_id} as {role.value}"))

                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # User already in project

    def remove_project_member(self, project_id: str, user_id: str, removed_by: str) -> bool:
        """Remove a user from a project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check permissions
            cursor.execute('''
                SELECT role FROM project_members 
                WHERE project_id = ? AND user_id = ? AND is_active = 1
            ''', (project_id, removed_by))

            remover_role = cursor.fetchone()
            if not remover_role or remover_role[0] != 'owner':
                return False

            # Don't allow removing the owner
            cursor.execute('''
                SELECT role FROM project_members 
                WHERE project_id = ? AND user_id = ? AND is_active = 1
            ''', (project_id, user_id))

            target_role = cursor.fetchone()
            if target_role and target_role[0] == 'owner':
                return False

            # Remove the member
            cursor.execute('''
                UPDATE project_members SET is_active = 0 
                WHERE project_id = ? AND user_id = ?
            ''', (project_id, user_id))

            # Log the activity
            cursor.execute('''
                INSERT INTO project_activity (project_id, user_id, activity_type, description)
                VALUES (?, ?, ?, ?)
            ''', (project_id, removed_by, 'member_removed', f"Removed {user_id}"))

            conn.commit()
            return True

    def get_project_members(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all members of a project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pm.user_id, pm.role, pm.added_at, u.display_name
                FROM project_members pm
                JOIN users u ON pm.user_id = u.user_id
                WHERE pm.project_id = ? AND pm.is_active = 1
                ORDER BY pm.added_at
            ''', (project_id,))

            return [
                {
                    "user_id": row[0],
                    "role": row[1],
                    "added_at": row[2],
                    "display_name": row[3]
                }
                for row in cursor.fetchall()
            ]

    def get_user_projects_collaborative(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all projects a user has access to (owned or member)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.project_id, p.project_name, p.current_phase, 
                       p.created_at, p.updated_at, pm.role
                FROM projects p
                JOIN project_members pm ON p.project_id = pm.project_id
                WHERE pm.user_id = ? AND pm.is_active = 1 AND p.is_active = 1
                ORDER BY p.updated_at DESC
            ''', (user_id,))

            return [
                {
                    "project_id": row[0],
                    "project_name": row[1],
                    "current_phase": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "role": row[5]
                }
                for row in cursor.fetchall()
            ]

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

    def can_user_edit_project(self, user_id: str, project_id: str) -> bool:
        """Check if user can edit project (owner or collaborator)"""
        role = self.can_user_access_project(user_id, project_id)
        return role in ['owner', 'collaborator']

    def get_project_activity(self, project_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent activity for a project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pa.activity_type, pa.description, pa.timestamp, u.display_name
                FROM project_activity pa
                JOIN users u ON pa.user_id = u.user_id
                WHERE pa.project_id = ?
                ORDER BY pa.timestamp DESC
                LIMIT ?
            ''', (project_id, limit))

            return [
                {
                    "activity_type": row[0],
                    "description": row[1],
                    "timestamp": row[2],
                    "user_name": row[3]
                }
                for row in cursor.fetchall()
            ]

    def create_collaborative_project(self, owner_id: str, project_name: str) -> str:
        """Create a project and automatically add the creator as owner"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            project_id = str(uuid.uuid4())

            # Create project
            cursor.execute('''
                INSERT INTO projects (project_id, owner_id, project_name, context_data)
                VALUES (?, ?, ?, ?)
            ''', (project_id, owner_id, project_name, json.dumps({})))

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


# Usage example for the collaborative system
class CollaborativeCommands:
    """Command handlers for collaborative features"""

    def __init__(self, db_manager):
        self.db = db_manager

    def invite_user_to_project(self, inviter_id: str, project_id: str,
                               invitee_id: str, role: str = "collaborator"):
        """Invite a user to join a project"""
        role_enum = ProjectRole(role)
        success = self.db.add_project_member(project_id, invitee_id, role_enum, inviter_id)

        if success:
            return f"✅ User {invitee_id} added to project as {role}"
        else:
            return f"❌ Failed to add user to project (permission denied or already member)"

    def show_project_members(self, project_id: str):
        """Display all members of a project"""
        members = self.db.get_project_members(project_id)

        if not members:
            return "No members found for this project"

        result = f"👥 Project Members ({len(members)}):\n"
        for member in members:
            role_emoji = {"owner": "👑", "collaborator": "🤝", "viewer": "👀"}
            emoji = role_emoji.get(member["role"], "❓")
            result += f"{emoji} {member['display_name']} ({member['role']})\n"

        return result

    def show_project_activity(self, project_id: str):
        """Display recent project activity"""
        activity = self.db.get_project_activity(project_id)

        if not activity:
            return "No recent activity"

        result = "📊 Recent Activity:\n"
        for item in activity:
            result += f"• {item['description']} - {item['user_name']} ({item['timestamp']})\n"

        return result


# Example usage in the main system:
"""
# Create collaborative database
collab_db = CollaborativeDatabaseManager()
collab_commands = CollaborativeCommands(collab_db)

# User Alice creates a project
project_id = collab_db.create_collaborative_project("alice", "Shared Web App")

# Alice invites Bob as collaborator
collab_commands.invite_user_to_project("alice", project_id, "bob", "collaborator")

# Alice invites Charlie as viewer
collab_commands.invite_user_to_project("alice", project_id, "charlie", "viewer")

# Check if Bob can edit the project
can_edit = collab_db.can_user_edit_project("bob", project_id)  # True

# Check if Charlie can edit the project
can_edit = collab_db.can_user_edit_project("charlie", project_id)  # False

# Show project members
print(collab_commands.show_project_members(project_id))

# Show project activity
print(collab_commands.show_project_activity(project_id))
"""
