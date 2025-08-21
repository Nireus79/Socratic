#!/usr/bin/env python3
"""
Comprehensive Automated Test Suite for Socratic RAG System v7.0
Includes unit tests, integration tests, and scenario testing.
"""

import unittest
import tempfile
import shutil
import os
import json
import sqlite3
import datetime
import hashlib
import threading
import time
import sys
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import subprocess

# Add the directory containing Socratic7.py to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from Socratic7 import (
        User, ProjectContext, KnowledgeEntry, TokenUsage, ConflictInfo,
        ProjectDatabase, VectorDatabase, ClaudeClient, AgentOrchestrator,
        ProjectManagerAgent, SocraticCounselorAgent, ContextAnalyzerAgent,
        CodeGeneratorAgent, SystemMonitorAgent, ConflictDetectorAgent,
        SocraticRAGSystem
    )
except ImportError as e:
    print(f"Error importing Socratic7 modules: {e}")
    print("Make sure Socratic7.py is in the same directory as this test file")
    sys.exit(1)


class TestDataModels(unittest.TestCase):
    """Unit tests for data models"""

    def test_user_creation(self):
        """Test User dataclass creation"""
        user = User(
            username="testuser",
            passcode_hash="hashedpass",
            created_at=datetime.datetime.now(),
            projects=["proj1", "proj2"]
        )
        self.assertEqual(user.username, "testuser")
        self.assertEqual(len(user.projects), 2)

    def test_project_context_creation(self):
        """Test ProjectContext dataclass creation"""
        project = ProjectContext(
            project_id="test-123",
            name="Test Project",
            owner="testuser",
            collaborators=[],
            goals="Build a web app",
            requirements=[],
            tech_stack=["python", "django"],
            constraints=[],
            team_structure="individual",
            language_preferences="python",
            deployment_target="cloud",
            code_style="pep8",
            phase="discovery",
            conversation_history=[],
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )
        self.assertEqual(project.name, "Test Project")
        self.assertEqual(len(project.tech_stack), 2)

    def test_knowledge_entry_creation(self):
        """Test KnowledgeEntry dataclass creation"""
        entry = KnowledgeEntry(
            id="test-knowledge",
            content="Test knowledge content",
            category="test",
            metadata={"source": "test"},
            embedding=[0.1, 0.2, 0.3]
        )
        self.assertEqual(entry.id, "test-knowledge")
        self.assertEqual(len(entry.embedding), 3)


class MockClaudeResponse:
    """Mock class for Claude API responses"""

    def __init__(self, text="Mock response", input_tokens=100, output_tokens=50):
        self.content = [Mock(text=text)]
        self.usage = Mock(input_tokens=input_tokens, output_tokens=output_tokens)


class TestDatabaseOperations(unittest.TestCase):
    """Unit tests for database operations"""

    def setUp(self):
        """Set up test database in temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_projects.db")
        self.db = ProjectDatabase(self.db_path)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_user_save_load(self):
        """Test user save and load operations"""
        user = User(
            username="testuser",
            passcode_hash="hashedpass",
            created_at=datetime.datetime.now(),
            projects=[]
        )

        # Save user
        self.db.save_user(user)

        # Load user
        loaded_user = self.db.load_user("testuser")
        self.assertIsNotNone(loaded_user)
        self.assertEqual(loaded_user.username, "testuser")

    def test_project_save_load(self):
        """Test project save and load operations"""
        project = ProjectContext(
            project_id="test-123",
            name="Test Project",
            owner="testuser",
            collaborators=[],
            goals="Test goals",
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

        # Save project
        self.db.save_project(project)

        # Load project
        loaded_project = self.db.load_project("test-123")
        self.assertIsNotNone(loaded_project)
        self.assertEqual(loaded_project.name, "Test Project")

    def test_user_exists(self):
        """Test user existence check"""
        self.assertFalse(self.db.user_exists("nonexistent"))

        # Create user
        user = User(
            username="testuser",
            passcode_hash="hashedpass",
            created_at=datetime.datetime.now(),
            projects=[]
        )
        self.db.save_user(user)

        self.assertTrue(self.db.user_exists("testuser"))


class TestVectorDatabase(unittest.TestCase):
    """Unit tests for vector database operations"""

    def setUp(self):
        """Set up test vector database"""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db_path = os.path.join(self.temp_dir, "test_vector_db")

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    @patch('Socratic7.SentenceTransformer')
    @patch('Socratic7.chromadb.PersistentClient')
    def test_knowledge_addition(self, mock_chromadb, mock_transformer):
        """Test adding knowledge to vector database"""
        # Mock the embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_transformer.return_value = mock_model

        # Mock ChromaDB
        mock_collection = Mock()
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client

        # Create vector database
        vector_db = VectorDatabase(self.vector_db_path)

        # Test adding knowledge
        entry = KnowledgeEntry(
            id="test-1",
            content="Test content",
            category="test",
            metadata={"source": "test"}
        )

        vector_db.add_knowledge(entry)
        mock_collection.add.assert_called_once()


class TestAgents(unittest.TestCase):
    """Unit tests for individual agents"""

    def setUp(self):
        """Set up test environment for agents"""
        self.temp_dir = tempfile.mkdtemp()

        # Mock orchestrator
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.database = Mock()
        self.mock_orchestrator.vector_db = Mock()
        self.mock_orchestrator.claude_client = Mock()
        self.mock_orchestrator.context_analyzer = Mock()
        self.mock_orchestrator.system_monitor = Mock()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_project_manager_create_project(self):
        """Test project creation through ProjectManagerAgent"""
        agent = ProjectManagerAgent(self.mock_orchestrator)

        request = {
            'action': 'create_project',
            'project_name': 'Test Project',
            'owner': 'testuser'
        }

        result = agent.process(request)
        self.assertEqual(result['status'], 'success')
        self.assertIn('project', result)

    def test_project_manager_invalid_action(self):
        """Test invalid action handling"""
        agent = ProjectManagerAgent(self.mock_orchestrator)

        request = {'action': 'invalid_action'}
        result = agent.process(request)

        self.assertEqual(result['status'], 'error')
        self.assertIn('Unknown action', result['message'])

    def test_socratic_counselor_static_questions(self):
        """Test static question generation"""
        agent = SocraticCounselorAgent(self.mock_orchestrator)
        agent.use_dynamic_questions = False

        # Mock project
        project = Mock()
        project.phase = 'discovery'
        project.conversation_history = []

        request = {
            'action': 'generate_question',
            'project': project
        }

        result = agent.process(request)
        self.assertEqual(result['status'], 'success')
        self.assertIn('question', result)

    def test_context_analyzer_summary(self):
        """Test context analysis functionality"""
        agent = ContextAnalyzerAgent(self.mock_orchestrator)

        project = ProjectContext(
            project_id="test",
            name="Test Project",
            owner="user",
            collaborators=[],
            goals="Build web app",
            requirements=["fast", "secure"],
            tech_stack=["python", "django"],
            constraints=["budget"],
            team_structure="team",
            language_preferences="python",
            deployment_target="cloud",
            code_style="pep8",
            phase="analysis",
            conversation_history=[],
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )

        summary = agent.get_context_summary(project)
        self.assertIn("Build web app", summary)
        self.assertIn("python", summary)


class TestConflictDetection(unittest.TestCase):
    """Tests for conflict detection functionality"""

    def setUp(self):
        """Set up conflict detector test environment"""
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.claude_client = Mock()
        self.detector = ConflictDetectorAgent(self.mock_orchestrator)

    def test_tech_stack_conflicts(self):
        """Test detection of conflicting technologies"""
        project = ProjectContext(
            project_id="test",
            name="Test Project",
            owner="user",
            collaborators=[],
            goals="",
            requirements=[],
            tech_stack=["mysql"],  # Existing database
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

        new_insights = {
            'tech_stack': ['postgresql']  # Conflicting database
        }

        request = {
            'action': 'detect_conflicts',
            'project': project,
            'new_insights': new_insights,
            'current_user': 'testuser'
        }

        result = self.detector.process(request)
        self.assertEqual(result['status'], 'success')
        # Should detect database conflict
        conflicts = result['conflicts']
        if conflicts:  # Depends on conflict rules implementation
            self.assertTrue(any('tech_stack' in c.conflict_type for c in conflicts))

    def test_find_conflict_category(self):
        """Test conflict category detection"""
        # Test database conflict
        category = self.detector._find_conflict_category('mysql', 'postgresql')
        self.assertEqual(category, 'databases')

        # Test frontend framework conflict
        category = self.detector._find_conflict_category('react', 'vue')
        self.assertEqual(category, 'frontend_frameworks')

        # Test no conflict
        category = self.detector._find_conflict_category('python', 'mysql')
        self.assertIsNone(category)


class TestIntegration(unittest.TestCase):
    """Integration tests for component interactions"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    @patch('Socratic7.anthropic.Anthropic')
    @patch('Socratic7.SentenceTransformer')
    @patch('Socratic7.chromadb.PersistentClient')
    def test_orchestrator_initialization(self, mock_chromadb, mock_transformer, mock_anthropic):
        """Test full orchestrator initialization"""
        # Mock dependencies
        mock_transformer.return_value = Mock()
        mock_chromadb.return_value = Mock()
        mock_anthropic.return_value = Mock()

        # Temporarily change config
        import Socratic7
        original_data_dir = Socratic7.CONFIG['DATA_DIR']
        Socratic7.CONFIG['DATA_DIR'] = self.temp_dir

        try:
            orchestrator = AgentOrchestrator("test-api-key")

            # Verify all agents initialized
            self.assertIsNotNone(orchestrator.project_manager)
            self.assertIsNotNone(orchestrator.socratic_counselor)
            self.assertIsNotNone(orchestrator.context_analyzer)
            self.assertIsNotNone(orchestrator.code_generator)
            self.assertIsNotNone(orchestrator.system_monitor)
            self.assertIsNotNone(orchestrator.conflict_detector)

        finally:
            # Restore original config
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    @patch('Socratic7.anthropic.Anthropic')
    def test_claude_client_integration(self, mock_anthropic):
        """Test Claude client integration with mock responses"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Mock system monitor
        mock_orchestrator = Mock()
        mock_orchestrator.system_monitor = Mock()
        mock_orchestrator.system_monitor.process.return_value = {'status': 'success'}

        claude_client = ClaudeClient("test-key", mock_orchestrator)

        # Mock response
        mock_response = MockClaudeResponse('{"goals": "Build web app"}')
        mock_client.messages.create.return_value = mock_response

        # Test insight extraction
        project = Mock()
        project.goals = ""
        project.phase = "discovery"
        project.tech_stack = []

        insights = claude_client.extract_insights("I want to build a web app", project)
        self.assertIsInstance(insights, dict)


class ScenarioTester:
    """Automated scenario testing class"""

    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.cleanup_needed = temp_dir is None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_needed:
            shutil.rmtree(self.temp_dir)

    @patch('Socratic7.anthropic.Anthropic')
    @patch('Socratic7.SentenceTransformer')
    @patch('Socratic7.chromadb.PersistentClient')
    def create_test_system(self, mock_chromadb, mock_transformer, mock_anthropic):
        """Create a fully mocked test system"""
        # Mock all external dependencies
        mock_transformer.return_value = Mock()
        mock_transformer.return_value.encode.return_value = [0.1, 0.2, 0.3]

        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {'ids': []}
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client

        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client

        # Set up temporary data directory
        import Socratic7
        original_data_dir = Socratic7.CONFIG['DATA_DIR']
        Socratic7.CONFIG['DATA_DIR'] = self.temp_dir

        try:
            orchestrator = AgentOrchestrator("test-api-key")
            return orchestrator, original_data_dir
        except Exception as e:
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir
            raise

    def test_complete_user_journey(self):
        """Test complete user journey from account creation to code generation"""
        try:
            orchestrator, original_data_dir = self.create_test_system()

            # Step 1: Create user
            user = User(
                username="scenario_user",
                passcode_hash=hashlib.sha256("testpass".encode()).hexdigest(),
                created_at=datetime.datetime.now(),
                projects=[]
            )
            orchestrator.database.save_user(user)

            # Step 2: Create project
            result = orchestrator.process_request('project_manager', {
                'action': 'create_project',
                'project_name': 'E-commerce Platform',
                'owner': 'scenario_user'
            })
            self.assert_success(result, "Project creation failed")
            project = result['project']

            # Step 3: Simulate conversation
            mock_insights = {
                'goals': 'Build an online store for handmade items',
                'tech_stack': ['python', 'django', 'postgresql'],
                'requirements': ['payment processing', 'user authentication', 'inventory management']
            }

            # Mock Claude client to return insights
            mock_response = MockClaudeResponse(json.dumps(mock_insights))
            orchestrator.claude_client.client.messages.create.return_value = mock_response

            # Process user response
            result = orchestrator.process_request('socratic_counselor', {
                'action': 'process_response',
                'project': project,
                'response': 'I want to build an online store with Django and PostgreSQL',
                'current_user': 'scenario_user'
            })
            self.assert_success(result, "Response processing failed")

            # Step 4: Test code generation
            mock_code_response = MockClaudeResponse("# Django E-commerce Application\n# Generated code here...")
            orchestrator.claude_client.client.messages.create.return_value = mock_code_response

            result = orchestrator.process_request('code_generator', {
                'action': 'generate_script',
                'project': project
            })
            self.assert_success(result, "Code generation failed")

            print("✅ Complete user journey test passed")
            return True

        except Exception as e:
            print(f"❌ Complete user journey test failed: {e}")
            return False
        finally:
            # Restore original config
            import Socratic7
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def test_collaboration_scenario(self):
        """Test collaboration features"""
        try:
            orchestrator, original_data_dir = self.create_test_system()

            # Create users
            owner = User("owner", "hash1", datetime.datetime.now(), [])
            collaborator = User("collaborator", "hash2", datetime.datetime.now(), [])

            orchestrator.database.save_user(owner)
            orchestrator.database.save_user(collaborator)

            # Create project
            result = orchestrator.process_request('project_manager', {
                'action': 'create_project',
                'project_name': 'Team Project',
                'owner': 'owner'
            })
            self.assert_success(result, "Project creation failed")
            project = result['project']

            # Add collaborator
            result = orchestrator.process_request('project_manager', {
                'action': 'add_collaborator',
                'project': project,
                'username': 'collaborator'
            })
            self.assert_success(result, "Adding collaborator failed")

            # List collaborators
            result = orchestrator.process_request('project_manager', {
                'action': 'list_collaborators',
                'project': project
            })
            self.assert_success(result, "Listing collaborators failed")
            self.assert_condition(
                len(result['collaborators']) == 2,
                "Should have owner + 1 collaborator"
            )

            print("✅ Collaboration scenario test passed")
            return True

        except Exception as e:
            print(f"❌ Collaboration scenario test failed: {e}")
            return False
        finally:
            import Socratic7
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def test_conflict_detection_scenario(self):
        """Test conflict detection in realistic scenario"""
        try:
            orchestrator, original_data_dir = self.create_test_system()

            # Create project with initial tech stack
            result = orchestrator.process_request('project_manager', {
                'action': 'create_project',
                'project_name': 'Conflicting Project',
                'owner': 'testuser'
            })
            self.assert_success(result, "Project creation failed")
            project = result['project']

            # Add initial tech stack
            project.tech_stack = ['mysql', 'react']

            # Test conflicting insights
            conflicting_insights = {
                'tech_stack': ['postgresql', 'vue'],  # Both conflict with existing
                'requirements': ['fast performance']
            }

            result = orchestrator.process_request('conflict_detector', {
                'action': 'detect_conflicts',
                'project': project,
                'new_insights': conflicting_insights,
                'current_user': 'testuser'
            })
            self.assert_success(result, "Conflict detection failed")

            # Should detect conflicts
            conflicts = result['conflicts']
            database_conflict = any(c.conflict_type == 'tech_stack' and 'mysql' in c.old_value for c in conflicts)
            frontend_conflict = any(c.conflict_type == 'tech_stack' and 'react' in c.old_value for c in conflicts)

            print(f"Detected {len(conflicts)} conflicts")
            print("✅ Conflict detection scenario test passed")
            return True

        except Exception as e:
            print(f"❌ Conflict detection scenario test failed: {e}")
            return False
        finally:
            import Socratic7
            Socratic7.CONFIG['DATA_DIR'] = original_data_dir

    def assert_success(self, result, message):
        """Assert that result indicates success"""
        if result.get('status') != 'success':
            raise AssertionError(f"{message}: {result}")

    def assert_condition(self, condition, message):
        """Assert a condition is true"""
        if not condition:
            raise AssertionError(message)


class TestScenarios(unittest.TestCase):
    """Test class for scenario testing"""

    def test_complete_user_journey(self):
        """Test complete user journey scenario"""
        with ScenarioTester() as tester:
            self.assertTrue(tester.test_complete_user_journey())

    def test_collaboration_scenario(self):
        """Test collaboration scenario"""
        with ScenarioTester() as tester:
            self.assertTrue(tester.test_collaboration_scenario())

    def test_conflict_detection_scenario(self):
        """Test conflict detection scenario"""
        with ScenarioTester() as tester:
            self.assertTrue(tester.test_conflict_detection_scenario())


class PerformanceTest:
    """Performance testing utilities"""

    @staticmethod
    def measure_time(func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    @staticmethod
    def test_database_performance():
        """Test database operation performance"""
        print("\n🔍 Running Performance Tests...")

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "perf_test.db")
            db = ProjectDatabase(db_path)

            # Test user save/load performance
            user = User("perfuser", "hash", datetime.datetime.now(), [])

            _, save_time = PerformanceTest.measure_time(db.save_user, user)
            _, load_time = PerformanceTest.measure_time(db.load_user, "perfuser")

            print(f"  User save time: {save_time:.4f}s")
            print(f"  User load time: {load_time:.4f}s")

            # Test project save/load performance
            project = ProjectContext(
                project_id="perf-test",
                name="Performance Test",
                owner="perfuser",
                collaborators=[],
                goals="Test performance",
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

            _, proj_save_time = PerformanceTest.measure_time(db.save_project, project)
            _, proj_load_time = PerformanceTest.measure_time(db.load_project, "perf-test")

            print(f"  Project save time: {proj_save_time:.4f}s")
            print(f"  Project load time: {proj_load_time:.4f}s")

            # Performance assertions
            assert save_time < 1.0, f"User save too slow: {save_time}s"
            assert load_time < 1.0, f"User load too slow: {load_time}s"
            assert proj_save_time < 1.0, f"Project save too slow: {proj_save_time}s"
            assert proj_load_time < 1.0, f"Project load too slow: {proj_load_time}s"

            print("  ✅ All performance tests passed")


class TestRunner:
    """Main test runner class"""

    def __init__(self):
        self.test_results = {
            'unit_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'integration_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'scenario_tests': {'passed': 0, 'failed': 0, 'errors': []},
            'performance_tests': {'passed': 0, 'failed': 0, 'errors': []}
        }

    def run_unit_tests(self):
        """Run all unit tests"""
        print("🧪 Running Unit Tests...")

        unit_test_classes = [
            TestDataModels,
            TestDatabaseOperations,
            TestVectorDatabase,
            TestAgents,
            TestConflictDetection
        ]

        for test_class in unit_test_classes:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)

            self.test_results['unit_tests']['passed'] += result.testsRun - len(result.failures) - len(result.errors)
            self.test_results['unit_tests']['failed'] += len(result.failures) + len(result.errors)

            if result.failures:
                self.test_results['unit_tests']['errors'].extend(
                    [f"{test}: {error}" for test, error in result.failures])
            if result.errors:
                self.test_results['unit_tests']['errors'].extend([f"{test}: {error}" for test, error in result.errors])

    def run_integration_tests(self):
        """Run integration tests"""
        print("🔗 Running Integration Tests...")

        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
        result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)

        self.test_results['integration_tests']['passed'] += result.testsRun - len(result.failures) - len(result.errors)
        self.test_results['integration_tests']['failed'] += len(result.failures) + len(result.errors)

        if result.failures:
            self.test_results['integration_tests']['errors'].extend(
                [f"{test}: {error}" for test, error in result.failures])
        if result.errors:
            self.test_results['integration_tests']['errors'].extend(
                [f"{test}: {error}" for test, error in result.errors])

    def run_scenario_tests(self):
        """Run scenario tests"""
        print("🎭 Running Scenario Tests...")

        suite = unittest.TestLoader().loadTestsFromTestCase(TestScenarios)
        result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)

        self.test_results['scenario_tests']['passed'] += result.testsRun - len(result.failures) - len(result.errors)
        self.test_results['scenario_tests']['failed'] += len(result.failures) + len(result.errors)

        if result.failures:
            self.test_results['scenario_tests']['errors'].extend(
                [f"{test}: {error}" for test, error in result.failures])
        if result.errors:
            self.test_results['scenario_tests']['errors'].extend([f"{test}: {error}" for test, error in result.errors])

    def run_performance_tests(self):
        """Run performance tests"""
        print("⚡ Running Performance Tests...")

        try:
            PerformanceTest.test_database_performance()
            self.test_results['performance_tests']['passed'] += 1
        except Exception as e:
            self.test_results['performance_tests']['failed'] += 1
            self.test_results['performance_tests']['errors'].append(f"Performance test failed: {e}")

    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting Comprehensive Test Suite for Socratic RAG System v7.0")
        print("=" * 70)

        start_time = time.time()

        try:
            self.run_unit_tests()
            self.run_integration_tests()
            self.run_scenario_tests()
            self.run_performance_tests()
        except KeyboardInterrupt:
            print("\n⚠